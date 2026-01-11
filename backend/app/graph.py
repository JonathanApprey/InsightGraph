"""
LangGraph Agentic Workflow for InsightGraph

Implements a stateful agent that:
1. Retrieves documents from vector store
2. Grades document relevance
3. Rewrites query if documents are irrelevant
4. Generates response using relevant context

Uses HuggingFace models (not OpenAI) as per requirements.
"""

import logging
from typing import Literal, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langgraph.graph import END, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_settings
from .ingest import get_pipeline
from .state import AgentState, AgentStep

logger = logging.getLogger(__name__)

# Maximum query rewrite iterations to prevent infinite loops
MAX_REWRITE_ITERATIONS = 2

# LLM instance cache
_llm_instance: Optional[HuggingFaceEndpoint] = None


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


def get_llm():
    """
    Get the HuggingFace LLM instance with caching.
    
    Uses HuggingFaceEndpoint for API-based inference.
    """
    global _llm_instance
    
    if _llm_instance is not None:
        return _llm_instance
    
    settings = get_settings()
    
    if settings.huggingface_api_key:
        # Use HuggingFace Inference API
        logger.info(f"Initializing HuggingFace LLM: {settings.llm_model}")
        _llm_instance = HuggingFaceEndpoint(
            repo_id=settings.llm_model,
            huggingfacehub_api_token=settings.huggingface_api_key,
            temperature=0.7,
            max_new_tokens=512,
        )
        return _llm_instance
    else:
        # Fallback to local inference (requires more resources)
        logger.warning("No HuggingFace API key found, using local pipeline")
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model=settings.llm_model,
            max_new_tokens=512,
            temperature=0.7,
        )
        _llm_instance = HuggingFacePipeline(pipeline=pipe)
        return _llm_instance


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def invoke_llm_with_retry(chain, inputs: dict) -> str:
    """Invoke LLM chain with retry logic."""
    try:
        result = chain.invoke(inputs)
        return result.strip() if isinstance(result, str) else str(result)
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise LLMError(f"LLM call failed: {e}") from e


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def retrieve_node(state: AgentState) -> AgentState:
    """
    Node: Retrieve documents from vector store.
    
    Uses the current query (original or rewritten) to fetch relevant documents.
    """
    logger.info("=== RETRIEVE NODE ===")
    
    # Use rewritten query if available, otherwise use original question
    query = state.get("rewritten_query") or state["question"]
    
    try:
        pipeline = get_pipeline()
        documents = pipeline.similarity_search(query, k=4)
        
        # Filter out placeholder documents
        documents = [
            doc for doc in documents 
            if doc.metadata.get("type") != "placeholder"
        ]
        
        logger.info(f"Retrieved {len(documents)} documents from vector store")
        
    except Exception as e:
        logger.error(f"Vector store retrieval failed: {e}")
        documents = []
    
    # Add step to thought process
    step = AgentStep(
        node="retrieve",
        action=f"Searched documents for: '{query}'",
        result=f"Found {len(documents)} relevant document chunks",
    )
    
    return {
        "question": state["question"],
        "documents": documents,
        "doc_grades": state.get("doc_grades", []),
        "rewritten_query": state.get("rewritten_query", ""),
        "rewrite_count": state.get("rewrite_count", 0),
        "generation": state.get("generation", ""),
        "thought_process": state.get("thought_process", []) + [step],
        "should_rewrite": False,
    }


def grade_documents_node(state: AgentState) -> AgentState:
    """
    Node: Grade retrieved documents for relevance.
    
    Uses LLM to evaluate if each document is relevant to the question.
    """
    logger.info("=== GRADE NODE ===")
    
    documents = state.get("documents", [])
    question = state["question"]
    
    if not documents:
        # No documents to grade
        step = AgentStep(
            node="grade",
            action="Evaluating document relevance",
            result="No documents found - will generate response without context",
        )
        return {
            "question": state["question"],
            "documents": [],
            "doc_grades": [],
            "rewritten_query": state.get("rewritten_query", ""),
            "rewrite_count": state.get("rewrite_count", 0),
            "generation": state.get("generation", ""),
            "thought_process": state.get("thought_process", []) + [step],
            "should_rewrite": False,
        }
    
    try:
        # Grade each document
        llm = get_llm()
        
        grade_prompt = PromptTemplate(
            template="""You are a document relevance grader. Given a user question and a document chunk, 
determine if the document is relevant to answering the question.

Question: {question}

Document:
{document}

Is this document relevant to the question? Answer with just 'yes' or 'no'.
Answer:""",
            input_variables=["question", "document"],
        )
        
        grader_chain = grade_prompt | llm | StrOutputParser()
        
        relevant_docs = []
        grades = []
        
        for doc in documents:
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            try:
                grade_result = invoke_llm_with_retry(grader_chain, {
                    "question": question,
                    "document": doc.page_content,
                })
                
                is_relevant = "yes" in grade_result.lower()
                
                grades.append({
                    "content_preview": content_preview,
                    "source": doc.metadata.get("source_file", doc.metadata.get("source", "unknown")),
                    "is_relevant": is_relevant,
                    "reason": grade_result.strip(),
                })
                
                if is_relevant:
                    relevant_docs.append(doc)
                    
            except Exception as e:
                logger.error(f"Error grading document: {e}")
                # On error, include the document (fail-safe)
                relevant_docs.append(doc)
                grades.append({
                    "content_preview": content_preview,
                    "source": doc.metadata.get("source_file", doc.metadata.get("source", "unknown")),
                    "is_relevant": True,
                    "reason": "Included due to grading error",
                })
        
        relevant_count = len(relevant_docs)
        total_count = len(documents)
        
    except Exception as e:
        logger.error(f"Grading process failed: {e}")
        # On total failure, use all documents
        relevant_docs = documents
        relevant_count = len(documents)
        total_count = len(documents)
        grades = []
    
    # Decide if we need to rewrite
    rewrite_count = state.get("rewrite_count", 0)
    should_rewrite = relevant_count == 0 and rewrite_count < MAX_REWRITE_ITERATIONS
    
    result_msg = f"{relevant_count}/{total_count} documents are relevant"
    if should_rewrite:
        result_msg += " - will rewrite query"
    
    step = AgentStep(
        node="grade",
        action=f"Evaluated {total_count} documents for relevance",
        result=result_msg,
    )
    
    logger.info(f"Graded {total_count} docs, {relevant_count} relevant, should_rewrite={should_rewrite}")
    
    return {
        "question": state["question"],
        "documents": relevant_docs,
        "doc_grades": grades,
        "rewritten_query": state.get("rewritten_query", ""),
        "rewrite_count": state.get("rewrite_count", 0),
        "generation": state.get("generation", ""),
        "thought_process": state.get("thought_process", []) + [step],
        "should_rewrite": should_rewrite,
    }


def rewrite_query_node(state: AgentState) -> AgentState:
    """
    Node: Rewrite the query to improve retrieval.
    
    Called when graded documents are not relevant.
    """
    logger.info("=== REWRITE NODE ===")
    
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0) + 1
    
    try:
        llm = get_llm()
        
        rewrite_prompt = PromptTemplate(
            template="""You are a query rewriter. The original question did not return relevant documents.
Rewrite the question to be more specific and likely to find relevant information.

Original Question: {question}

Rewritten Question (be more specific, use different keywords):""",
            input_variables=["question"],
        )
        
        rewriter_chain = rewrite_prompt | llm | StrOutputParser()
        
        rewritten = invoke_llm_with_retry(rewriter_chain, {"question": question})
        
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        rewritten = question  # Fallback to original
    
    step = AgentStep(
        node="rewrite",
        action="Reformulating query for better results",
        result=f"New query: '{rewritten}'",
    )
    
    logger.info(f"Rewritten query (attempt {rewrite_count}): {rewritten}")
    
    return {
        "question": state["question"],
        "documents": state.get("documents", []),
        "doc_grades": state.get("doc_grades", []),
        "rewritten_query": rewritten,
        "rewrite_count": rewrite_count,
        "generation": state.get("generation", ""),
        "thought_process": state.get("thought_process", []) + [step],
        "should_rewrite": False,
    }


def generate_node(state: AgentState) -> AgentState:
    """
    Node: Generate the final response using relevant context.
    
    Uses RAG pattern - combines retrieved documents with the question
    to generate a grounded, accurate response.
    """
    logger.info("=== GENERATE NODE ===")
    
    question = state["question"]
    documents = state.get("documents", [])
    
    try:
        llm = get_llm()
        
        # Format context from documents
        if documents:
            context_parts = []
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
                context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "No relevant documents found. Please provide a general response based on your knowledge."
        
        generate_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context.
Use the information from the context to answer. If the context doesn't contain relevant information,
acknowledge this and provide what help you can. Be concise but thorough. Always cite your sources when using context."""),
            ("human", """Context:
{context}

Question: {question}

Answer:"""),
        ])
        
        generator_chain = generate_prompt | llm | StrOutputParser()
        
        response = invoke_llm_with_retry(generator_chain, {
            "context": context,
            "question": question,
        })
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response = f"I apologize, but I encountered an error generating a response. Error: {str(e)}. Please try again."
    
    step = AgentStep(
        node="generate",
        action="Synthesizing answer from relevant context",
        result=f"Generated response ({len(response)} chars)",
    )
    
    logger.info(f"Generated response: {response[:100]}...")
    
    return {
        "question": state["question"],
        "documents": state.get("documents", []),
        "doc_grades": state.get("doc_grades", []),
        "rewritten_query": state.get("rewritten_query", ""),
        "rewrite_count": state.get("rewrite_count", 0),
        "generation": response,
        "thought_process": state.get("thought_process", []) + [step],
        "should_rewrite": False,
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_rewrite_or_generate(state: AgentState) -> Literal["rewrite", "generate"]:
    """
    Conditional edge: Decide whether to rewrite query or generate response.
    """
    if state.get("should_rewrite", False):
        return "rewrite"
    return "generate"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph() -> StateGraph:
    """
    Build the LangGraph agent workflow.
    
    Graph Structure:
    
        [START]
           │
           ▼
       ┌─────────┐
       │ RETRIEVE │
       └─────┬───┘
             │
             ▼
       ┌─────────┐
       │  GRADE  │
       └─────┬───┘
             │
         ┌───┴───┐
         ▼       ▼
    ┌────────┐ ┌────────┐
    │REWRITE │ │GENERATE│
    └────┬───┘ └────┬───┘
         │          │
         │          ▼
         │       [END]
         │
         └──────► (back to RETRIEVE)
    """
    
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add edges
    workflow.add_edge("retrieve", "grade")
    
    # Conditional edge from grade
    workflow.add_conditional_edges(
        "grade",
        should_rewrite_or_generate,
        {
            "rewrite": "rewrite",
            "generate": "generate",
        },
    )
    
    # Rewrite loops back to retrieve
    workflow.add_edge("rewrite", "retrieve")
    
    # Generate ends the workflow
    workflow.add_edge("generate", END)
    
    return workflow


def get_compiled_graph():
    """Get the compiled LangGraph for execution."""
    workflow = build_graph()
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_agent(question: str) -> dict:
    """
    Execute the agent workflow for a given question.
    
    Args:
        question: The user's question
        
    Returns:
        dict with response, thought_process, and sources
    """
    logger.info(f"Running agent for question: {question}")
    
    # Validate input
    if not question or not question.strip():
        return {
            "response": "Please provide a valid question.",
            "thought_process": [],
            "sources": [],
        }
    
    # Initialize state
    initial_state: AgentState = {
        "question": question.strip(),
        "documents": [],
        "doc_grades": [],
        "rewritten_query": "",
        "rewrite_count": 0,
        "generation": "",
        "thought_process": [],
        "should_rewrite": False,
    }
    
    try:
        # Get compiled graph
        graph = get_compiled_graph()
        
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        # Extract sources
        sources = list(set([
            doc.metadata.get("source_file", doc.metadata.get("source", "unknown"))
            for doc in final_state.get("documents", [])
        ]))
        
        return {
            "response": final_state.get("generation", ""),
            "thought_process": final_state.get("thought_process", []),
            "sources": sources,
        }
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            "response": f"I apologize, but I encountered an error processing your request: {str(e)}",
            "thought_process": [
                AgentStep(
                    node="error",
                    action="Processing failed",
                    result=str(e),
                )
            ],
            "sources": [],
        }
