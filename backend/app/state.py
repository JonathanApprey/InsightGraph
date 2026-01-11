"""
LangGraph Agent State and Schema Definitions

Defines the state that flows through the agentic graph.
"""

from operator import add
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class AgentStep(TypedDict):
    """Represents a single step in the agent's reasoning process."""
    node: str
    action: str
    result: str


class AgentState(TypedDict):
    """
    State schema for the InsightGraph agent.
    
    This state flows through all nodes in the LangGraph and tracks:
    - The user's question
    - Retrieved documents
    - Document relevance grades
    - The generated answer
    - The full thought process for transparency
    """
    
    # User input
    question: str
    
    # Retrieved context
    documents: list[Document]
    
    # Document relevance (per-document grading)
    doc_grades: list[dict]
    
    # Query rewriting
    rewritten_query: str
    rewrite_count: int
    
    # Generation
    generation: str
    
    # Thought process tracking (for UI "Brain" panel)
    thought_process: Annotated[list[AgentStep], add]
    
    # Flow control
    should_rewrite: bool


class DocumentGrade(TypedDict):
    """Grade for a single document's relevance."""
    content_preview: str
    source: str
    is_relevant: bool
    reason: str


class ChatRequest(TypedDict):
    """Request schema for chat endpoint."""
    message: str
    history: list[dict]


class ChatResponse(TypedDict):
    """Response schema for chat endpoint."""
    response: str
    thought_process: list[AgentStep]
    sources: list[str]
