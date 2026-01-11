# InsightGraph: An Agentic RAG Platform

## Overview

InsightGraph is an intelligent document assistant powered by LangGraph agents. Unlike traditional RAG systems that simply retrieve and generate, InsightGraph **thinks** about your documents.

## Key Features

### 1. Document Ingestion
Upload PDF, TXT, or Markdown files and InsightGraph will:
- Parse and extract content
- Split into optimal chunks (1000 chars with 200 overlap)
- Generate embeddings using HuggingFace's all-MiniLM-L6-v2
- Store in FAISS vector database for fast retrieval

### 2. Agentic Query Processing
When you ask a question, the LangGraph agent:
1. **Retrieves** relevant document chunks using semantic search
2. **Grades** each document for relevance using LLM evaluation
3. **Rewrites** the query if no relevant documents are found
4. **Generates** a grounded response with source citations

### 3. Transparent Reasoning
Watch the agent's "brain" in real-time as it processes your query. See exactly which nodes are activated and what decisions are made.

## Technical Architecture

- **Backend**: FastAPI with LangGraph for agentic workflows
- **Frontend**: Next.js 16 with real-time visualization
- **Embeddings**: HuggingFace Sentence Transformers (local)
- **Vector Store**: FAISS for fast similarity search
- **LLM**: HuggingFace Inference API

## Why InsightGraph?

Traditional RAG systems are "dumb pipes" - they retrieve and generate without any quality control. InsightGraph adds **intelligence** to the pipeline by:

1. Evaluating document relevance before generation
2. Automatically improving queries when results are poor
3. Providing full transparency into the reasoning process

This makes InsightGraph more reliable and trustworthy for real-world applications.

## Getting Started

1. Upload a document using the Upload button
2. Ask questions about the document content
3. Watch the Agent Brain panel to see the reasoning process
4. Review the answer with source citations

## Contact

Built by Jonathan Ekowapprey as a portfolio demonstration of agentic AI capabilities.
