"""
FastAPI Application Entry Point for InsightGraph (Enhanced)

Provides REST API endpoints with:
- Rate limiting
- Comprehensive error handling
- Health checks
- Request validation
"""

import logging
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from .config import get_settings
from .graph import run_agent
from .ingest import get_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting InsightGraph API...")
    
    # Pre-load the pipeline (embeddings model)
    try:
        pipeline = get_pipeline()
        _ = pipeline.embeddings  # Trigger lazy loading
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize pipeline on startup: {e}")
    
    logger.info("InsightGraph API ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down InsightGraph API...")


# Initialize FastAPI app
app = FastAPI(
    title="InsightGraph API",
    description="Agentic RAG Platform - Smart Document Assistant with Web Search Fallback",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploads directory
UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred. Please try again.",
            "error_type": type(exc).__name__,
        },
    )


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request body for chat endpoint with validation."""
    message: str = Field(..., min_length=1, max_length=10000)
    history: list[dict] = Field(default_factory=list)
    
    @validator("message")
    def validate_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    response: str
    thought_process: list[dict]
    sources: list[str]
    processing_time_ms: int


class UploadResponse(BaseModel):
    """Response body for upload endpoint."""
    status: str
    file_name: str
    pages_loaded: int
    chunks_created: int
    processing_time_ms: int


class StatusResponse(BaseModel):
    """Response body for status endpoint."""
    status: str
    version: str
    embedding_model: str
    llm_model: str
    documents_indexed: int
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    checks: dict[str, bool]


class AgentStepEvent(BaseModel):
    """SSE event for agent step updates."""
    node: str
    action: str
    result: str


# Track startup time
_startup_time = time.time()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """
    Health check and system status endpoint.
    
    Returns current configuration and indexed document count.
    """
    try:
        pipeline = get_pipeline()
        doc_count = len(pipeline.vectorstore.docstore._dict) if pipeline._vectorstore else 0
    except Exception as e:
        logger.warning(f"Could not get document count: {e}")
        doc_count = 0
    
    return StatusResponse(
        status="healthy",
        version="1.1.0",
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        documents_indexed=doc_count,
        uptime_seconds=round(time.time() - _startup_time, 2),
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Checks all critical system components.
    """
    checks = {
        "api": True,
        "embeddings": False,
        "vector_store": False,
    }
    
    try:
        pipeline = get_pipeline()
        
        # Check embeddings
        try:
            _ = pipeline.embeddings
            checks["embeddings"] = True
        except Exception as e:
            logger.warning(f"Embeddings check failed: {e}")
        
        # Check vector store
        try:
            _ = pipeline.vectorstore
            checks["vector_store"] = True
        except Exception as e:
            logger.warning(f"Vector store check failed: {e}")
            
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
    
    all_healthy = all(checks.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        checks=checks,
    )


@app.post("/api/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload and ingest a document into the vector store.
    
    Rate limited to 10 uploads per minute.
    Supports PDF and text files. Documents are chunked, embedded,
    and stored in FAISS for retrieval.
    """
    start_time = time.time()
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )
    
    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )
    
    # Validate file size (max 10MB)
    MAX_SIZE = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {MAX_SIZE // (1024*1024)}MB",
        )
    
    # Save uploaded file
    file_id = str(uuid4())[:8]
    safe_filename = f"{file_id}_{file.filename}"
    file_path = UPLOADS_DIR / safe_filename
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Ingest document
        pipeline = get_pipeline()
        result = pipeline.ingest_document(file_path)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return UploadResponse(
            **result,
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """
    Chat with the document assistant.
    
    Rate limited to 30 messages per minute.
    
    Runs the LangGraph agent workflow:
    1. Routes query to best source
    2. Retrieves documents (vector store or web)
    3. Grades document relevance
    4. Rewrites query if needed
    5. Generates response
    
    Returns the response along with the thought process for transparency.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing chat request: {chat_request.message[:50]}...")
        
        # Run the agent
        result = run_agent(chat_request.message)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return ChatResponse(
            response=result["response"],
            thought_process=result["thought_process"],
            sources=result["sources"],
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}",
        )


@app.post("/api/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, chat_request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events.
    
    Rate limited to 30 messages per minute.
    Sends agent step updates as they happen, followed by the final response.
    This enables the "Brain" panel to show real-time progress.
    """
    async def generate_events():
        try:
            start_time = time.time()
            
            # Run agent and yield steps
            result = run_agent(chat_request.message)
            
            # Yield each thought process step
            for step in result["thought_process"]:
                yield {
                    "event": "step",
                    "data": step,
                }
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Yield final response
            yield {
                "event": "response",
                "data": {
                    "response": result["response"],
                    "sources": result["sources"],
                    "processing_time_ms": processing_time,
                },
            }
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": {"error": str(e)},
            }
    
    return EventSourceResponse(generate_events())


@app.delete("/api/documents")
@limiter.limit("5/minute")
async def clear_documents(request: Request):
    """
    Clear all indexed documents.
    
    Rate limited to 5 requests per minute.
    This resets the vector store to empty.
    """
    try:
        # Clear uploads directory
        for file in UPLOADS_DIR.glob("*"):
            if file.is_file():
                file.unlink()
        
        # Clear vector store
        pipeline = get_pipeline()
        faiss_path = settings.faiss_index_dir
        
        if faiss_path.exists():
            shutil.rmtree(faiss_path)
        
        # Reset pipeline
        pipeline._vectorstore = None
        
        logger.info("Cleared all documents")
        
        return {"status": "success", "message": "All documents cleared"}
        
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear documents: {str(e)}",
        )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "InsightGraph API",
        "description": "Agentic RAG Platform - Smart Document Assistant with Web Search",
        "version": "1.1.0",
        "features": [
            "Document Q&A with RAG",
            "Agentic query routing",
            "Web search fallback",
            "Intelligent document grading",
            "Query rewriting",
            "Real-time thought process visualization",
        ],
        "docs": "/docs",
        "endpoints": {
            "status": "GET /api/status",
            "health": "GET /api/health",
            "upload": "POST /api/upload",
            "chat": "POST /api/chat",
            "chat_stream": "POST /api/chat/stream",
            "clear_documents": "DELETE /api/documents",
        },
    }
