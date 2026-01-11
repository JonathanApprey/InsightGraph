"""
Document Ingestion Pipeline for InsightGraph

Handles PDF/text loading, chunking, embedding, and vector store upsertion.
Uses HuggingFace embeddings (not OpenAI) as per requirements.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_settings

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Pipeline for ingesting documents into the FAISS vector store.
    
    Workflow:
    1. Load document (PDF, TXT, or other formats)
    2. Split into chunks using RecursiveCharacterTextSplitter
    3. Generate embeddings using HuggingFace sentence-transformers
    4. Upsert into FAISS vector store
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[FAISS] = None
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load HuggingFace embeddings model."""
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"device": "cpu"},  # Use CPU for portability
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully")
        return self._embeddings
    
    @property
    def vectorstore(self) -> FAISS:
        """Get or create FAISS vector store."""
        if self._vectorstore is None:
            index_path = self.settings.faiss_index_dir
            index_file = index_path / "index.faiss"
            
            if index_file.exists():
                logger.info(f"Loading existing FAISS index from {index_path}")
                self._vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                logger.info("Creating new FAISS index")
                # Create empty index with a placeholder document
                placeholder_doc = Document(
                    page_content="InsightGraph Vector Store Initialized",
                    metadata={"source": "system", "type": "placeholder"},
                )
                self._vectorstore = FAISS.from_documents(
                    [placeholder_doc],
                    self.embeddings,
                )
                self._save_index()
        
        return self._vectorstore
    
    def _save_index(self) -> None:
        """Persist FAISS index to disk."""
        if self._vectorstore:
            index_path = self.settings.faiss_index_dir
            logger.info(f"Saving FAISS index to {index_path}")
            self._vectorstore.save_local(str(index_path))
    
    def _load_document(self, file_path: Path) -> list[Document]:
        """Load document based on file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(str(file_path))
        else:
            # Fallback to text loader for other formats
            loader = TextLoader(str(file_path))
        
        logger.info(f"Loading document: {file_path}")
        return loader.load()
    
    def ingest_document(self, file_path: str | Path) -> dict:
        """
        Ingest a document into the vector store.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            dict with ingestion statistics
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Step 1: Load document
        documents = self._load_document(file_path)
        logger.info(f"Loaded {len(documents)} pages/sections from {file_path.name}")
        
        # Step 2: Split into chunks
        chunks = self._text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source_file"] = file_path.name
        
        # Step 3: Add to vector store (embeddings generated automatically)
        self.vectorstore.add_documents(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector store")
        
        # Step 4: Persist index
        self._save_index()
        
        return {
            "status": "success",
            "file_name": file_path.name,
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
        }
    
    def get_retriever(self, k: int = 4):
        """Get a retriever for the vector store."""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
    
    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Perform similarity search on the vector store."""
        return self.vectorstore.similarity_search(query, k=k)


# Global pipeline instance
_pipeline: Optional[DocumentIngestionPipeline] = None


def get_pipeline() -> DocumentIngestionPipeline:
    """Get or create the global ingestion pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentIngestionPipeline()
    return _pipeline
