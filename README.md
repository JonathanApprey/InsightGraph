# InsightGraph: Agentic RAG Platform

> 🧠 A Smart Document Assistant powered by LangGraph agents that doesn't just retrieve—it **thinks**.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-FF6F00)
![Next.js](https://img.shields.io/badge/Next.js-16+-000000?logo=next.js&logoColor=white)

## ✨ Features

- **📄 Document Q&A (RAG)** — Upload PDFs/text and ask questions with context-aware answers
- **📊 Document Grading** — LLM evaluates retrieved documents for relevance
- **✏️ Query Rewriting** — Automatically improves queries that don't return good results
- **👁️ Transparent AI** — Watch the agent's reasoning in real-time via the "Brain" panel
- **⚡ Rate Limiting** — Built-in protection against API abuse
- **🔄 Retry Logic** — Robust error handling with automatic retries

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js)                       │
│  ┌──────────────────┐    ┌────────────────────────────────────┐ │
│  │   Chat Window    │    │   "Brain" Panel (Agent Steps)      │ │
│  │                  │    │   Retrieve → Grade → Generate      │ │
│  │                  │    │                                    │ │
│  └──────────────────┘    └────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ REST / SSE
┌─────────────────────────────────▼───────────────────────────────┐
│                         BACKEND (FastAPI)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LangGraph Agent                        │   │
│  │                                                           │   │
│  │                ┌───────────┐                             │   │
│  │                │  RETRIEVE │                             │   │
│  │                └─────┬─────┘                             │   │
│  │                      │                                   │   │
│  │                      ▼                                   │   │
│  │                ┌───────────┐                             │   │
│  │                │   GRADE   │                             │   │
│  │                └─────┬─────┘                             │   │
│  │                ┌─────┴─────┐                             │   │
│  │                ▼           ▼                             │   │
│  │          ┌─────────┐  ┌──────────┐                       │   │
│  │          │ REWRITE │  │ GENERATE │                       │   │
│  │          └────┬────┘  └──────────┘                       │   │
│  │               │                                          │   │
│  │               └──────▶ (back to RETRIEVE)                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │                    FAISS Vector Store                      │  │
│  │              HuggingFace Embeddings (Local)                │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer        | Technology                                                   |
|--------------|--------------------------------------------------------------|
| LLM          | HuggingFace (via `langchain-huggingface`)                    |
| Embeddings   | HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)       |
| Agent        | LangGraph (StateGraph with conditional routing)              |
| Vector DB    | FAISS (CPU)                                                  |
| Backend      | FastAPI + Uvicorn + SlowAPI (rate limiting)                  |
| Frontend     | Next.js 16 with App Router                                   |
| Containers   | Docker + Docker Compose                                      |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ (< 3.13)
- Node.js 18+
- Docker (optional, for containerized deployment)

### Backend Setup
```bash
cd backend
poetry install
cp .env.example .env  # Add your HuggingFace API key
poetry run uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Docker (Full Stack)
```bash
docker-compose up --build
```

## 📡 API Endpoints

| Method | Endpoint             | Rate Limit    | Description                              |
|--------|----------------------|---------------|------------------------------------------|
| GET    | `/api/status`        | None          | System status and configuration          |
| GET    | `/api/health`        | None          | Comprehensive health check               |
| POST   | `/api/upload`        | 10/min        | Upload PDF/text documents for ingestion  |
| POST   | `/api/chat`          | 30/min        | Send message & receive agent response    |
| POST   | `/api/chat/stream`   | 30/min        | Streaming chat with SSE                  |
| DELETE | `/api/documents`     | 5/min         | Clear all indexed documents              |

## 🧠 Agent Workflow

The InsightGraph agent follows this decision process:

1. **Retrieve** — Searches the vector store for relevant document chunks

2. **Grade** — LLM evaluates each document for relevance to the question

3. **Rewrite** (if needed) — Reformulates query if no relevant documents found

4. **Generate** — Synthesizes final answer using relevant context

## 🔒 Security Features

- **Rate Limiting** — Prevents API abuse (configurable per endpoint)
- **Input Validation** — Pydantic models with strict validation
- **File Size Limits** — Max 10MB per upload
- **CORS Protection** — Configurable allowed origins

## 📁 Project Structure

```
InsightGraph/
├── .github/workflows/ci.yml    # CI/CD pipeline
├── docker-compose.yml          # Multi-service orchestration
├── README.md                   # This file
├── backend/
│   ├── pyproject.toml          # Poetry dependencies
│   ├── Dockerfile
│   └── app/
│       ├── config.py           # Settings management
│       ├── ingest.py           # Document ingestion pipeline
│       ├── state.py            # LangGraph state schema
│       ├── graph.py            # Agentic workflow
│       └── main.py             # FastAPI endpoints
└── frontend/
    ├── package.json
    ├── Dockerfile
    └── app/
        ├── layout.tsx
        ├── page.tsx            # Chat + Brain visualization
        ├── globals.css         # Design system
        └── page.module.css     # Component styles
```

## 🎨 Design System

The frontend features a premium dark theme with:
- Glassmorphism effects
- Gradient accents
- Micro-animations
- Real-time agent step visualization
- Responsive layout

## 🧪 Testing

```bash
# Backend tests
cd backend
poetry run pytest -v

# Frontend linting
cd frontend
npm run lint
```

## 📄 License

MIT License - Built for portfolio demonstration.

---

**Built with ❤️ using LangGraph • HuggingFace • FastAPI • Next.js**
