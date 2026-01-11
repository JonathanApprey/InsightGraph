"use client";

import { useState, useRef, useEffect } from "react";
import styles from "./page.module.css";

// Types
interface AgentStep {
  node: string;
  action: string;
  result: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  thoughtProcess?: AgentStep[];
  sources?: string[];
}

interface ApiStatus {
  status: string;
  version: string;
  embedding_model: string;
  llm_model: string;
  documents_indexed: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeThoughtProcess, setActiveThoughtProcess] = useState<AgentStep[]>([]);
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check API status on mount
  useEffect(() => {
    checkApiStatus();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/status`);
      if (response.ok) {
        const data = await response.json();
        setApiStatus(data);
      }
    } catch (error) {
      console.error("Failed to fetch API status:", error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    setActiveThoughtProcess([]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();

      // Animate thought process
      for (const step of data.thought_process) {
        setActiveThoughtProcess((prev) => [...prev, step]);
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
        timestamp: new Date(),
        thoughtProcess: data.thought_process,
        sources: data.sources,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I apologize, but I encountered an error. Please ensure the backend is running and try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadSuccess(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      setUploadSuccess(`✓ Uploaded ${data.file_name} (${data.chunks_created} chunks)`);
      checkApiStatus();
    } catch (error) {
      console.error("Upload error:", error);
      setUploadSuccess("✗ Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const getNodeIcon = (node: string) => {
    switch (node) {
      case "retrieve":
        return "🔍";
      case "grade":
        return "📊";
      case "rewrite":
        return "✏️";
      case "generate":
        return "✨";
      case "error":
        return "⚠️";
      default:
        return "⚡";
    }
  };

  const getNodeColor = (node: string) => {
    switch (node) {
      case "retrieve":
        return "#3b82f6";
      case "grade":
        return "#f59e0b";
      case "rewrite":
        return "#8b5cf6";
      case "generate":
        return "#10b981";
      case "error":
        return "#ef4444";
      default:
        return "#6366f1";
    }
  };

  return (
    <main className={styles.main}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>🧠</span>
          <h1>InsightGraph</h1>
        </div>
        <p className={styles.tagline}>Agentic RAG Platform</p>

        {apiStatus && (
          <div className={styles.statusBadge}>
            <span className={styles.statusDot}></span>
            {apiStatus.documents_indexed} documents indexed
          </div>
        )}
      </header>

      {/* Main Content */}
      <div className={styles.content}>
        {/* Chat Panel */}
        <section className={styles.chatPanel}>
          <div className={styles.chatHeader}>
            <h2>💬 Chat</h2>
            <div className={styles.uploadSection}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.md,.doc,.docx"
                onChange={handleFileUpload}
                className={styles.fileInput}
                id="file-upload"
              />
              <label htmlFor="file-upload" className={`btn btn-secondary ${styles.uploadBtn}`}>
                {isUploading ? (
                  <span className="animate-spin">⏳</span>
                ) : (
                  <>📄 Upload Document</>
                )}
              </label>
            </div>
          </div>

          {uploadSuccess && (
            <div className={`${styles.uploadFeedback} animate-fadeIn`}>
              {uploadSuccess}
            </div>
          )}

          {/* Messages */}
          <div className={styles.messagesContainer}>
            {messages.length === 0 ? (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>🧠</div>
                <h3>Ready to think</h3>
                <p>Upload a document and ask questions. Watch the AI reason through your query.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${styles[message.role]} animate-fadeIn`}
                >
                  <div className={styles.messageAvatar}>
                    {message.role === "user" ? "👤" : "🤖"}
                  </div>
                  <div className={styles.messageContent}>
                    <p>{message.content}</p>
                    {message.sources && message.sources.length > 0 && (
                      <div className={styles.sources}>
                        <span>Sources:</span>
                        {message.sources.map((source, idx) => (
                          <span key={idx} className={styles.sourceTag}>
                            📄 {source}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className={`${styles.message} ${styles.assistant} animate-fadeIn`}>
                <div className={styles.messageAvatar}>🤖</div>
                <div className={styles.messageContent}>
                  <div className={styles.typingIndicator}>
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Form */}
          <form onSubmit={handleSubmit} className={styles.inputForm}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about your documents..."
              className="input"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isLoading || !inputValue.trim()}
            >
              {isLoading ? "Thinking..." : "Send"}
            </button>
          </form>
        </section>

        {/* Brain Panel */}
        <aside className={styles.brainPanel}>
          <div className={styles.brainHeader}>
            <h2>🧠 Agent Brain</h2>
            <p>Real-time reasoning visualization</p>
          </div>

          <div className={styles.brainContent}>
            {activeThoughtProcess.length === 0 ? (
              <div className={styles.brainEmpty}>
                <div className={styles.brainPlaceholder}>
                  <div className={styles.node}>🔍 Retrieve</div>
                  <div className={styles.nodeArrow}>↓</div>
                  <div className={styles.node}>📊 Grade</div>
                  <div className={styles.nodeArrow}>↓</div>
                  <div className={styles.node}>✨ Generate</div>
                </div>
                <p>Send a message to see the agent&apos;s thought process</p>
              </div>
            ) : (
              <div className={styles.thoughtProcess}>
                {activeThoughtProcess.map((step, idx) => (
                  <div
                    key={idx}
                    className={`${styles.thoughtStep} animate-fadeIn`}
                    style={{
                      borderLeftColor: getNodeColor(step.node),
                      animationDelay: `${idx * 100}ms`
                    }}
                  >
                    <div className={styles.stepHeader}>
                      <span className={styles.stepIcon}>{getNodeIcon(step.node)}</span>
                      <span
                        className={styles.stepNode}
                        style={{ color: getNodeColor(step.node) }}
                      >
                        {step.node.toUpperCase()}
                      </span>
                    </div>
                    <div className={styles.stepAction}>{step.action}</div>
                    <div className={styles.stepResult}>{step.result}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Graph Visualization */}
          <div className={styles.graphSection}>
            <h3>📊 Agent Flow</h3>
            <div className={styles.graph}>
              {/* Retrieve Node */}
              <div
                className={`${styles.graphNode} ${activeThoughtProcess.some(s => s.node === "retrieve") ? styles.active : ""}`}
                style={{ top: "10%", left: "50%" }}
              >
                🔍 Retrieve
              </div>
              {/* Grade Node */}
              <div
                className={`${styles.graphNode} ${activeThoughtProcess.some(s => s.node === "grade") ? styles.active : ""}`}
                style={{ top: "35%", left: "50%" }}
              >
                📊 Grade
              </div>
              {/* Rewrite Node */}
              <div
                className={`${styles.graphNode} ${activeThoughtProcess.some(s => s.node === "rewrite") ? styles.active : ""}`}
                style={{ top: "60%", left: "25%" }}
              >
                ✏️ Rewrite
              </div>
              {/* Generate Node */}
              <div
                className={`${styles.graphNode} ${activeThoughtProcess.some(s => s.node === "generate") ? styles.active : ""}`}
                style={{ top: "60%", left: "75%" }}
              >
                ✨ Generate
              </div>

              {/* Edges */}
              <svg className={styles.graphEdges}>
                <defs>
                  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1" />
                  </marker>
                </defs>
                {/* Retrieve -> Grade */}
                <line x1="50%" y1="18%" x2="50%" y2="32%" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowhead)" />
                {/* Grade -> Rewrite */}
                <line x1="42%" y1="42%" x2="32%" y2="57%" stroke="#f59e0b" strokeWidth="2" markerEnd="url(#arrowhead)" />
                {/* Grade -> Generate */}
                <line x1="58%" y1="42%" x2="68%" y2="57%" stroke="#f59e0b" strokeWidth="2" markerEnd="url(#arrowhead)" />
                {/* Rewrite -> Retrieve */}
                <line x1="18%" y1="60%" x2="12%" y2="18%" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="5,5" markerEnd="url(#arrowhead)" />
              </svg>
            </div>
          </div>
        </aside>
      </div>

      {/* Footer */}
      <footer className={styles.footer}>
        <p>Built with LangGraph • HuggingFace • FastAPI • Next.js</p>
      </footer>
    </main>
  );
}
