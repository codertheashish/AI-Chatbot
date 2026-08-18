"""
=====================================================================================
 AI Chatbot — Production-ready single-file backend (FastAPI + OpenRouter + FAISS/RAG)
=====================================================================================

Everything lives in this one file on purpose (per project requirements):
  - FastAPI app + routing (HTML pages, REST APIs, streaming chat)
  - SQLite persistence (chats, messages, documents)
  - File ingestion (PDF / TXT / DOCX) -> chunking -> embeddings -> vector store
  - A lightweight FAISS-backed (with automatic NumPy fallback) vector store for RAG
  - OpenRouter streaming chat completions (Server-Sent Events to the browser)
  - Conversation memory (per-chat message history pulled from SQLite)

DEPLOYMENT NOTE (read this):
  Vercel Serverless Functions have a read-only filesystem except for `/tmp`, and
  `/tmp` is EPHEMERAL (wiped between cold starts / across instances). This app
  therefore automatically switches its SQLite DB, uploads folder and vector index
  to `/tmp` when the `VERCEL` env var is present, so the app *runs* correctly on
  Vercel. However this also means data (chat history, uploaded documents) is NOT
  guaranteed to persist long-term on Vercel's free serverless tier -- for real
  persistence in production, point DB_PATH / VECTOR_DB_DIR at a mounted volume,
  or swap SQLite -> a hosted Postgres and the local vector store -> a hosted
  vector DB (e.g. Pinecone/Qdrant). Locally (or on a normal VM/container) storage
  is fully persistent on disk. This tradeoff is intentional and documented so the
  app is deployable "as is" while being honest about serverless storage limits.
"""

import os
import re
import io
import json
import uuid
import sqlite3
import hashlib
import asyncio
import contextlib
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator

import httpx
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Optional (best-effort) document parsers -- degrade gracefully if not installed
try:
    from pypdf import PdfReader
    HAVE_PYPDF = True
except ImportError:
    HAVE_PYPDF = False

try:
    import docx  # python-docx
    HAVE_DOCX = True
except ImportError:
    HAVE_DOCX = False

# Optional FAISS -- falls back to a pure NumPy cosine-similarity index if missing
# (keeps the app deployable even where the faiss-cpu wheel is too large / unavailable)
try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False


# ======================================================================================
# Configuration
# ======================================================================================

load_dotenv()

ON_VERCEL = bool(os.environ.get("VERCEL"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vercel's filesystem is read-only except /tmp -> redirect writable paths there.
RUNTIME_DIR = "/tmp" if ON_VERCEL else BASE_DIR
DB_PATH = os.path.join(RUNTIME_DIR, "database.db")
UPLOAD_DIR = os.path.join(RUNTIME_DIR, "uploads")
VECTOR_DB_DIR = os.path.join(RUNTIME_DIR, "vector_db")
VECTOR_STORE_FILE = os.path.join(VECTOR_DB_DIR, "store.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("MODEL", "openai/gpt-4.1-mini")

# Models offered in the Settings dropdown (OpenRouter model ids). Free-tier models are
# listed first since many users start on OpenRouter's free credits.
AVAILABLE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large",
]
# Always guarantee whatever MODEL is set in .env actually appears (and is selected)
# in the dropdown, even if it's not in the curated list above -- this is what fixes
# the dropdown not "remembering"/matching your configured default on page load.
if DEFAULT_MODEL not in AVAILABLE_MODELS:
    AVAILABLE_MODELS.insert(0, DEFAULT_MODEL)

MAX_HISTORY_MESSAGES = 20      # how many past messages to feed back as memory
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 2048))  # cap reply length (avoids "insufficient credits" errors)
CHUNK_SIZE = 900                # characters per RAG chunk
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 4                # how many chunks to retrieve per query
EMBED_DIM = 384                 # dimensionality of the local hashing embedding

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable AI assistant in a ChatGPT-style chat app. "
    "Be concise, accurate, and format answers with Markdown (code blocks, lists, "
    "bold) where it helps readability. If context from uploaded documents is "
    "provided below, ground your answer in it and cite sources using the given "
    "[Source: filename] tags; if the context doesn't contain the answer, say so "
    "and answer from general knowledge instead."
)


# ======================================================================================
# Database layer (SQLite)
# ======================================================================================

def get_db() -> sqlite3.Connection:
    """Open a new SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables if they don't already exist."""
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT 'New Chat',
            model       TEXT NOT NULL DEFAULT 'openai/gpt-4.1-mini',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT NOT NULL,
            role        TEXT NOT NULL,               -- 'user' | 'assistant' | 'system'
            content     TEXT NOT NULL,
            sources     TEXT,                         -- JSON-encoded citation list
            created_at  TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            chat_id     TEXT NOT NULL,
            filename    TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# ======================================================================================
# Lightweight local embeddings (no external API calls / no heavy ML deps)
# ----------------------------------------------------------------------
# A deterministic hashing-vectorizer + TF weighting scheme. It's not as strong as a
# transformer embedding model, but it needs zero extra dependencies or network calls,
# keeps cold starts fast on serverless, and works well enough for keyword/semantic-ish
# retrieval over a user's own uploaded documents. Swap `embed_text()` for a call to
# OpenRouter/OpenAI embeddings (or sentence-transformers) if you need higher recall.
# ======================================================================================

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def embed_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Turn text into a fixed-size vector via hashed term-frequency weighting."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ======================================================================================
# Vector store (FAISS if available, else a NumPy cosine-similarity fallback)
# ======================================================================================

class VectorStore:
    """A small persisted vector index storing (chat_id, doc_id, filename, text, vector)."""

    def __init__(self):
        self.ids: List[str] = []
        self.chat_ids: List[str] = []
        self.doc_ids: List[str] = []
        self.filenames: List[str] = []
        self.texts: List[str] = []
        self.vectors: Optional[np.ndarray] = None  # shape (N, EMBED_DIM)
        self._faiss_index = None
        self.load()

    # -- persistence -------------------------------------------------------------
    def load(self) -> None:
        if not os.path.exists(VECTOR_STORE_FILE):
            return
        try:
            with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ids = data.get("ids", [])
            self.chat_ids = data.get("chat_ids", [])
            self.doc_ids = data.get("doc_ids", [])
            self.filenames = data.get("filenames", [])
            self.texts = data.get("texts", [])
            vecs = data.get("vectors", [])
            self.vectors = np.array(vecs, dtype=np.float32) if vecs else None
            self._rebuild_faiss()
        except (json.JSONDecodeError, OSError):
            # Corrupt or missing store -> start fresh rather than crash the app
            self.ids, self.chat_ids, self.doc_ids = [], [], []
            self.filenames, self.texts, self.vectors = [], [], None

    def persist(self) -> None:
        data = {
            "ids": self.ids,
            "chat_ids": self.chat_ids,
            "doc_ids": self.doc_ids,
            "filenames": self.filenames,
            "texts": self.texts,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        with open(VECTOR_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # -- index maintenance --------------------------------------------------------
    def _rebuild_faiss(self) -> None:
        if not HAVE_FAISS or self.vectors is None or len(self.vectors) == 0:
            self._faiss_index = None
            return
        index = faiss.IndexFlatIP(EMBED_DIM)  # inner product on normalized vectors = cosine
        index.add(self.vectors.astype(np.float32))
        self._faiss_index = index

    def add(self, chat_id: str, doc_id: str, filename: str, chunks: List[str]) -> int:
        new_vecs = np.stack([embed_text(c) for c in chunks]) if chunks else np.zeros((0, EMBED_DIM))
        for c in chunks:
            self.ids.append(str(uuid.uuid4()))
            self.chat_ids.append(chat_id)
            self.doc_ids.append(doc_id)
            self.filenames.append(filename)
            self.texts.append(c)
        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = new_vecs
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])
        self._rebuild_faiss()
        self.persist()
        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        keep = [i for i, d in enumerate(self.doc_ids) if d != doc_id]
        self._filter_to(keep)

    def delete_chat(self, chat_id: str) -> None:
        keep = [i for i, c in enumerate(self.chat_ids) if c != chat_id]
        self._filter_to(keep)

    def _filter_to(self, keep_idx: List[int]) -> None:
        self.ids = [self.ids[i] for i in keep_idx]
        self.chat_ids = [self.chat_ids[i] for i in keep_idx]
        self.doc_ids = [self.doc_ids[i] for i in keep_idx]
        self.filenames = [self.filenames[i] for i in keep_idx]
        self.texts = [self.texts[i] for i in keep_idx]
        self.vectors = self.vectors[keep_idx] if (self.vectors is not None and len(keep_idx)) else None
        self._rebuild_faiss()
        self.persist()

    # -- search ---------------------------------------------------------------
    def search(self, chat_id: str, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict[str, Any]]:
        """Return the top_k most similar chunks scoped to a given chat."""
        local_idx = [i for i, c in enumerate(self.chat_ids) if c == chat_id]
        if not local_idx or self.vectors is None:
            return []

        q = embed_text(query).astype(np.float32)
        sub_vectors = self.vectors[local_idx]

        if HAVE_FAISS and len(local_idx) > 0:
            index = faiss.IndexFlatIP(EMBED_DIM)
            index.add(sub_vectors)
            k = min(top_k, len(local_idx))
            scores, ids = index.search(q.reshape(1, -1), k)
            scores, ids = scores[0], ids[0]
        else:
            sims = sub_vectors @ q  # cosine similarity (vectors are pre-normalized)
            k = min(top_k, len(local_idx))
            ids = np.argsort(-sims)[:k]
            scores = sims[ids]

        results = []
        for score, i in zip(scores, ids):
            if i < 0:
                continue
            real_idx = local_idx[int(i)]
            results.append(
                {
                    "text": self.texts[real_idx],
                    "filename": self.filenames[real_idx],
                    "score": float(score),
                }
            )
        # Only keep reasonably relevant matches
        return [r for r in results if r["score"] > 0.05]


vector_store = VectorStore()


# ======================================================================================
# Document parsing + chunking
# ======================================================================================

def extract_text_from_file(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded PDF / DOCX / TXT file."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        if not HAVE_PYPDF:
            raise HTTPException(500, "pypdf is not installed on the server")
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        if not HAVE_DOCX:
            raise HTTPException(500, "python-docx is not installed on the server")
        document = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in document.paragraphs)

    if ext == "txt":
        return raw.decode("utf-8", errors="ignore")

    raise HTTPException(400, f"Unsupported file type: .{ext}. Use PDF, DOCX or TXT.")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks on paragraph/sentence-friendly boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # try to break on a sentence boundary near the end of the window
        window = text[start:end]
        last_period = window.rfind(". ")
        if last_period > size * 0.5 and end < len(text):
            end = start + last_period + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end) if overlap >= end - start else end - overlap
        if end >= len(text):
            break
    return [c for c in chunks if c]


# ======================================================================================
# AI provider integration (streaming chat completions)
# ----------------------------------------------------------------------------------
# User-facing text anywhere in this section deliberately avoids naming the specific
# upstream provider/API -- errors shown in the UI are short, generic, and actionable.
# ======================================================================================

class AIProviderError(Exception):
    """Raised for any upstream failure; carries a short, user-friendly explanation."""

    def __init__(self, title: str, solution: str):
        self.title = title
        self.solution = solution
        super().__init__(title)


def _friendly_error(status_code: Optional[int], raw_body: str) -> AIProviderError:
    """Map a raw upstream error into a single-line title + a short, plain solution."""
    body_lower = (raw_body or "").lower()

    if status_code == 401 or "invalid" in body_lower and "key" in body_lower:
        return AIProviderError(
            "The assistant isn't configured correctly.",
            "Ask the site owner to check the server's setup, then try again.",
        )
    if status_code == 402 or "credit" in body_lower or "afford" in body_lower:
        return AIProviderError(
            "This model needs more capacity than is currently available.",
            "Try a smaller/free model from the dropdown, or send a shorter message.",
        )
    if status_code == 429 or "rate limit" in body_lower:
        return AIProviderError(
            "Too many requests right now.",
            "Wait a few seconds and try sending your message again.",
        )
    if status_code == 403:
        return AIProviderError(
            "This request was blocked.",
            "Try a different model, or contact the site owner if this continues.",
        )
    if status_code and status_code >= 500:
        return AIProviderError(
            "The assistant is temporarily unavailable.",
            "Please try again in a moment.",
        )
    return AIProviderError(
        "Something went wrong generating a response.",
        "Try again, or switch models in Settings.",
    )


async def stream_openrouter_chat(
    messages: List[Dict[str, str]], model: str
) -> AsyncGenerator[str, None]:
    """Stream tokens from the upstream chat completions endpoint. Raises AIProviderError
    (never yields raw error text) so the caller can show a clean, short message."""
    if not OPENROUTER_API_KEY:
        raise AIProviderError(
            "The assistant isn't set up yet.",
            "Ask the site owner to add a provider key on the server, then restart.",
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:8000"),
        "X-Title": "AI Chatbot",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,  # keeps replies within typical free/low-credit limits
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload
            ) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    raise _friendly_error(resp.status_code, err_body.decode(errors="ignore"))
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except httpx.RequestError:
        raise AIProviderError(
            "Couldn't reach the assistant service.",
            "Check your internet connection and try again.",
        )


# ======================================================================================
# FastAPI app setup
# ======================================================================================

app = FastAPI(title="AI Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.on_event("startup")
def on_startup():
    init_db()


# ======================================================================================
# Pydantic request/response models
# ======================================================================================

class NewChatRequest(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = DEFAULT_MODEL


class RenameChatRequest(BaseModel):
    title: str


class ChatMessageRequest(BaseModel):
    chat_id: str
    message: str
    model: Optional[str] = DEFAULT_MODEL
    use_rag: Optional[bool] = True


# ======================================================================================
# HTML page routes
# ======================================================================================

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": AVAILABLE_MODELS, "default_model": DEFAULT_MODEL},
    )


# ======================================================================================
# Chat CRUD APIs
# ======================================================================================

@app.get("/api/chats")
async def list_chats():
    conn = get_db()
    rows = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@app.post("/api/chats")
async def create_chat(payload: NewChatRequest):
    chat_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO chats (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, payload.title or "New Chat", payload.model or DEFAULT_MODEL, now, now),
    )
    conn.commit()
    conn.close()
    return {"id": chat_id, "title": payload.title, "model": payload.model, "created_at": now, "updated_at": now}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    conn = get_db()
    chat = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")
    messages = conn.execute(
        "SELECT * FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)
    ).fetchall()
    documents = conn.execute(
        "SELECT * FROM documents WHERE chat_id = ? ORDER BY uploaded_at ASC", (chat_id,)
    ).fetchall()
    conn.close()

    msg_list = []
    for m in messages:
        d = row_to_dict(m)
        d["sources"] = json.loads(d["sources"]) if d.get("sources") else []
        msg_list.append(d)

    return {
        "chat": row_to_dict(chat),
        "messages": msg_list,
        "documents": [row_to_dict(d) for d in documents],
    }


@app.put("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, payload: RenameChatRequest):
    conn = get_db()
    cur = conn.execute(
        "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
        (payload.title, datetime.utcnow().isoformat(), chat_id),
    )
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Chat not found")
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    conn = get_db()
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))  # cascades messages/documents
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM documents WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()
    vector_store.delete_chat(chat_id)
    return {"ok": True}


@app.get("/api/chats/{chat_id}/export")
async def export_chat(chat_id: str, fmt: str = Query("json", enum=["json", "txt", "md"])):
    conn = get_db()
    chat = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")
    messages = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)
    ).fetchall()
    conn.close()

    title = chat["title"] or "chat"
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:50]

    if fmt == "json":
        content = json.dumps(
            {"title": title, "messages": [row_to_dict(m) for m in messages]}, indent=2
        )
        media_type = "application/json"
        filename = f"{safe_title}.json"
    else:
        lines = [f"# {title}\n"]
        for m in messages:
            speaker = "You" if m["role"] == "user" else "Assistant"
            lines.append(f"**{speaker}** ({m['created_at']}):\n{m['content']}\n")
        content = "\n".join(lines)
        media_type = "text/markdown" if fmt == "md" else "text/plain"
        filename = f"{safe_title}.{fmt}"

    return PlainTextResponse(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ======================================================================================
# Model listing / settings
# ======================================================================================

@app.get("/api/models")
async def get_models():
    return {"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL}


# ======================================================================================
# File upload -> parse -> chunk -> embed -> store (RAG ingestion)
# ======================================================================================

@app.post("/api/upload")
async def upload_file(chat_id: str = Form(...), file: UploadFile = File(...)):
    conn = get_db()
    chat = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")

    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        conn.close()
        raise HTTPException(413, "File too large (max 15MB)")

    text = extract_text_from_file(file.filename, raw)
    if not text.strip():
        conn.close()
        raise HTTPException(400, "No extractable text found in this file")

    # Persist the raw file to disk (best-effort; ephemeral on serverless)
    doc_id = str(uuid.uuid4())
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", file.filename)
    with contextlib.suppress(OSError):
        with open(os.path.join(UPLOAD_DIR, f"{doc_id}_{safe_name}"), "wb") as f:
            f.write(raw)

    chunks = chunk_text(text)
    added = vector_store.add(chat_id, doc_id, file.filename, chunks)

    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO documents (id, chat_id, filename, chunk_count, uploaded_at) VALUES (?, ?, ?, ?, ?)",
        (doc_id, chat_id, file.filename, added, now),
    )
    conn.commit()
    conn.close()

    return {"id": doc_id, "filename": file.filename, "chunks": added}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    conn = get_db()
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    vector_store.delete_document(doc_id)
    return {"ok": True}


# ======================================================================================
# Chat streaming endpoint (RAG + memory + OpenRouter streaming -> SSE)
# ======================================================================================

@app.post("/api/chat/stream")
async def chat_stream(payload: ChatMessageRequest):
    conn = get_db()
    chat = conn.execute("SELECT * FROM chats WHERE id = ?", (payload.chat_id,)).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")

    model = payload.model or chat["model"] or DEFAULT_MODEL
    now = datetime.utcnow().isoformat()

    # 1) Save the user's message immediately
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (payload.chat_id, "user", payload.message, now),
    )
    # Auto-title new chats from the first message
    if chat["title"] == "New Chat":
        auto_title = payload.message.strip()[:60] or "New Chat"
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (auto_title, payload.chat_id))
    conn.execute("UPDATE chats SET updated_at = ?, model = ? WHERE id = ?", (now, model, payload.chat_id))
    conn.commit()

    # 2) Retrieve conversation memory (recent messages)
    history_rows = conn.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
        (payload.chat_id, MAX_HISTORY_MESSAGES),
    ).fetchall()
    conn.close()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(history_rows)]

    # 3) RAG retrieval scoped to this chat's uploaded documents
    sources: List[Dict[str, Any]] = []
    system_content = SYSTEM_PROMPT
    if payload.use_rag:
        matches = vector_store.search(payload.chat_id, payload.message)
        if matches:
            context_blocks = "\n\n".join(
                f"[Source: {m['filename']}]\n{m['text']}" for m in matches
            )
            system_content += f"\n\n--- Retrieved context from uploaded documents ---\n{context_blocks}"
            sources = [{"filename": m["filename"], "score": round(m["score"], 3)} for m in matches]

    messages_for_model = [{"role": "system", "content": system_content}] + history

    # 4) Stream the response back to the browser as Server-Sent Events
    async def event_generator() -> AsyncGenerator[str, None]:
        if sources:
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        full_reply = ""
        try:
            async for token in stream_openrouter_chat(messages_for_model, model):
                full_reply += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0)  # yield control so chunks flush promptly
        except AIProviderError as err:
            yield f"data: {json.dumps({'type': 'error', 'title': err.title, 'solution': err.solution})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return  # nothing to persist -- the failed turn isn't saved as an assistant message

        # Persist the assistant's full reply once streaming completes successfully
        save_conn = get_db()
        save_conn.execute(
            "INSERT INTO messages (chat_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (payload.chat_id, "assistant", full_reply, json.dumps(sources), datetime.utcnow().isoformat()),
        )
        save_conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?", (datetime.utcnow().isoformat(), payload.chat_id)
        )
        save_conn.commit()
        save_conn.close()

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens stream live
        },
    )


# ======================================================================================
# Health check (useful for uptime monitors / Vercel diagnostics)
# ======================================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "faiss": HAVE_FAISS,
        "pdf_support": HAVE_PYPDF,
        "docx_support": HAVE_DOCX,
        "model": DEFAULT_MODEL,
        "on_vercel": ON_VERCEL,
    }


# ======================================================================================
# Local dev entrypoint (Vercel imports `app` directly and never runs this block)
# ======================================================================================

if __name__ == "__main__":
    import uvicorn

    init_db()
    # NOTE: default host is 127.0.0.1 (localhost) for local development so the
    # printed URL is directly clickable/openable in a browser. Binding to
    # 0.0.0.0 (all network interfaces) is what servers use in production/containers,
    # but "0.0.0.0" itself is not a valid browser address on all OS/browsers --
    # use 127.0.0.1 or localhost instead, or set HOST=0.0.0.0 explicitly if you
    # need this machine reachable from other devices on your network.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  🚀 AI Chatbot running -> open http://{host}:{port} in your browser\n")
    uvicorn.run("app:app", host=host, port=port, reload=True)
