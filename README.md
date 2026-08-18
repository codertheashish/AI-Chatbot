# AI Chatbot — FastAPI + OpenRouter + RAG (Single-File Backend)

A production-ready, ChatGPT-style chatbot. The entire backend (routing, database,
document ingestion, embeddings, vector search, streaming chat) lives in **`app.py`**.
Frontend is plain HTML/CSS/JS — no React, no Streamlit, no build step.

## Features

- 💬 ChatGPT-like UI (sidebar, bubbles, streaming responses, dark mode)
- ⚡ Real-time token streaming from OpenRouter via Server-Sent Events
- 🧠 Conversation memory (per-chat history sent back to the model)
- 📎 Upload PDF / TXT / DOCX → chunked, embedded, and retrieved via RAG
- 📚 Source citations shown under grounded answers
- 🔍 FAISS-backed vector search (auto-falls-back to NumPy cosine similarity
  if `faiss-cpu` isn't installed/available on your host)
- 🗂️ Multiple chats: create, rename, delete, search, export (md/txt/json)
- 🎤 Voice input (browser `SpeechRecognition`) and 🔊 voice output
  (browser `speechSynthesis`) — no server-side audio processing needed
- 🌓 Dark/light theme toggle, persisted in `localStorage`
- 📱 Mobile-responsive layout with collapsible sidebar
- ⚙️ Model picker (switch OpenRouter models per chat from Settings)

## Project structure

```
AI-Chatbot/
├── app.py               # entire backend: FastAPI app, DB, RAG, streaming, routes
├── requirements.txt
├── vercel.json
├── .env.example          # copy to .env and fill in your key
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── uploads/               # uploaded source files land here (gitignored)
└── vector_db/             # persisted vector index (store.json, gitignored)
```

## 1. Local setup

```bash
cd AI-Chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=sk-or-v1-...

python app.py
# → http://localhost:8000
```

SQLite (`database.db`), uploaded files (`uploads/`) and the vector index
(`vector_db/store.json`) are created automatically on first run.

## 2. Environment variables

| Variable             | Required | Description                                             |
|-----------------------|:--------:|-----------------------------------------------------------|
| `OPENROUTER_API_KEY`  | ✅       | Your key from https://openrouter.ai/keys                 |
| `MODEL`               | ❌       | Default model id, e.g. `openai/gpt-4.1-mini`              |
| `APP_URL`             | ❌       | Sent as `HTTP-Referer` to OpenRouter for attribution      |
| `MAX_TOKENS`          | ❌       | Cap on reply length (default `2048`). Lower this if you're on a free/low-credit account and see "insufficient credits" errors. |
| `HOST` / `PORT`       | ❌       | Local dev bind address (default `127.0.0.1:8000`)         |

### Google sign-in (optional, off by default)

The sidebar has a "Sign in with Google" button (client-side only, via Google
Identity Services). To enable it:

1. Go to https://console.cloud.google.com/apis/credentials, open your OAuth
   **Web application** client (or create one).
2. Under **"Authorized JavaScript origins"** (⚠️ NOT "Authorised redirect
   URIs" — this button uses Google Identity Services, which only needs an
   origin, not a redirect URI), add the exact URL(s) you'll open the app
   from, each as a separate entry, e.g.:
   - `http://localhost:8000`
   - `http://127.0.0.1:8000`

   > If you see **"Access blocked: Authorisation error / no registered
   > origin / Error 401: invalid_client"**, that means step 2 was missed or
   > the origin doesn't exactly match the URL in your browser's address bar
   > (`localhost` and `127.0.0.1` are treated as different origins — add
   > both). Redirect URIs don't fix this; only "Authorized JavaScript
   > origins" does.
3. Copy the **Client ID** (looks like `xxxx.apps.googleusercontent.com`) and
   paste it into the `GOOGLE_CLIENT_ID` constant near the bottom of
   `static/script.js`.
4. Changes can take a few minutes to a few hours to propagate on Google's
   side — if it still fails right after saving, wait a bit and retry.

This is display-only (shows the signed-in name/avatar in the sidebar) — it
doesn't currently gate chats per-user. Wire it to the backend (e.g. verify the
credential server-side and scope `chat_id` rows to a `user_id` column) if you
need real per-user accounts.

### A note on user-facing wording

Anything shown in the chat UI (settings panel, error messages) intentionally
avoids naming the specific upstream AI provider or the word "API" — errors are
shown as one short line plus a one-to-two line fix, e.g. *"This model needs
more capacity than is currently available. Try a smaller/free model."* The
provider is still named "OpenRouter" throughout this README and in code
comments/env var names, since those are developer-facing, not shown to chat users.

Users can also switch models per-chat from the Settings panel in the UI —
no restart required.

## 3. Deploying to Vercel

```bash
npm i -g vercel      # if you don't already have the CLI
cd AI-Chatbot
vercel               # first deploy — follow the prompts
vercel --prod        # promote to production
```

In the Vercel dashboard, add the environment variables from step 2
(**Project → Settings → Environment Variables**) before/after the first
deploy, then redeploy so the function picks them up.

`vercel.json` builds `app.py` with `@vercel/python`, bundles `templates/`
and `static/` into the function, and routes every request to the FastAPI
app.

### ⚠️ Important: storage on serverless

Vercel Serverless Functions have a **read-only filesystem except `/tmp`**,
and `/tmp` is **wiped between cold starts**. `app.py` detects the `VERCEL`
env var and automatically redirects `database.db`, `uploads/`, and
`vector_db/` to `/tmp` so the app runs correctly out of the box — but this
means chat history and uploaded documents are **not guaranteed to persist**
long-term on Vercel's serverless tier (a new cold start = a fresh `/tmp`).

This is fine for demos, evaluation, and short sessions. For real production
persistence:

- Swap SQLite → a hosted Postgres (e.g. Neon, Supabase) — update `get_db()`.
- Swap the local vector store → a hosted vector DB (Pinecone, Qdrant, pgvector).
- Or deploy on a normal long-running server/container (Render, Fly.io, a VM,
  Docker) instead of serverless, where the local filesystem is fully
  persistent and none of the above changes are needed.

The app code is structured so those are localized swaps (`get_db()`,
`VectorStore`), not a rewrite.

### faiss-cpu on Vercel

`faiss-cpu` can be a large dependency. If your Vercel build fails on size,
comment it out of `requirements.txt` — `app.py` automatically falls back to
a NumPy-based cosine-similarity search with identical behavior (just less
optimized for very large document sets).

## 4. How the RAG pipeline works

1. Upload a PDF/TXT/DOCX → text is extracted (`pypdf` / `python-docx`).
2. Text is split into ~900-character overlapping chunks.
3. Each chunk is embedded with a fast, dependency-free hashing vectorizer
   (`embed_text()` in `app.py`) and stored in the vector index, scoped to
   the chat it was uploaded in.
4. On each user message, the top-k most similar chunks for that chat are
   retrieved and injected into the system prompt with `[Source: filename]`
   tags, and the model is asked to cite them.
5. Retrieved sources are also shown as chips under the assistant's reply.

> Want stronger retrieval? Swap `embed_text()` for a call to an embeddings
> API (OpenAI/OpenRouter embeddings, or `sentence-transformers` locally) —
> the rest of the pipeline (chunking, FAISS index, citation injection)
> doesn't need to change.

## 5. API reference

| Method & Path                         | Description                          |
|----------------------------------------|---------------------------------------|
| `GET  /`                               | Chat UI                               |
| `GET  /api/chats`                      | List chats                            |
| `POST /api/chats`                      | Create a chat                         |
| `GET  /api/chats/{id}`                 | Get chat + messages + documents       |
| `PUT  /api/chats/{id}`                 | Rename a chat                         |
| `DELETE /api/chats/{id}`               | Delete a chat                         |
| `GET  /api/chats/{id}/export?fmt=`     | Export chat (`json`/`txt`/`md`)       |
| `POST /api/chat/stream`                | Send a message, stream SSE response   |
| `POST /api/upload`                     | Upload a document for RAG             |
| `DELETE /api/documents/{id}`           | Remove a document from the index      |
| `GET  /api/models`                     | List available OpenRouter models      |
| `GET  /api/health`                     | Health/diagnostics check              |

## License

MIT — do whatever you like with this.
