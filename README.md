# Mapper — Agentic Schema Mapping

An application that maps source CSV columns to a target schema using heuristics, semantic memory (FAISS), and LLM reasoning via the **OpenRouter API**. Upload a CSV, get suggested mappings, review or correct them, and train the system from your feedback.

---

## Demo & documentation

| Link | Description |
|------|-------------|
| [Demo video](https://drive.google.com/file/d/1D0iKdvP8UimkvgY3RdWcgkCjti0eQUVJ/view?usp=sharing) | End-to-end application walkthrough / usage demo |
| [Additional documentation](SYSTEM_ARCHITECTURE.md) | System architecture, components, and design reference |
| [Presentation pdf](https://drive.google.com/file/d/1z7wV4HKLAJ8o026sGwMZlw82Mdn1qzLv/view?usp=sharing) | High-level technical overview and Mapper product summary |

---

## Features

- **CSV upload & profiling** — Ingest CSV files and infer column types, null ratios, uniqueness, and other metrices.
- **Multi-step mapping pipeline** — Heuristics → memory (FAISS) → target schema similarity → **LLM disambiguation** (OpenRouter)
- **Review workflow** — Approve, reject, or correct mappings; save tasks and resume later
- **Memory & training** — Historical mappings stored in MongoDB; train from Excel/CSV for tenant-specific patterns
- **Saved tasks** — Save in-progress reviews and complete them later

---

## 🛠 Tech Stack

| Layer        | Technology |
|-------------|------------|
| **Backend** | Python 3.x, FastAPI, Uvicorn |
| **LLM**     | OpenRouter API (OpenAI-compatible: GPT-4o-mini, Gemini, Claude, etc.) |
| **Database**| MongoDB (memory, audit, saved tasks) |
| **Vector search** | FAISS, Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Matching** | RapidFuzz, Pandas, NumPy |
| **Frontend**| Vanilla HTML/CSS/JS (no framework) |

---

## 📋 Prerequisites

- **Python 3.10+**
- **MongoDB** (local or Atlas) — default: `mongodb://localhost:27017`
- **OpenRouter API key** (see below)

---

## 🚀 Installation

### 1. Clone and enter the project

```bash
git clone https://github.com/kunal5711/Megatron.git
cd Megatron
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

*(First run may take a few minutes while Sentence Transformers and FAISS download.)*

### 4. Environment variables

Copy the example env file and add your keys:

```bash
# macOS / Linux
cp env.example .env

# Windows (PowerShell)
Copy-Item env.example .env
```

Edit `.env` and set at least:

- `OPENROUTER_API_KEY` — **Required for LLM-based mapping.** See [Getting an OpenRouter API key](#-getting-an-openrouter-api-key) below.
- `MONGO_URI` — If MongoDB is not on `localhost:27017`.
- `MONGO_DB_NAME` — Optional; default is `mapper_db`.

---

## 🔑 Getting an OpenRouter API Key

The app uses [OpenRouter](https://openrouter.ai) to call LLMs (e.g. GPT-4o-mini, Gemini, Claude) for mapping decisions. OpenRouter is **OpenAI-compatible**, so one key gives access to many models.

### Step 1: Create an account

1. Go to **[openrouter.ai](https://openrouter.ai)** and sign up (e.g. with Google/GitHub).

### Step 2: Create an API key

1. Open **[openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)**.
2. Click **Create Key**.
3. Set a **name** (e.g. `Mapper Hackathon`).
4. Optionally set a **spending limit** and **expiration**.
5. Copy the key — it is shown **only once** (format: `sk-or-v1-...`). Store it securely.

### Step 3: Add the key to your app

In your project `.env` (create from `env.example` if needed):

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Optional (defaults are fine for most setups):

```env
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Other models you can try: `google/gemini-flash-1.5`, `anthropic/claude-3-haiku`, etc. See [OpenRouter models](https://openrouter.ai/docs/features/models).

**If you leave `OPENROUTER_API_KEY` empty:** the app still runs; mapping falls back to heuristics only (no LLM calls).

---

## ▶️ How to Run

### 1. Start MongoDB

Ensure MongoDB is running locally, or that `MONGO_URI` in `.env` points to your instance (e.g. Atlas).

### 2. Start the backend

From the project root (with `.venv` activated):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- API: **http://localhost:8000**
- Docs: **http://localhost:8000/docs**

*(First run may take a few minutes while Sentence Transformers and FAISS are loading.)*

### 3. Open the frontend

- Open `frontend/index.html` in a browser **http://127.0.0.1:5500/frontend/index.html**.

Use the **Mapping** page: enter a tenant name, upload a CSV, and click **Generate** to get mappings. Use **Memory**, **Saved Tasks**, and **Training** as needed.

---

## 📁 Project structure

```
Megatron/
├── backend/
│   ├── main.py           # FastAPI app, routes
│   ├── config.py         # Env vars, OpenRouter & MongoDB settings
│   ├── models.py         # Pydantic models
│   └── services/
│       ├── ingestion.py  # CSV read/preview
│       ├── profiler.py   # Column profiling
│       ├── target_schema.py  # Load schema, embeddings
│       ├── decision.py   # Heuristics + memory + LLM pipeline
│       ├── llm_service.py    # OpenRouter API client
│       ├── memory.py     # MongoDB memory CRUD
│       ├── vector_store.py  # FAISS index
│       └── training.py   # Training data ingestion
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── target_schema.json    # Target field definitions (large)
├── requirements.txt
├── env.example           # Copy to .env and add your keys
└── README.md
```

---

## ⚙️ Configuration reference (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *(none)* | **Required for LLM.** Get from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys). |
| `OPENROUTER_API_URL` | `https://openrouter.ai/api/v1/chat/completions` | OpenRouter endpoint. |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model ID (e.g. `google/gemini-flash-1.5`). |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string. |
| `MONGO_DB_NAME` | `mapper_db` | Database name. |
| `LLM_MAX_RPM` | `30` | Max LLM requests per minute. |
| `LLM_MAX_RPD` | `14400` | Max LLM requests per day. |
| `ENABLE_LLM_CACHE` | `1` | Set to `0` to disable response cache. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for FAISS embeddings. |

---

## 🧪 Quick health check

- **API:** `GET http://localhost:8000/` → `{"status":"ok","message":"Mapper Phase 3 API"}`
- **Targets:** `GET http://localhost:8000/targets/` → list of target fields
- **Memory stats:** `GET http://localhost:8000/memory/stats/` → memory and LLM stats

---