# ⬡ EasyRAG

A lightweight Retrieval-Augmented Generation (RAG) application built with Streamlit, Claude (Anthropic), and ChromaDB. Upload your documents, ask questions, and get answers grounded in your own data — with web search as a fallback.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![LLM](https://img.shields.io/badge/LLM-Claude_Sonnet_4-purple)

---

## Features

- **Document Ingestion** — Upload `.txt`, `.md`, and `.docx` files; they are chunked, normalized, and embedded into ChromaDB collections
- **Flat RAG** — Documents are chunked and embedded directly into ChromaDB; queries search passages without summary tiers
- **Agentic RAG** — Claude can list collections, search document chunks, search the web, or fetch a URL
- **Collection Management** — Create named collections to organize documents by topic, project, or domain
- **Web Search & Fetch** — Tavily-powered web search and page extraction for questions outside your local corpus
- **Context Window Meter** — Visual sidebar gauge shows how much of the model's 128k context is in use
- **Streaming Responses** — Toggle real-time token streaming on or off
- **Structured Logging** — Dual-format logging (plain text + redacted JSON Lines) with automatic API key scrubbing

---

## System Architecture

### High-Level Overview

```mermaid
graph TB
    User([fa:fa-user User])

    subgraph Streamlit["Streamlit UI (app.py)"]
        Sidebar["Sidebar<br/>Collection mgmt<br/>File upload<br/>Context meter"]
        Chat["Chat Interface<br/>Message history<br/>Streaming toggle"]
    end

    subgraph AgentCore["Agent Core (util/)"]
        Agent["Agent<br/>Tool-use loop"]
        CA["ContextAssembler<br/>Message builder"]
        Ctx["Context<br/>Turns / Steps / Retrieval"]
    end

    subgraph Skills["Skills (util/skills/)"]
        KB["knowledge_base<br/>• list_collections<br/>• search_knowledge_base"]
        WS["web_search<br/>Tavily search"]
        FU["fetch_url<br/>Tavily extract"]
    end

    subgraph StorageLayer["Vector Store (util/vector_store.py)"]
        VSM["VectorDBManager<br/>Storage & Retrieval"]
    end

    subgraph Storage["Storage"]
        Chroma[("ChromaDB<br/>(local persistent)")]
        Logs["Log Files<br/>easy_rag.log<br/>easy_rag.jsonl"]
    end

    subgraph External["External APIs"]
        Claude["Anthropic Claude<br/>Sonnet 4"]
        OpenAI["OpenAI Embeddings<br/>text-embedding-3-small"]
        Tavily["Tavily API<br/>Search + Extract"]
    end

    subgraph Embedding["Document Processing (util/)"]
        Processor["DocumentProcessor<br/>Read + Validate + Ingest"]
        Chunker["Chunker<br/>Split + Normalize"]
        Provider["EmbeddingProvider<br/>Ollama / OpenAI / Local"]
    end

    User --> Sidebar
    User --> Chat
    Sidebar -->|"upload files"| VSM
    Chat -->|"user query"| Agent
    Agent --> CA
    CA -->|"build messages"| Ctx
    Agent <-->|"API calls"| Claude
    Agent -->|"tool calls"| Skills
    KB <-->|"query"| VSM
    VSM <-->|"similarity search"| Chroma
    Sidebar -->|"process document"| Processor
    Processor --> Chunker
    Processor --> Provider
    Provider -->|"embed vectors"| OpenAI
    WS <-->|"search"| Tavily
    FU <-->|"extract"| Tavily

    Agent -.->|"log"| Logs

    style Streamlit fill:#1a1a2e,stroke:#4a9eff,color:#e8e8e8
    style AgentCore fill:#16213e,stroke:#4a9eff,color:#e8e8e8
    style Skills fill:#0f3460,stroke:#4a9eff,color:#e8e8e8
    style StorageLayer fill:#16213e,stroke:#4ade80,color:#e8e8e8
    style Storage fill:#1a1a2e,stroke:#666,color:#e8e8e8
    style External fill:#2d1b4e,stroke:#a855f7,color:#e8e8e8
    style Embedding fill:#1b2e1b,stroke:#4ade80,color:#e8e8e8

    linkStyle default stroke:#4a9eff,stroke-width:2px
```

### Document Ingestion Pipeline

```mermaid
flowchart LR
    Upload["📄 File Upload"] --> Validate{"Supported<br/>type?"}
    Validate -->|No| Reject["❌ Reject"]
    Validate -->|Yes| Dedup{"Duplicate<br/>check"}
    Dedup -->|Exists| Skip["⚠️ Skip"]
    Dedup -->|New| Transform["Read &<br/>Decode"]
    Transform --> Chunk["Recursive<br/>Text Split<br/>(512 chars)"]
    Chunk --> Normalize["Normalize<br/>whitespace"]
    Normalize --> EmbedChunks["Embed chunks<br/>(OpenAI)"]
    EmbedChunks --> StoreChunks[("ChromaDB<br/>document collection")]
```

### Agent Query Loop

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant C as Claude API
    participant S as Skills
    participant DB as VectorDBManager

    U->>A: User query
    A->>A: Append Turn to Context
    
    loop Agent tool-use loop
        A->>C: Send messages + tool defs
        C-->>A: Response

        alt stop_reason = "end_turn"
            A-->>U: Final text answer
        else stop_reason = "tool_use"
            A->>S: Execute tool(s)
            
            alt knowledge_base search
                S->>DB: search_chunks()
                DB-->>S: Relevant results
            else web_search
                S->>S: Tavily search
            else fetch_url
                S->>S: Tavily extract
            end
            
            S-->>A: Tool results
            Note over A: Feed results back as<br/>tool_result messages
        end
    end
```

---

## Project Structure

```
EasyRag/
├── app.py                      # Streamlit UI — sidebar, chat, file upload
├── config.py                   # Supported MIME types, system prompt
├── style.py                    # Custom dark-theme CSS (IBM Plex fonts)
│
├── util/
│   ├── __init__.py             # Package exports, loads .env
│   ├── agent.py                # Agent class — tool-use loop over Anthropic API
│   ├── context.py              # Context / Turn / Step / RetrievalContext dataclasses
│   ├── ContextAssembler.py     # Builds API message list from Context history
│   ├── chunker.py              # Text splitting and normalization
│   ├── embedding.py            # Ollama, OpenAI, and local embedding providers
│   ├── document_processor.py   # File validation and flat-RAG ingestion
│   ├── logging_utils.py        # Dual-format logger with API key redaction
│   ├── vector_store.py         # ChromaDB persistence and flat chunk search
│   │
│   └── skills/
│       ├── __init__.py         # SkillRegistry assembly
│       ├── base.py             # Skill + SkillRegistry abstractions
│       ├── knowledge_base.py   # Collection listing and chunk search tools
│       ├── web_search.py       # Tavily web search skill
│       └── fetch_url.py        # Tavily page extraction skill
│
├── chroma_db/                  # Persistent ChromaDB storage (auto-created)
├── data/                       # Sample documents for testing
├── log/                        # Runtime logs (easy_rag.log, easy_rag.jsonl)
├── scripts/                    # Utility scripts (e.g. inspect Chroma collections)
│
├── .env                        # API keys (DO NOT commit — see Security below)
├── .gitignore
└── LICENSE                     # MIT
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for [Anthropic](https://console.anthropic.com/) and [Tavily](https://tavily.com/)
- An embedding backend: Ollama (default), OpenAI, or the local SentenceTransformers model

### Installation

```bash
# Clone the repo
git clone https://github.com/JosiahKeime/EasyRag.git
cd EasyRag

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_KEY=tvly-...
```

Set `EMBEDDING_BACKEND` to `ollama`, `openai`, or `local`. The OpenAI backend also requires `OPENAI_API_KEY`.

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload documents in the sidebar, then start chatting.

---

## How It Works

### 1. Upload & Embed
Upload a supported text file via the sidebar. **DocumentProcessor** validates and reads it, **Chunker** splits it into 512-character chunks with 64-character overlap and normalizes whitespace, and **EmbeddingProvider** embeds the chunks into ChromaDB using the configured backend.

### 2. Ask a Question
Type a question in the chat box. The **Agent** sends your full conversation history (managed by the **ContextAssembler**) to Claude Sonnet 4 along with tool definitions.

### 3. Agentic Tool Use
Claude decides which tools to call — it might:
- **List collections** to see available document collections
- **Search the knowledge base** directly for relevant passages
- **Search the web** via Tavily for current information
- **Fetch a URL** to extract full page content

The Agent loops until Claude produces a final text answer (`end_turn`).

---

## Adding a New Skill

The skills system is designed for easy extension:

1. Create `util/skills/your_skill.py` with a `register(registry)` function
2. Use the `@registry.register(name=..., description=..., input_schema=...)` decorator
3. Import and call it in `util/skills/base.py` → `build_default_registry()`

See [`web_search.py`](util/skills/web_search.py) for a minimal example.

---

## Security

> ⚠️ **Important**: Never commit API keys to version control. Add `.env` to your `.gitignore`:
>
> ```gitignore
> .env
> ```
>
> If keys have been previously committed, rotate them immediately on each provider's dashboard.

---

## License

[MIT](LICENSE) © 2026 JosiahKeime
