import streamlit as st
import json
import style
from pathlib import Path
from util.context import Context
from util.vector_store import VectorDBManager
from util.document_processor import DocumentProcessor
from datetime import datetime
from util import agent

DEFAULT_COLLECTION_NAME = "knowledge-base"
CONVERSATIONS_DIR = Path(__file__).resolve().parent / "conversations"

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(style.style_sheet, unsafe_allow_html=True)

# ─── Session state initialisation ─────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],           # list of {"role": ..., "content": ...}
        "context_tokens": 0,      # tokens used so far
        "context_limit": 128_000, # model context window size
        "uploaded_files": [],     # list of collection names already embedded
        "embedded_collections": [],
    }
    defaults.update({
        "current_collection": DEFAULT_COLLECTION_NAME,
        "collection_created_notice": "",
        "upload_widget_version": 0,
        "upload_notices": [],
        "conversation_notice": "",
        "conversation_notice_kind": "success",
    })

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if not st.session_state.current_collection:
        st.session_state.current_collection = DEFAULT_COLLECTION_NAME

    # ── Agent ──────────────────────────────────────────────────────
    # The old code checked for "llm" but stored "agent", so the agent
    # was rebuilt on every rerun.
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorDBManager()

    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor(
            vector_db_manager=st.session_state.vector_store,
            embedding_provider=st.session_state.vector_store.embedding_provider,
        )

    if "agent" not in st.session_state:
        st.session_state.agent = agent.build_agent(
            vector_store=st.session_state.vector_store
        )

    # ── Context ────────────────────────────────────────────────────
    if "context" not in st.session_state:
        st.session_state.context = st.session_state.agent.context

    


def refresh_embedded_collections() -> list[str]:
    """Refresh collection names and tolerate ChromaDB API differences."""
    try:
        names = st.session_state.vector_store.list_collections()
    except Exception as exc:
        st.session_state.collection_error = str(exc)
        names = []

    st.session_state.uploaded_files = names
    st.session_state.embedded_collections = names
    return names


def collection_files(collection_name: str) -> list[str]:
    """Return distinct source/file names stored in a Chroma collection."""
    if not collection_name:
        return []

    try:
        return st.session_state.vector_store.get_documents_in_collection(
            collection_name
        )
    except Exception:
        return []


init_state()
VECTOR_STORE = st.session_state.vector_store
DOC_PROCESSOR = st.session_state.doc_processor
CONTEXT = st.session_state.context
AGENT = st.session_state.agent
refresh_embedded_collections()


# ─── Backend stubs (replace with your real implementations) ───────────────────
 
def embed_file(uploaded_file) -> tuple[bool, str]:
    return DOC_PROCESSOR.process_document(
        uploaded_file, DEFAULT_COLLECTION_NAME
    )
 
def query_rag(prompt: str) -> str:
    """Run the RAG agent and normalize its output to text."""
    response = AGENT.run(prompt)

    if isinstance(response, str):
        return response

    # Be defensive if the agent returns an object/dict instead of text.
    if isinstance(response, dict):
        for key in ("response", "answer", "content", "text"):
            if key in response and response[key] is not None:
                return str(response[key])

    return str(response)
 
 
def count_context_tokens() -> int:
    """Count the current agent context using Anthropic's message tokenizer."""
    count = AGENT.client.messages.count_tokens(
        model=AGENT.model,
        messages=AGENT.ContextAssembler.build_messages(AGENT.context),
        system=AGENT.system_prompt,
        tools=AGENT._tools(),
    )
    return count.input_tokens


def list_saved_conversations() -> list[dict[str, str]]:
    """Return saved conversations newest first, skipping invalid files."""
    conversations = []
    if not CONVERSATIONS_DIR.exists():
        return conversations

    for path in sorted(CONVERSATIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            conversations.append(
                {
                    "id": path.stem,
                    "title": data.get("title", "Untitled conversation"),
                    "saved_at": data.get("saved_at", ""),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    return conversations


def save_conversation() -> dict[str, str]:
    """Persist the visible chat and full agent context to a JSON file."""
    messages = st.session_state.messages
    first_user_message = next(
        (message["content"] for message in messages if message["role"] == "user"),
        "Untitled conversation",
    )
    title = " ".join(first_user_message.split())[:64] or "Untitled conversation"
    saved_at = datetime.now()
    conversation_id = saved_at.strftime("%Y%m%d-%H%M%S-%f")
    data = {
        "title": title,
        "saved_at": saved_at.isoformat(timespec="seconds"),
        "messages": messages,
        "context": AGENT.context.to_dict(),
    }

    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONVERSATIONS_DIR / f"{conversation_id}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"id": conversation_id, "title": title}


def stream_query_rag(prompt: str):
    """
    Streaming version of query_rag — yields string chunks.
    Use this with st.write_stream().
 
    Example with Anthropic streaming:
        with client.messages.stream(...) as stream:
            for text in stream.text_stream:
                yield text
 
    For now we yield the stub response word by word.
    """
    response = query_rag(prompt)
    for word in response.split(" "):
        yield word + " "
 


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⬡ AI Assistant</div>', unsafe_allow_html=True)

    collection_names = refresh_embedded_collections()


    if st.session_state.get("collection_created_notice"):
        st.success(st.session_state.collection_created_notice)
        st.session_state.collection_created_notice = ""

    if st.session_state.conversation_notice:
        getattr(st, st.session_state.conversation_notice_kind)(
            st.session_state.conversation_notice
        )
        st.session_state.conversation_notice = ""

    for level, message in st.session_state.upload_notices:
        getattr(st, level)(message)
    st.session_state.upload_notices = []

    # ── File uploader ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Upload documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload files to embed",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"document_uploader_{st.session_state.upload_widget_version}",
    )

    if uploaded:
        upload_notices = []
        with st.spinner(f"Embedding {len(uploaded)} file(s) into vector store…"):
            for f in uploaded:
                success, message = embed_file(f)
                if success:
                    upload_notices.append(("success", message))
                else:
                    upload_notices.append(
                        ("error", f"Could not embed '{f.name}': {message}")
                    )

        st.session_state.upload_notices = upload_notices
        st.session_state.upload_widget_version += 1
        st.rerun()

    # ── Files in collection ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Files in knowledge base</div>', unsafe_allow_html=True)

    if collection_names:
        files = collection_files(st.session_state.current_collection)
        if files:
            for filename in files:
                st.markdown(
                    f'<div class="file-pill">📄 {filename}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span style="font-size:12px;color:#666;">No files found in this knowledge base.</span>',
                unsafe_allow_html=True,
            )

    # ── Context window meter ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Conversations</div>', unsafe_allow_html=True)

    save_column, load_column = st.columns(2)
    with save_column:
        if st.button(
            "Save conversation",
            use_container_width=True,
            disabled=not st.session_state.messages,
        ):
            saved = save_conversation()
            st.session_state.conversation_notice = f"Saved: {saved['title']}"
            st.session_state.conversation_notice_kind = "success"
            st.rerun()

    saved_conversations = list_saved_conversations()
    conversations_by_id = {
        conversation["id"]: conversation for conversation in saved_conversations
    }
    conversation_ids = list(conversations_by_id)
    selected_conversation_id = None
    if conversation_ids:
        selected_conversation_id = st.selectbox(
            "Saved conversations",
            options=conversation_ids,
            format_func=lambda conversation_id: conversations_by_id[conversation_id]["title"],
            label_visibility="collapsed",
        )
    else:
        st.caption("No saved conversations")

    with load_column:
        if st.button(
            "Load conversation",
            use_container_width=True,
            disabled=selected_conversation_id is None,
        ):
            conversation_path = CONVERSATIONS_DIR / f"{selected_conversation_id}.json"
            try:
                data = json.loads(conversation_path.read_text(encoding="utf-8"))
                restored_context = Context.from_dict(data["context"])
                AGENT.context = restored_context
                st.session_state.context = restored_context
                st.session_state.messages = data["messages"]
                try:
                    st.session_state.context_tokens = count_context_tokens()
                except Exception:
                    st.session_state.context_tokens = 0
                st.session_state.conversation_notice = (
                    f"Loaded: {data.get('title', 'Untitled conversation')}"
                )
                st.session_state.conversation_notice_kind = "success"
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                st.session_state.conversation_notice = f"Could not load conversation: {exc}"
                st.session_state.conversation_notice_kind = "error"
            st.rerun()

    # ── Context window meter ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Context window</div>', unsafe_allow_html=True)

    used = st.session_state.context_tokens
    limit = st.session_state.context_limit
    pct = min(used / limit, 1.0) if limit else 0.0

    st.markdown(
        f'<div class="ctx-numbers">'
        f'{used:,} <span style="color:#444">/ {limit:,} tokens</span>'
        f' &nbsp;<span style="color:#555;font-size:11px;">({pct:.1%})</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    bar_color = "#4a9eff"
    if pct > 0.6:
        bar_color = "#f5a623"
    if pct > 0.85:
        bar_color = "#e25c5c"

    st.markdown(
        f"""
        <div style="background:#1e1e1e;border-radius:3px;height:6px;overflow:hidden;margin-bottom:8px;">
          <div style="width:{pct*100:.1f}%;height:100%;background:{bar_color};
                      border-radius:3px;transition:width 0.4s ease;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pct > 0.85:
        st.warning("Context nearly full — consider starting a new conversation.")
    elif pct > 0.6:
        st.info("Context over 60% used.")

    # ── Controls ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)

    if st.button("Start New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context_tokens = 0
        AGENT.context = Context()
        st.session_state.context = AGENT.context
        st.rerun()

    use_streaming = True #st.toggle("Stream responses", value=True)


# ─── Main chat area ────────────────────────────────────────────────────────────
st.markdown('<div class="chat-header">Ask your documents.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="chat-sub">Upload files in the sidebar, then ask anything about them.</div>',
    unsafe_allow_html=True,
)
 
# Render message history
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
          <div class="empty-icon">⬡</div>
          No messages yet.<br>
          Upload a document and start asking questions.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
 
# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your documents…"):
    print(f"User prompt: {prompt}")
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    # Generate assistant response
    with st.chat_message("assistant"):
        if use_streaming:
            # Stream token by token
            full_response = st.write_stream(
                stream_query_rag(prompt)
            )
        else:
            # Wait for full response
            with st.spinner("Thinking…"):
                full_response = query_rag(prompt)
            st.markdown(full_response)

        try:
            st.session_state.context_tokens = count_context_tokens()
        except Exception as exc:
            st.warning(f"Could not refresh the context token count: {exc}")
 
    # Save assistant response to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
 
    # Rerun so sidebar context meter updates immediately
    st.rerun()
