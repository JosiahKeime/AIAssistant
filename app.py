import streamlit as st
import chromadb
import style
from util.context import Context
from util.embedder import Embedder
from datetime import datetime
from util import agent

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="EasyRAG",
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
        "index": None,            # your vector store / index object goes here
    }
    defaults.update({
        "current_collection": "",
        "collection_created_notice": "",
    })

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Agent ──────────────────────────────────────────────────────
    # The old code checked for "llm" but stored "agent", so the agent
    # was rebuilt on every rerun.
    if "agent" not in st.session_state:
        st.session_state.agent = agent.build_agent()

    # ── Context ────────────────────────────────────────────────────
    if "context" not in st.session_state:
        st.session_state.context = st.session_state.agent.context

    # ── Embedder ──────────────────────────────────────────────────
    if "embedder" not in st.session_state:
        st.session_state.embedder = Embedder()




def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


def refresh_embedded_collections() -> list[str]:
    """Refresh collection names and tolerate ChromaDB API differences."""
    try:
        client = get_chroma_client()
        collections = client.list_collections()

        names = []
        for collection in collections:
            # ChromaDB versions differ: some return Collection objects,
            # newer versions may return collection names directly.
            name = collection.name if hasattr(collection, "name") else str(collection)
            names.append(name)

        names = sorted(set(names))
    except Exception as exc:
        st.session_state.collection_error = str(exc)
        names = []

    EMBEDDER.collection_names = names
    st.session_state.uploaded_files = names
    st.session_state.embedded_collections = names
    return names


def collection_files(collection_name: str) -> list[str]:
    """Return distinct source/file names stored in a Chroma collection."""
    if not collection_name:
        return []

    try:
        collection = get_chroma_client().get_collection(collection_name)
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        files = set()
        for metadata in metadatas:
            if not metadata:
                continue
            filename = (
                metadata.get("file_name")
                or metadata.get("filename")
                or metadata.get("source")
                or metadata.get("file")
            )
            if filename:
                files.add(str(filename))

        return sorted(files)
    except Exception:
        return []


init_state()
EMBEDDER = st.session_state.embedder
CONTEXT = st.session_state.context
AGENT = st.session_state.agent
refresh_embedded_collections()


# ─── Backend stubs (replace with your real implementations) ───────────────────
 
def embed_file(uploaded_file, collection_name: str | None = None) -> bool:
    success, _ = EMBEDDER.embed_file(uploaded_file, collection_name=collection_name)
    return success
 
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
    st.markdown('<div class="sidebar-title">⬡ EasyRAG</div>', unsafe_allow_html=True)

    # ── Collection creation / upload ─────────────────────────────────────────
    st.markdown('<div class="section-label">Collections</div>', unsafe_allow_html=True)

    collection_names = refresh_embedded_collections()

    new_collection = st.text_input(
        "New collection name",
        placeholder="e.g. legal-docs",
        key="new_collection_name",
    )

    if st.button("Create collection", use_container_width=True):
        name = new_collection.strip()
        if not name:
            st.warning("Enter a collection name first.")
        else:
            try:
                # Chroma creates the collection if it does not already exist.
                get_chroma_client().get_or_create_collection(name=name)
                refresh_embedded_collections()
                st.session_state.current_collection = name
                st.session_state.collection_created_notice = f"Created collection '{name}'."
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create collection: {exc}")

    if st.session_state.get("collection_created_notice"):
        st.success(st.session_state.collection_created_notice)
        st.session_state.collection_created_notice = ""

    # ── File uploader ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Upload documents</div>', unsafe_allow_html=True)

    if collection_names:
        selected_collection = st.selectbox(
            "Upload into collection",
            options=collection_names,
            index=(
                collection_names.index(st.session_state.current_collection)
                if st.session_state.current_collection in collection_names
                else 0
            ),
        )
        st.session_state.current_collection = selected_collection
    else:
        selected_collection = st.session_state.current_collection
        st.info("Create a collection before uploading files.")

    uploaded = st.file_uploader(
        "Upload files to embed",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if not selected_collection:
            st.warning("Select or create a collection before uploading files.")
        else:
            with st.spinner(
                f"Embedding {len(uploaded)} file(s) into '{selected_collection}'…"
            ):
                for f in uploaded:
                    success = embed_file(f, collection_name=selected_collection)
                    if not success:
                        st.error(f"Could not embed '{f.name}'.")

            refresh_embedded_collections()

    # ── Files in collection ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Files in collection</div>', unsafe_allow_html=True)

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
                '<span style="font-size:12px;color:#666;">No files found in this collection.</span>',
                unsafe_allow_html=True,
            )

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

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context_tokens = 0
        st.rerun()

    use_streaming = st.toggle("Stream responses", value=True)


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
            # Approximate token update for streaming
            # (replace with real token count from your API)
            st.session_state.context_tokens = min(
                st.session_state.context_tokens + len(prompt.split()) * 3,
                st.session_state.context_limit,
            )
        else:
            # Wait for full response
            with st.spinner("Thinking…"):
                full_response = query_rag(prompt)
                total_tokens = min(
                    st.session_state.context_tokens + len(prompt.split()) * 3,
                    st.session_state.context_limit,
                )
            st.markdown(full_response)
            st.session_state.context_tokens = total_tokens
 
    # Save assistant response to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
 
    # Rerun so sidebar context meter updates immediately
    st.rerun()
