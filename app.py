from __future__ import annotations

import html
import os

import streamlit as st
from dotenv import load_dotenv

from rag_engine import DocumentIndex, OpenAIClient, SearchResult, extract_text


load_dotenv()

st.set_page_config(
    page_title="CogniSync | Document Intelligence",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #e8edf7;
        --muted: #929cb1;
        --panel: #151c2d;
        --line: #273148;
        --accent: #8d7cff;
        --accent-soft: #201f3c;
    }
    .stApp { background: #0c1120; color: var(--ink); }
    [data-testid="stSidebar"] {
        background: #101728;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    .brand {
        color: #f3f5ff; font-size: 1.25rem; font-weight: 750;
        letter-spacing: -.04em; margin-bottom: .15rem;
    }
    .eyebrow {
        color: #a99eff; font-size: .73rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase;
    }
    .hero-title {
        color: #f4f6ff; font-size: clamp(2.2rem, 5vw, 3.8rem);
        font-weight: 750; letter-spacing: -.055em; line-height: 1.05;
        margin: .45rem 0 .7rem;
    }
    .hero-copy { color: var(--muted); font-size: 1.04rem; max-width: 720px; }
    .metric-card, .info-card {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 14px; padding: 1rem 1.1rem;
    }
    .metric-label { color: var(--muted); font-size: .82rem; }
    .metric-value { color: #f3f5ff; font-size: 1.6rem; font-weight: 700; }
    .source-card {
        background: #11192a; border: 1px solid var(--line);
        border-radius: 12px; padding: .8rem 1rem; margin: .5rem 0;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent); border-color: var(--accent); color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background: #7765ed; border-color: #7765ed;
    }
    [data-testid="stChatMessage"] {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 14px; margin-bottom: .65rem;
    }
    [data-testid="stFileUploader"] {
        background: #11192a; border: 1px dashed #3b4660; border-radius: 12px;
    }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_index() -> DocumentIndex:
    if "document_index" not in st.session_state:
        st.session_state.document_index = DocumentIndex()
    return st.session_state.document_index


def set_notice(kind: str, message: str) -> None:
    st.session_state.notice = (kind, message)


def local_evidence_summary(results: list[SearchResult]) -> str:
    if not results:
        return "I couldn't find a relevant passage in the documents currently indexed."
    excerpts = [
        f"**[{index}] {result.source}**\n\n> {result.text}"
        for index, result in enumerate(results[:3], start=1)
    ]
    return (
        "I found these relevant passages. Add an OpenAI API key in the sidebar to "
        "generate a synthesized answer with semantic retrieval.\n\n"
        + "\n\n".join(excerpts)
    )


def render_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return
    with st.expander("Retrieved sources"):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f'<div class="source-card"><strong>[{number}] '
                f'{html.escape(source["source"])}</strong><br>'
                f'{html.escape(source["text"])}</div>',
                unsafe_allow_html=True,
            )


index = get_index()
configured_key = os.getenv("OPENAI_API_KEY", "")

with st.sidebar:
    st.markdown('<div class="brand">CogniSync</div>', unsafe_allow_html=True)
    st.caption("DOCUMENT INTELLIGENCE")
    st.divider()
    st.markdown("#### Your knowledge base")
    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="PDF, Word, text, Markdown, and CSV files are supported (25 MB each).",
    )
    if uploaded_files and st.button("Index documents", type="primary", use_container_width=True):
        success_count = 0
        errors = []
        for uploaded_file in uploaded_files:
            if uploaded_file.size > 25 * 1024 * 1024:
                errors.append(f"{uploaded_file.name}: exceeds the 25 MB file limit.")
                continue
            try:
                text = extract_text(uploaded_file.name, uploaded_file.getvalue())
                index.add_document(uploaded_file.name, text)
                success_count += 1
            except (ValueError, RuntimeError) as error:
                errors.append(f"{uploaded_file.name}: {error}")
        if success_count:
            set_notice("success", f"Indexed {success_count} document(s) into the workspace.")
        if errors:
            set_notice("error", "\n".join(errors))
        st.rerun()

    st.markdown("#### AI settings")
    api_key = st.text_input(
        "OpenAI API key",
        value=configured_key,
        type="password",
        placeholder="Optional — enables semantic search and AI answers",
        help="Your key is used only from this app session and is not saved to disk.",
    )
    if api_key != st.session_state.get("active_api_key", ""):
        try:
            client = OpenAIClient(api_key) if api_key.strip() else None
            index.set_embedder(client.embed if client else None)
            st.session_state.active_api_key = api_key
            st.session_state.openai_client = client
            if client and index.chunk_count:
                set_notice("success", "Semantic search is ready for your indexed documents.")
        except (RuntimeError, ValueError) as error:
            set_notice("error", f"Could not enable semantic search: {error}")
            index.set_embedder(None)
            st.session_state.active_api_key = api_key
            st.session_state.openai_client = None
        st.rerun()

    client = st.session_state.get("openai_client")
    if client:
        st.caption("Semantic retrieval · AI answers enabled")
    else:
        st.caption("Local keyword retrieval · no API key required")
    st.divider()
    sources = index.sources
    st.markdown(f"#### Indexed files · {len(sources)}")
    if sources:
        for source in sources:
            left, right = st.columns([5, 1])
            with left:
                st.caption(f"▤  {source}")
            with right:
                if st.button("×", key=f"remove-{source}", help=f"Remove {source}"):
                    index.remove_document(source)
                    set_notice("success", f"Removed {source} from the workspace.")
                    st.rerun()
    else:
        st.caption("Your workspace is empty. Add a document to get started.")

st.markdown('<div class="eyebrow">YOUR KNOWLEDGE, IN CONTEXT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Make every document<br>answerable.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Bring your files together and ask questions in plain language. '
    'CogniSync finds the supporting passages so every answer stays connected to its source.</div>',
    unsafe_allow_html=True,
)

st.write("")
metric_columns = st.columns(3)
metrics = (
    ("Documents", len(index.sources)),
    ("Indexed passages", index.chunk_count),
    ("Retrieval mode", "Semantic" if client else "Local"),
)
for column, (label, value) in zip(metric_columns, metrics):
    with column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

notice = st.session_state.pop("notice", None)
if notice:
    kind, message = notice
    if kind == "error":
        st.error(message)
    else:
        st.success(message)

st.write("")
st.markdown("### Ask your documents")
if not index.sources:
    st.info("Upload a document from the sidebar, then ask a question about its contents.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question grounded in your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            results = index.search(question, limit=5)
            answer = client.answer(question, results) if client and results else local_evidence_summary(results)
            with st.chat_message("assistant"):
                st.markdown(answer)
                retrieved_sources = [
                    {"source": result.source, "text": result.text} for result in results
                ]
                render_sources(retrieved_sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": retrieved_sources}
            )
        except (RuntimeError, ValueError) as error:
            st.error(f"Could not answer this question: {error}")
