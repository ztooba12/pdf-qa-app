"""
PDF Studio — Q&A + Summarize + Edit
-----------------------------------
Upload a PDF, then:
  • Chat / ask questions about it
  • Summarize it (short / detailed / bullets)
  • Edit it — add, change, delete or modify text manually OR by telling the
    AI what to change in plain English — then download the final edited PDF.

Stack (all free):
  Streamlit (UI) + LangChain (logic) + Groq (LLM) + HuggingFace (local embeddings)
  + reportlab (rebuilds the edited PDF)

Run it with:   streamlit run app.py
"""

import os
import tempfile
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# reportlab — used to rebuild a real PDF from the edited text
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Optional Arabic shaping (only used if the libs are installed) -----------------
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _ARABIC_OK = True
except Exception:
    _ARABIC_OK = False


# =============================================================================
#  PAGE CONFIG + ANIMATED UI (custom CSS)
# =============================================================================
st.set_page_config(page_title="PDF Studio", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* --- Background --- */
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(201,162,77,0.10), transparent 60%),
            radial-gradient(1000px 500px at 110% 10%, rgba(201,162,77,0.06), transparent 55%),
            #0b0b0d;
        color: #e9e6df;
    }

    /* --- Animated gradient title --- */
    .hero-title {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 700;
        font-size: 3.2rem;
        line-height: 1.05;
        background: linear-gradient(90deg,#c9a24d,#f7e7b0,#c9a24d,#f7e7b0);
        background-size: 300% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 6s linear infinite;
        margin-bottom: .2rem;
    }
    @keyframes shimmer { to { background-position: 300% center; } }

    .hero-sub {
        font-family: 'Inter', sans-serif;
        color: #b7b1a3;
        font-size: 1rem;
        letter-spacing: .3px;
        margin-bottom: 1.2rem;
    }

    /* --- Fade-in for content blocks --- */
    .block-container { animation: fadeUp .7s ease both; }
    @keyframes fadeUp { from {opacity:0; transform: translateY(14px);} to {opacity:1; transform:none;} }

    /* --- Buttons --- */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg,#c9a24d,#a6802f);
        color: #14110a;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: .55rem 1.1rem;
        transition: transform .15s ease, box-shadow .2s ease, filter .2s ease;
        box-shadow: 0 6px 18px rgba(201,162,77,.18);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.07);
        box-shadow: 0 10px 26px rgba(201,162,77,.30);
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(201,162,77,0.18);
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(201,162,77,0.14);
        border-bottom: 2px solid #c9a24d;
        color: #f7e7b0;
    }

    /* --- Inputs / text areas --- */
    textarea, input, .stTextInput input {
        background: rgba(255,255,255,0.04) !important;
        color: #f1ede3 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(201,162,77,0.20) !important;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background: #0e0e11;
        border-right: 1px solid rgba(201,162,77,0.15);
    }

    /* --- Chat bubbles glow --- */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(201,162,77,0.12);
        border-radius: 14px;
        animation: fadeUp .4s ease both;
    }
    </style>

    <div class="hero-title">📄 PDF Studio</div>
    <div class="hero-sub">Ask • Summarize • Edit — and download your changed PDF. Free stack: Groq + LangChain.</div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
#  SIDEBAR: API KEY + UPLOAD
# =============================================================================
st.sidebar.header("⚙️ Setup")
groq_api_key = st.sidebar.text_input(
    "1. Groq API Key",
    type="password",
    value=os.environ.get("GROQ_API_KEY", ""),
    help="Get a free key at console.groq.com → API Keys",
)
st.sidebar.markdown("---")
st.sidebar.subheader("2. Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 Manual editing + PDF download work **without** a key. "
    "The key is only needed for AI chat, summaries and AI-assisted edits."
)


# =============================================================================
#  CACHED EMBEDDING MODEL
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# =============================================================================
#  LLM HELPER
# =============================================================================
def get_llm(api_key, temperature=0):
    # if this model name ever errors, check console.groq.com for the current one
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature, api_key=api_key)


# =============================================================================
#  PDF -> SEARCHABLE INDEX (for Q&A)
# =============================================================================
def build_vectorstore(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    documents = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, load_embeddings())
    return vectorstore, len(documents), len(chunks)


def build_chain(vectorstore, api_key):
    llm = get_llm(api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# =============================================================================
#  RAW TEXT EXTRACTION (per page, for the editor)
# =============================================================================
def extract_pages(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


# =============================================================================
#  SUMMARIZE
# =============================================================================
def summarize_text(full_text, style, api_key):
    llm = get_llm(api_key)
    styles = {
        "Short (TL;DR)": "Write a concise 3–4 sentence summary capturing the core message.",
        "Detailed": "Write a thorough, well-structured summary in clear paragraphs, covering all key points.",
        "Bullet points": "Summarize the document as clean, organized bullet points grouped by topic.",
    }
    instruction = styles.get(style, styles["Detailed"])

    max_chars = 90000  # llama-3.3-70b has a large context window; stuff if it fits
    if len(full_text) <= max_chars:
        return llm.invoke(f"{instruction}\n\nDocument:\n{full_text}\n\nSummary:").content

    # Map-reduce for very long documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    partials = []
    for i, c in enumerate(chunks):
        partials.append(
            llm.invoke(f"Summarize part {i+1} of {len(chunks)} of a document:\n\n{c}").content
        )
    combined = "\n\n".join(partials)
    return llm.invoke(
        f"{instruction}\n\nCombine these partial summaries into one cohesive result:\n\n{combined}\n\nSummary:"
    ).content


# =============================================================================
#  AI-ASSISTED EDIT
# =============================================================================
def ai_edit(text, instruction, api_key):
    llm = get_llm(api_key)
    prompt = (
        "You are a precise document editor. Apply the user's instruction to the text below. "
        "Return ONLY the complete edited text — no commentary, no preamble, no markdown code fences.\n\n"
        f"INSTRUCTION:\n{instruction}\n\n"
        f"CURRENT TEXT:\n{text}\n\n"
        "EDITED TEXT:"
    )
    return llm.invoke(prompt).content


# =============================================================================
#  REBUILD A PDF FROM EDITED TEXT
# =============================================================================
_FONT_NAME = None


def _doc_font():
    """Use a Unicode TTF if we can find one (better for accents / Arabic), else Helvetica."""
    global _FONT_NAME
    if _FONT_NAME:
        return _FONT_NAME
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",                  # macOS
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",                     # Windows (has Arabic)
        "C:\\Windows\\Fonts\\tahoma.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("DocFont", p))
                _FONT_NAME = "DocFont"
                return _FONT_NAME
            except Exception:
                continue
    _FONT_NAME = "Helvetica"
    return _FONT_NAME


def _has_arabic(s):
    return any("\u0600" <= ch <= "\u06FF" for ch in s)


def _shape(line):
    if _ARABIC_OK and _has_arabic(line):
        try:
            return get_display(arabic_reshaper.reshape(line))
        except Exception:
            return line
    return line


def pages_to_pdf_bytes(pages):
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
    )
    font = _doc_font()
    body = ParagraphStyle("Body", fontName=font, fontSize=11, leading=16)

    flow = []
    for pi, page in enumerate(pages):
        lines = page.split("\n") if page else [""]
        for line in lines:
            if line.strip():
                flow.append(Paragraph(escape(_shape(line)), body))
            else:
                flow.append(Spacer(1, 10))
        if pi < len(pages) - 1:
            flow.append(PageBreak())
    if not flow:
        flow.append(Paragraph("(empty document)", body))

    doc.build(flow)
    return buf.getvalue()


# =============================================================================
#  SESSION STATE
# =============================================================================
def _init_state():
    defaults = {
        "messages": [],
        "chain": None,
        "current_file": None,
        "edited_pages": [],
        "pdf_bytes": None,
        "summary": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _clear_page_keys():
    for k in list(st.session_state.keys()):
        if k.startswith("page_text_"):
            del st.session_state[k]


_init_state()


# =============================================================================
#  HANDLE A NEW UPLOAD
# =============================================================================
if uploaded_file is not None and uploaded_file.name != st.session_state.current_file:
    with st.spinner("Reading and indexing your PDF... (first run downloads a small model)"):
        file_bytes = uploaded_file.getvalue()

        # Editor text (works even without an API key)
        st.session_state.edited_pages = extract_pages(file_bytes)

        # Q&A index + chain (needs the key)
        if groq_api_key:
            vectorstore, n_pages, n_chunks = build_vectorstore(uploaded_file)
            st.session_state.chain = build_chain(vectorstore, groq_api_key)
            st.sidebar.success(f"✅ Indexed {n_pages} pages into {n_chunks} chunks.")
        else:
            st.session_state.chain = None
            st.sidebar.info("PDF loaded for editing. Add a key to enable chat & summaries.")

        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.pdf_bytes = None
        _clear_page_keys()


# =============================================================================
#  MAIN TABS
# =============================================================================
tab_chat, tab_sum, tab_edit = st.tabs(["💬  Chat / Q&A", "📝  Summarize", "✏️  Edit PDF"])


# ---------- TAB 1: CHAT ----------
with tab_chat:
    if st.session_state.current_file is None:
        st.info("👈 Upload a PDF in the sidebar to get started.")
    elif st.session_state.chain is None:
        st.warning("⬅️ Enter your Groq API key in the sidebar to enable chat.")
    else:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask a question about your PDF...")
        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = st.session_state.chain.invoke(user_q)
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------- TAB 2: SUMMARIZE ----------
with tab_sum:
    if st.session_state.current_file is None:
        st.info("👈 Upload a PDF first.")
    elif not groq_api_key:
        st.warning("⬅️ A Groq API key is required to summarize.")
    else:
        c1, c2 = st.columns([2, 1])
        with c2:
            style = st.selectbox("Summary style", ["Short (TL;DR)", "Detailed", "Bullet points"])
            if st.button("✨ Summarize", use_container_width=True):
                full_text = "\n\n".join(st.session_state.edited_pages)
                if not full_text.strip():
                    st.error("No extractable text found in this PDF (it may be scanned images).")
                else:
                    with st.spinner("Summarizing..."):
                        st.session_state.summary = summarize_text(full_text, style, groq_api_key)
        with c1:
            if st.session_state.summary:
                st.markdown("#### Summary")
                st.markdown(st.session_state.summary)
                st.download_button(
                    "⬇️ Download summary (.txt)",
                    data=st.session_state.summary.encode("utf-8"),
                    file_name="summary.txt",
                    mime="text/plain",
                )
            else:
                st.caption("Pick a style and press **Summarize**.")


# ---------- TAB 3: EDIT PDF ----------
with tab_edit:
    if st.session_state.current_file is None:
        st.info("👈 Upload a PDF first.")
    elif len(st.session_state.edited_pages) == 0:
        st.error("No extractable text found (the PDF may be scanned images).")
    else:
        pages = st.session_state.edited_pages
        st.caption(
            "Edit the text directly, or use AI to apply changes. "
            "Add or delete whole pages, then build & download the final PDF."
        )

        top = st.columns([3, 1, 1])
        with top[0]:
            idx = st.selectbox(
                "Page to edit",
                range(len(pages)),
                format_func=lambda i: f"Page {i + 1} of {len(pages)}",
            )
        with top[1]:
            if st.button("➕ Add page", use_container_width=True):
                st.session_state.edited_pages.append("")
                _clear_page_keys()
                st.rerun()
        with top[2]:
            if st.button("🗑️ Delete page", use_container_width=True):
                if len(st.session_state.edited_pages) > 1:
                    st.session_state.edited_pages.pop(idx)
                    _clear_page_keys()
                    st.rerun()
                else:
                    st.warning("Can't delete the only page.")

        # --- editable text area (state persists per page) ---
        key = f"page_text_{idx}"
        if key not in st.session_state:
            st.session_state[key] = st.session_state.edited_pages[idx]

        st.text_area("Page text", key=key, height=380)
        st.session_state.edited_pages[idx] = st.session_state[key]

        # --- AI-assisted edit ---
        st.markdown("##### 🤖 Or tell the AI what to change on this page")
        ai_cols = st.columns([4, 1])
        with ai_cols[0]:
            instruction = st.text_input(
                "Instruction",
                placeholder="e.g. 'Fix grammar and make it more formal' or 'Remove the second paragraph'",
                label_visibility="collapsed",
            )
        with ai_cols[1]:
            apply_ai = st.button("Apply AI edit", use_container_width=True)

        if apply_ai:
            if not groq_api_key:
                st.warning("⬅️ A Groq API key is required for AI edits.")
            elif not instruction.strip():
                st.warning("Type an instruction first.")
            else:
                with st.spinner("Applying your changes..."):
                    new_text = ai_edit(st.session_state[key], instruction, groq_api_key)
                st.session_state[key] = new_text
                st.session_state.edited_pages[idx] = new_text
                st.rerun()

        st.markdown("---")

        # --- build & download ---
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("📄 Build final PDF", use_container_width=True):
                with st.spinner("Rebuilding your PDF..."):
                    st.session_state.pdf_bytes = pages_to_pdf_bytes(st.session_state.edited_pages)
                st.success("Done! Download it on the right →")
        with b2:
            if st.session_state.pdf_bytes:
                out_name = (st.session_state.current_file or "document").replace(".pdf", "") + "_edited.pdf"
                st.download_button(
                    "⬇️ Download edited PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=out_name,
                    mime="application/pdf",
                    use_container_width=True,
                )
