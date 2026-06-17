"""
PDF Studio — Ask · Summarize · Edit · Export
Free stack: Streamlit (UI) + LangChain (logic) + Groq (free LLM) + HuggingFace (free local embeddings)
Run locally with:   streamlit run app.py
"""

import io
import os
import tempfile
import streamlit as st
from docx import Document as DocxDocument

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ======================= PAGE + THEME =======================
st.set_page_config(page_title="PDF Studio", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Inter:wght@400;500&display=swap');

.stApp{
  background: radial-gradient(1200px 600px at 20% -10%, #1c1c22 0%, #0d0d0f 55%);
  color:#ececec; font-family:'Inter',sans-serif;
}
.hero-title{
  font-family:'Cormorant Garamond',serif; font-size:3.2rem; font-weight:700; text-align:center;
  background:linear-gradient(90deg,#d4af37,#f7e7a6,#d4af37,#bfa14a);
  background-size:300% 100%; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; animation:shimmer 6s ease infinite; margin-bottom:0;
}
@keyframes shimmer{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero-sub{text-align:center; color:#b9b9b9; margin-top:.2rem; letter-spacing:1px; font-size:1rem;}
.block-container{ animation:fadeIn .8s ease both; }
@keyframes fadeIn{from{opacity:0; transform:translateY(10px)} to{opacity:1; transform:none}}

.stButton>button, .stDownloadButton>button{
  background:linear-gradient(135deg,#d4af37,#b8941f); color:#0d0d0f; border:none;
  border-radius:10px; font-weight:600; padding:.55rem 1.1rem; transition:all .25s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-2px); box-shadow:0 6px 22px rgba(212,175,55,.45); filter:brightness(1.05);
}
.stTabs [data-baseweb="tab"]{ background:#16161b; border-radius:8px 8px 0 0; color:#c9c9c9; padding:8px 18px; }
.stTabs [aria-selected="true"]{ background:#23231c; color:#d4af37; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">PDF Studio</div>'
            '<div class="hero-sub">ASK · SUMMARIZE · EDIT · EXPORT</div>', unsafe_allow_html=True)
st.write("")


# ======================= SIDEBAR =======================
st.sidebar.header("Setup")
groq_api_key = st.sidebar.text_input(
    "1. Groq API Key", type="password",
    value=os.environ.get("GROQ_API_KEY", ""),
    help="Free key from console.groq.com -> API Keys",
)
st.sidebar.markdown("---")
st.sidebar.subheader("2. Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
key = groq_api_key


# ======================= HELPERS (THE BACKEND) =======================
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def make_llm(api_key):
    # if "llama-3.3-70b-versatile" ever errors, check console.groq.com for the current model name
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


def build_vectorstore(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    documents = PyPDFLoader(tmp_path).load()
    full_text = "\n\n".join(d.page_content for d in documents)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, load_embeddings())
    return vectorstore, full_text, len(documents), len(chunks)


def build_chain(vectorstore, api_key):
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
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | make_llm(api_key) | StrOutputParser()
    )


def summarize_text(text, llm):
    # For long PDFs, summarize section by section, then combine (map-reduce)
    blocks = [text[i:i + 6000] for i in range(0, len(text), 6000)]
    if len(blocks) == 1:
        return llm.invoke("Write a clear, structured summary of this document:\n\n" + text).content
    partials = [llm.invoke("Summarize this section:\n\n" + b).content for b in blocks]
    combined = "\n\n".join(partials)
    return llm.invoke("Combine these section summaries into one clear overall summary:\n\n" + combined).content


def apply_ai_edit(text, instruction, llm):
    prompt = (
        "You are editing a document. Apply the user's instruction and return the FULL updated "
        "document text only - no commentary, no preamble, no markdown fences.\n\n"
        f"INSTRUCTION:\n{instruction}\n\nCURRENT DOCUMENT:\n{text}"
    )
    return llm.invoke(prompt).content


def to_docx_bytes(text):
    doc = DocxDocument()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ======================= SESSION STATE =======================
for k, v in {"messages": [], "chain": None, "current_file": None,
             "full_text": None, "edited_text": "", "summary": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ======================= PROCESS UPLOAD =======================
if uploaded_file is not None:
    if not key:
        st.warning("Enter your Groq API key in the sidebar to continue.")
    elif uploaded_file.name != st.session_state.current_file:
        with st.spinner("Reading and indexing your PDF... (first run downloads a small model)"):
            vs, full_text, n_pages, n_chunks = build_vectorstore(uploaded_file)
            st.session_state.chain = build_chain(vs, key)
            st.session_state.full_text = full_text
            st.session_state.edited_text = full_text
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            st.session_state.summary = ""
        st.sidebar.success(f"Indexed {n_pages} pages into {n_chunks} chunks.")


ready = st.session_state.full_text is not None


# ======================= TABS =======================
tab_chat, tab_sum, tab_edit = st.tabs(["Ask", "Summarize", "Edit & Export"])

# ---------- TAB 1: ASK ----------
with tab_chat:
    if not ready:
        st.info("Enter your API key and upload a PDF to start.")
    else:
        q = st.text_input("Ask a question about your PDF:", key="chat_q")
        if st.button("Send", key="chat_send") and q.strip():
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                ans = st.session_state.chain.invoke(q)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

# ---------- TAB 2: SUMMARIZE ----------
with tab_sum:
    if not ready:
        st.info("Upload a PDF first.")
    else:
        if st.button("Summarize this PDF"):
            with st.spinner("Reading the whole document and summarizing..."):
                st.session_state.summary = summarize_text(st.session_state.full_text, make_llm(key))
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
            st.download_button("Download summary (.txt)",
                               data=st.session_state.summary, file_name="summary.txt")

# ---------- TAB 3: EDIT & EXPORT ----------
with tab_edit:
    if not ready:
        st.info("Upload a PDF first.")
    else:
        st.caption("Edit the text directly below, or tell the AI what to change. Then download the final file.")
        edited = st.text_area("Document text:", value=st.session_state.edited_text, height=420)
        st.session_state.edited_text = edited   # capture manual edits

        instruction = st.text_input(
            "Tell the AI what to change "
            "(e.g. 'fix grammar', 'remove section 3', 'add a 30-day payment-terms clause'):"
        )
        c1, c2 = st.columns(2)
        if c1.button("Apply AI changes"):
            if instruction.strip():
                with st.spinner("Applying your changes..."):
                    st.session_state.edited_text = apply_ai_edit(
                        st.session_state.edited_text, instruction, make_llm(key))
                st.rerun()   # refresh the text box to show the AI's result
            else:
                st.warning("Type an instruction first.")
        if c2.button("Reset to original"):
            st.session_state.edited_text = st.session_state.full_text
            st.rerun()

        st.markdown("##### Download the final file")
        d1, d2 = st.columns(2)
        d1.download_button(
            "Word (.docx)", data=to_docx_bytes(st.session_state.edited_text),
            file_name="edited_document.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        d2.download_button(
            "Text (.txt)", data=st.session_state.edited_text, file_name="edited_document.txt",
        )
