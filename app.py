"""
PDF Q&A Bot — Streamlit UI
Upload a PDF, ask questions about it.
Stack: Streamlit (UI) + LangChain (logic) + Groq (free LLM) + HuggingFace (free local embeddings)

Run it with:   streamlit run app.py
"""

import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ---------- PAGE SETUP ----------
st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="centered")
st.title("📄 PDF Q&A Bot")
st.caption("Upload a PDF, then ask questions about it. Free stack: Groq + LangChain.")


# ---------- SIDEBAR: API KEY + UPLOAD ----------
st.sidebar.header("Setup")
groq_api_key = st.sidebar.text_input(
    "1. Groq API Key",
    type="password",
    value=os.environ.get("GROQ_API_KEY", ""),
    help="Get a free key at console.groq.com → API Keys",
)
st.sidebar.markdown("---")
st.sidebar.subheader("2. Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")


# ---------- LOAD EMBEDDING MODEL ONCE (cached so it doesn't reload) ----------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------- BACKEND: TURN THE PDF INTO A SEARCHABLE INDEX ----------
def build_vectorstore(file):
    # Save the uploaded file to a temporary file on disk so the loader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    # 1. Load every page of the PDF
    documents = PyPDFLoader(tmp_path).load()

    # 2. Split into overlapping chunks so retrieval is precise
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # 3. Embed the chunks (free, runs locally) and store them in FAISS
    vectorstore = FAISS.from_documents(chunks, load_embeddings())
    return vectorstore, len(documents), len(chunks)


# ---------- BACKEND: BUILD THE RAG CHAIN (retrieve -> prompt -> LLM) ----------
def build_chain(vectorstore, api_key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # if this errors, check console.groq.com for the current model name
        temperature=0,
        api_key=api_key,
    )

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


# ---------- SESSION STATE (Streamlit re-runs the whole script on every click,
#            so we store the chain + chat history here to keep them alive) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None


# ---------- PROCESS THE UPLOAD ----------
if uploaded_file is not None:
    if not groq_api_key:
        st.warning("⬅️ Enter your Groq API key in the sidebar to continue.")
    elif uploaded_file.name != st.session_state.current_file:
        # New file detected -> build a fresh index and chain
        with st.spinner("Reading and indexing your PDF... (first run downloads a small model)"):
            vectorstore, n_pages, n_chunks = build_vectorstore(uploaded_file)
            st.session_state.chain = build_chain(vectorstore, groq_api_key)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []  # clear old chat for the new doc
        st.sidebar.success(f"✅ Indexed {n_pages} pages into {n_chunks} chunks.")


# ---------- CHAT UI ----------
# Show the conversation so far
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if st.session_state.chain is None:
    st.info("👈 Enter your API key and upload a PDF to get started.")
else:
    user_q = st.chat_input("Ask a question about your PDF...")
    if user_q:
        # Show the user's message
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Get and show the answer from the backend chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(user_q)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
