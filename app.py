import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Page Config (ONLY ONCE!) ───────────────────────────────────────────────
st.set_page_config(page_title="AI Doc Chatbot", page_icon="📄", layout="wide")
st.title("📄 AI Doc Chatbot — Powered by Groq ⚡")

if not GROQ_API_KEY:
    st.warning("Add GROQ_API_KEY in deployment secrets to enable answers.")

# ── Extract text from PDFs ───────────────────────────────────────────────────
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# ── Split text into chunks ───────────────────────────────────────────────────
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# ── Create FAISS Vector Store ────────────────────────────────────────────────
def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# ── Build QA Chain with Groq ─────────────────────────────────────────────────
def get_qa_chain():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Configure it in the app secrets and restart the app.")
        st.stop()

    prompt_template = """
    Use the context below to answer the question as accurately as possible.
    If the answer is not in the context, say "I don't know based on the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# ── Handle User Query ────────────────────────────────────────────────────────
def handle_query(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=3)

    chain = get_qa_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("### 🤖 Answer:")
    st.success(response["output_text"])

# ── Sidebar — Upload PDFs ────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Documents")
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("⚙️ Process Documents"):
        if pdf_files:
            with st.spinner("Processing... please wait ⏳"):
                raw_text = get_pdf_text(pdf_files)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("✅ Documents processed successfully!")
        else:
            st.warning("⚠️ Please upload at least one PDF file.")

# ── Main — Ask Question ───────────────────────────────────────────────────────
user_question = st.text_input("💬 Ask a question about your document:")

if user_question:
    if os.path.exists("faiss_index"):
        handle_query(user_question)
    else:
        st.warning("⚠️ Please upload and process your documents first!")
