import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# ── Create Lightweight Search Index ─────────────────────────────────────────
def get_vector_store(chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(chunks)
    st.session_state["chunks"] = chunks
    st.session_state["vectorizer"] = vectorizer
    st.session_state["vectors"] = vectors

# ── Ask Groq with Retrieved Context ─────────────────────────────────────────
def ask_groq(context, question):
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Configure it in the app secrets and restart the app.")
        st.stop()

    prompt = f"""
    Use the context below to answer the question as accurately as possible.
    If the answer is not in the context, say "I don't know based on the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ── Handle User Query ────────────────────────────────────────────────────────
def handle_query(user_question):
    vectorizer = st.session_state["vectorizer"]
    vectors = st.session_state["vectors"]
    chunks = st.session_state["chunks"]

    question_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(question_vector, vectors).flatten()
    top_indices = scores.argsort()[-3:][::-1]
    context = "\n\n".join(chunks[i] for i in top_indices if scores[i] > 0)

    if not context:
        st.warning("I couldn't find relevant text in the processed document.")
        return

    response = ask_groq(context, user_question)

    st.write("### 🤖 Answer:")
    st.success(response)

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
    if "vectors" in st.session_state:
        handle_query(user_question)
    else:
        st.warning("⚠️ Please upload and process your documents first!")
