import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()


# ---------- Helper Function: Clear Previous Source ----------
def clear_previous_source():
    keys_to_clear = [
        "full_text",
        "chunks",
        "vectorstore",
        "source_type",
        "source_name",
        "last_answer",
        "last_sources"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


# ---------- Helper Function: Website Text Extraction ----------
def extract_text_from_website(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noisy website elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    text = soup.get_text(separator="\n")

    # Clean empty lines and extra spaces
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    return clean_text


# ---------- Helper Function: Create Vector Database ----------
def create_vectorstore_from_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    collection_name = f"admission_source_{uuid.uuid4().hex}"

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )

    return chunks, vectorstore


# ---------- Page Configuration ----------
st.set_page_config(
    page_title="AI Master's Admission Assistant",
    page_icon="🎓",
    layout="wide"
)


# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 8px;
    }

    .subtitle {
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 30px;
    }

    .info-card {
        background-color: #f8fafc;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .small-text {
        color: #6b7280;
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🎓 Admission AI")
    st.write("A RAG-based assistant for Master's admission requirements.")

    st.markdown("---")

    st.subheader("Current Version")
    st.write("Version 4: PDF + Website RAG Answer Generation")

    st.subheader("Features Completed")
    st.write("✅ PDF upload")
    st.write("✅ PDF text extraction")
    st.write("✅ Website link input")
    st.write("✅ Website text extraction")
    st.write("✅ Text chunking")
    st.write("✅ Embeddings")
    st.write("✅ Vector search")
    st.write("✅ AI answer generation")

    st.subheader("Features Planned")
    st.write("🔜 Eligibility checker")
    st.write("🔜 Student profile form")
    st.write("🔜 University comparison")


# ---------- Main Header ----------
st.markdown(
    '<div class="main-title">🎓 AI Master’s Admission Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask admission and eligibility questions from official university PDFs or program webpages.</div>',
    unsafe_allow_html=True
)


# ---------- Intro Cards ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>📄 Upload PDFs</h3>
            <p class="small-text">Use official admission PDFs, program brochures, or requirement documents.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🔗 Paste Links</h3>
            <p class="small-text">Add university program URLs and extract requirements from webpages.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <h3>🤖 Ask Questions</h3>
            <p class="small-text">Ask about IELTS, GPA, ECTS, deadlines, documents, and eligibility.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------- Input Section ----------
st.markdown(
    '<div class="section-title">Choose Your Requirement Source</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["📄 Upload PDF", "🔗 Paste Website Link"])


# ---------- PDF Upload Tab ----------
with tab1:
    st.subheader("Upload University Requirement PDF")

    uploaded_file = st.file_uploader(
        "Upload a university admission PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:
        st.success(f"PDF uploaded successfully: {uploaded_file.name}")

        st.write("File details:")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File type:** {uploaded_file.type}")
        st.write(f"**File size:** {round(uploaded_file.size / 1024, 2)} KB")

        if st.button("Process PDF", key="process_pdf_button"):
            clear_previous_source()

            with st.spinner("Extracting text from PDF..."):
                reader = PdfReader(uploaded_file)

                full_text = ""

                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()

                    if page_text:
                        full_text += f"\n\n--- Page {page_number} ---\n\n"
                        full_text += page_text

            if full_text.strip():
                st.success("Text extracted successfully.")

                with st.spinner("Splitting text, creating embeddings, and building vector database..."):
                    chunks, vectorstore = create_vectorstore_from_text(full_text)

                st.session_state.full_text = full_text
                st.session_state.chunks = chunks
                st.session_state.vectorstore = vectorstore
                st.session_state.source_type = "PDF"
                st.session_state.source_name = uploaded_file.name

                st.success(f"Text split successfully into {len(chunks)} chunks.")
                st.success("Embeddings created and vector database is ready.")

                st.subheader("Extracted Text Preview")
                st.text_area(
                    "Preview of extracted text",
                    full_text[:3000],
                    height=250
                )

                st.subheader("Chunk Preview")

                for i, chunk in enumerate(chunks[:3], start=1):
                    with st.expander(f"Chunk {i}"):
                        st.write(chunk)

            else:
                st.error("No text could be extracted from this PDF.")


# ---------- Website Link Tab ----------
with tab2:
    st.subheader("Paste University Program Website Link")

    website_url = st.text_input(
        "Enter the official university program or admission requirements URL",
        placeholder="https://www.university.edu/program/admission-requirements"
    )

    if website_url:
        st.success("Website link received.")
        st.write(f"**URL:** {website_url}")

        if st.button("Process Website Link", key="process_website_button"):
            clear_previous_source()

            try:
                with st.spinner("Extracting text from website..."):
                    website_text = extract_text_from_website(website_url)

                if website_text.strip():
                    st.success("Website text extracted successfully.")

                    with st.spinner("Splitting website text, creating embeddings, and building vector database..."):
                        chunks, vectorstore = create_vectorstore_from_text(website_text)

                    st.session_state.full_text = website_text
                    st.session_state.chunks = chunks
                    st.session_state.vectorstore = vectorstore
                    st.session_state.source_type = "Website"
                    st.session_state.source_name = website_url

                    st.success(f"Website text split successfully into {len(chunks)} chunks.")
                    st.success("Embeddings created and vector database is ready.")

                    st.subheader("Website Text Preview")
                    st.text_area(
                        "Preview of extracted website text",
                        website_text[:3000],
                        height=250
                    )

                    st.subheader("Chunk Preview")

                    for i, chunk in enumerate(chunks[:3], start=1):
                        with st.expander(f"Website Chunk {i}"):
                            st.write(chunk)

                else:
                    st.error("No useful text could be extracted from this website.")

            except Exception as e:
                st.error("Website text extraction failed.")
                st.write("Some websites block scraping or load content using JavaScript.")
                st.write("Try another official university page first.")
                st.write("Error details:")
                st.code(str(e))


# ---------- Question Section ----------
st.markdown("---")
st.markdown(
    '<div class="section-title">Ask Admission Questions</div>',
    unsafe_allow_html=True
)

if "vectorstore" in st.session_state:
    st.success(
        f"{st.session_state.source_type} source is ready: "
        f"{st.session_state.source_name}"
    )
    st.write(f"Total chunks available: {len(st.session_state.chunks)}")
else:
    st.warning("Please upload a PDF or process a website link first.")

question = st.text_input(
    "Ask a question",
    placeholder="Example: What are the admission requirements of this program?"
)

if question:
    if "vectorstore" in st.session_state:
        with st.spinner("Searching the source and generating answer..."):
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            relevant_docs = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in relevant_docs]
            )

            prompt = f"""
You are an AI admission assistant.

Your task is to answer questions using ONLY the provided university requirement text.
Do not use outside knowledge.

If the answer is not clearly available in the provided text, say:
"I could not find this information clearly in the provided document."

Write your answer in a clear, structured, student-friendly way.

Question:
{question}

University requirement text:
{context}

Answer:
"""

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2
            )

            response = llm.invoke(prompt)

        st.session_state.last_answer = response.content
        st.session_state.last_sources = relevant_docs

        st.subheader("AI Answer")
        st.write(response.content)

        st.subheader("Source Sections Used")

        for i, doc in enumerate(relevant_docs, start=1):
            with st.expander(f"Source Section {i}"):
                st.write(doc.page_content)

    else:
        st.error("Please upload a PDF or process a website link before asking questions.")