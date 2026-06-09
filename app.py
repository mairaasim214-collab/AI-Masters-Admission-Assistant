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

# ---------- Helper Function: Format Student Profile ----------
def format_student_profile():
    if "student_profile" not in st.session_state:
        return "No student profile provided."

    profile = st.session_state.student_profile

    formatted_profile = f"""
Student Profile:
- Degree: {profile.get("degree", "Not provided")}
- GPA: {profile.get("gpa", "Not provided")}
- IELTS Score: {profile.get("ielts", "Not provided")}
- German Level: {profile.get("german_level", "Not provided")}
- ECTS / Credit Hours: {profile.get("ects_credits", "Not provided")}
- Target Field: {profile.get("target_field", "Not provided")}
- Notes: {profile.get("notes", "Not provided")}
"""

    return formatted_profile


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

# ---------- Student Profile Section ----------
st.markdown(
    '<div class="section-title">👤 Student Profile</div>',
    unsafe_allow_html=True
)

st.write(
    "Add your academic profile so the assistant can give eligibility-focused answers."
)

profile_col1, profile_col2 = st.columns(2)

with profile_col1:
    degree = st.text_input(
        "Degree / Background",
        placeholder="Example: Computer Engineering"
    )

    gpa = st.text_input(
        "GPA / Grade",
        placeholder="Example: 2.8 German grade or 3.1/4.0"
    )

    ielts = st.text_input(
        "IELTS / English Score",
        placeholder="Example: IELTS 7"
    )

with profile_col2:
    german_level = st.text_input(
        "German Level",
        placeholder="Example: A1 planned / A2 completed / Not required"
    )

    ects_credits = st.text_input(
        "ECTS / Credit Hours",
        placeholder="Example: 136 credit hours or 240 ECTS"
    )

    target_field = st.text_input(
        "Target Field",
        placeholder="Example: AI, Data Science, Computer Science"
    )

notes = st.text_area(
    "Additional Notes",
    placeholder="Example: AI/ML internship, software engineering experience, ML projects."
)

if st.button("Save Student Profile", key="save_student_profile_button"):
    st.session_state.student_profile = {
        "degree": degree,
        "gpa": gpa,
        "ielts": ielts,
        "german_level": german_level,
        "ects_credits": ects_credits,
        "target_field": target_field,
        "notes": notes
    }

    st.success("Student profile saved successfully.")

if "student_profile" in st.session_state:
    profile = st.session_state.student_profile

    st.info(
        f"Saved profile: {profile.get('degree', 'Not provided')} | "
        f"GPA: {profile.get('gpa', 'Not provided')} | "
        f"IELTS: {profile.get('ielts', 'Not provided')} | "
        f"Target: {profile.get('target_field', 'Not provided')}"
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

            student_profile_text = format_student_profile()

            prompt = f"""
            You are an AI admission assistant for Master's applicants.

            Your task is to answer questions using ONLY:
            1. The provided university requirement text
            2. The provided student profile

            Do not use outside knowledge.

            If the answer is not clearly available in the university requirement text, say:
            "I could not find this information clearly in the provided university requirement source."

            When the question asks about eligibility, give a structured assessment using this format:

            Eligibility Assessment:
            - Overall status: Likely eligible / Possibly eligible / Risky / Not enough information
            - Degree/background match:
            - GPA/grade requirement:
            - English language requirement:
            - German language requirement:
            - ECTS/credit requirement:
            - Missing or unclear requirements:
            - Final recommendation:

            Use these rules for the overall status:
            - Use "Likely eligible" only if the university requirement text clearly confirms that the student's degree/background, GPA/grade, English requirement, German requirement if any, and ECTS/credit requirement are all satisfied.
            - Use "Possibly eligible" if the student's profile seems related or promising, but one or more key requirements are missing, unclear, or not explicitly stated in the provided text.
            - Use "Risky" if the provided text shows a requirement that the student may not meet.
            - Use "Not enough information" if the provided text does not contain enough admission criteria to assess eligibility.

        Do not mark the student as "Likely eligible" when GPA, ECTS/credits, or language requirements are missing or unclear.

            Be careful. Do not guarantee admission. Explain that final eligibility depends on the university's official evaluation.

            Student profile:
            {student_profile_text}

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