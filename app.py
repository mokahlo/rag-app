import streamlit as st  # Web UI framework
import os  # File management
import fitz  # PyMuPDF for PDF text and annotation extraction
from pinecone import Pinecone  # Pinecone vector database
import openai  # OpenAI API for embeddings

# ========================
# ✅ Load API Keys from Streamlit Secrets
# ========================
OPENAI_API_KEY = st.secrets["OPENAI"]
PINECONE_API_KEY = st.secrets["PINECONE"]
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ample-traffic"  # Pinecone index name

# ========================
# ✅ Initialize Pinecone Client
# ========================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)  # Connect to Pinecone index

# ========================
# ✅ Function: Extract Text & Annotations from PDF
# ========================
def extract_text_from_pdf(uploaded_file):
    """
    Extracts both text and annotations (comments, highlights) from a PDF.
    - Extracts raw text content from each page.
    - Retrieves annotations such as comments, sticky notes, and highlights.
    - Combines extracted text and annotations into one document.
    """
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "\n".join([page.get_text("text") for page in doc])  # Extract text

        # Extract annotations (comments, highlights)
        annotations = []
        for page in doc:
            for annot in page.annots():
                if annot.info["content"]:  # Check if annotation has content
                    annotations.append(f"Annotation on Page {page.number + 1}: {annot.info['content']}")

        # Combine text and annotations (if available)
        full_text = text + "\n\n" + "\n".join(annotations) if annotations else text

    return full_text

# ========================
# ✅ Function: Store Processed Data in Pinecone
# ========================
def process_and_store(pdf_file, doc_type):
    """
    Processes a PDF file by extracting text and annotations, then stores them in Pinecone.
    - Uses OpenAI's text embedding model.
    - Generates embeddings (1536 dimensions) for the extracted text.
    - Stores embeddings and original text in Pinecone for future AI training.
    """
    if pdf_file:
        st.write(f"📄 Processing {doc_type}...")

        # Extract text & annotations from the PDF
        extracted_text = extract_text_from_pdf(pdf_file)

        # Generate OpenAI embeddings (1536 dimensions)
        embedding_model = "text-embedding-3-large"
        embedding_vector = openai.Embedding.create(
            input=extracted_text,
            model=embedding_model,
            api_key=OPENAI_API_KEY
        )["data"][0]["embedding"]

        # Store in Pinecone
        doc_id = f"{doc_type}_{pdf_file.name}"  # Unique ID
        index.upsert(vectors=[(doc_id, embedding_vector, {"text": extracted_text})])

        st.success(f"✅ {doc_type} successfully stored in Pinecone!")

# ========================
# ✅ Streamlit Web Interface
# ========================
st.title("🚦 AI-Powered Traffic Study Database")
st.write("Upload traffic study documents to store them in Pinecone for AI training.")

# Section: Upload PDFs
st.header("📑 Upload Traffic Study Documents")
raw_study = st.file_uploader("Upload Raw Traffic Study (Consultant Submission)", type=["pdf"])
annotated_study = st.file_uploader("Upload Annotated Study (City Comments)", type=["pdf"])
review_letter = st.file_uploader("Upload Final Review Letter", type=["pdf"])

# Button to Process & Store Documents
if st.button("📂 Process & Store Documents"):
    if raw_study:
        process_and_store(raw_study, "raw_study")
    if annotated_study:
        process_and_store(annotated_study, "annotated_study")
    if review_letter:
        process_and_store(review_letter, "review_letter")

st.success("✅ All uploaded documents have been processed and stored!")