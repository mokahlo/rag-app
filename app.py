import streamlit as st
import os
import openai  # OpenAI API for embeddings
import pinecone  # Pinecone for vector storage
import fitz  # PyMuPDF for PDF text and annotation extraction
import hashlib # Used for generating unique document IDs
import datetime # For timestamp-based naming
import json  # For handling annotation metadata

# ✅ Load API keys from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI"]
PINECONE_API_KEY = st.secrets["PINECONE"]

# ✅ Configure Pinecone
PINECONE_ENV = "us-east-1"  # Pinecone region
INDEX_NAME = "ample-traffic"  # Pinecone index name

# ✅ Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ✅ OpenAI Embeddings Configuration
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Ensure the "temp" directory exists
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ✅ Function to Extract Text & Annotations from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts both plain text and annotations from a given PDF file."""
    text_content = []
    annotations = []

    try:
        doc = fitz.open(pdf_path)  # ✅ Open the PDF
        for page in doc:
            text_content.append(page.get_text("text"))  # ✅ Extract text from page
            for annot in page.annots():  # ✅ Extract annotations (if any)
                annotations.append(annot.info["content"])  # ✅ Store annotation text
    except Exception as e:
        st.error(f"❌ Error extracting text: {e}")
        return "", ""

    return " ".join(text_content), " ".join(annotations)

# ✅ Function to Process & Store Documents in Pinecone
def process_and_store(file_path, file_type, project_name):
    """Extracts text & annotations from a file and stores them in Pinecone with metadata."""
    extracted_text, extracted_annotations = extract_text_from_pdf(file_path)

    # ✅ Generate a unique document ID
    doc_id = hashlib.md5((project_name + file_type).encode()).hexdigest()

    # ✅ OpenAI Embedding API Call
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",  # ✅ Matches 1536-dimension requirement
        input=extracted_text + " " + extracted_annotations  # ✅ Combine text & annotations
    )

    embedding_vector = response.data[0].embedding  # ✅ Correctly extract embedding

    # ✅ Store in Pinecone
    index.upsert(
        vectors=[(doc_id, embedding_vector, {"text": extracted_text, "annotations": extracted_annotations, "project_name": project_name})]
    )
    st.success(f"✅ Stored {file_type} for project '{project_name}' in Pinecone!")

# ✅ Upload and Store Documents
st.header("📂 Upload Traffic Study Documents")
project_name = st.text_input("Enter project name:")

raw_study = st.file_uploader("Upload Raw Study (Consultant Submission)", type=["pdf"])
annotated_study = st.file_uploader("Upload Annotated Study (City Edits)", type=["pdf"])
review_letter = st.file_uploader("Upload Final Traffic Review Letter", type=["pdf"])

# ✅ Process uploaded files if project name is provided
if st.button("Store in Pinecone"):
    if project_name:
        if raw_study: process_and_store(raw_study, "raw_study", project_name)
        if annotated_study: process_and_store(annotated_study, "annotated_study", project_name)
        if review_letter: process_and_store(review_letter, "review_letter", project_name)
    else:
        st.error("❌ Please enter a project name before storing data.")
