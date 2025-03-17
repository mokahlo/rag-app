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
def extract_text_from_pdf(pdf_file):
    """Extracts text and annotations from a PDF file."""
    text = []
    annotations = []

    with fitz.open(pdf_file) as doc:
        for page in doc:
            text.append(page.get_text())  # Extracts page text
            for annot in page.annots():  # Extracts annotations
                if annot:
                    annotations.append(annot.info["content"])

    return " ".join(text), " ".join(annotations)

# ✅ Function to Process & Store Documents in Pinecone
def process_and_store(uploaded_file, file_type):
    """Processes the uploaded PDF and stores embeddings in Pinecone."""
    if uploaded_file is None:
        return

    # 🔹 Generate Unique Project Name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"{file_type}_{timestamp}"

    # 🔹 Save Uploaded File Temporarily
    file_path = f"temp_{project_name}.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 🔹 Extract Text and Annotations
    extracted_text, extracted_annotations = extract_text_from_pdf(file_path)

    # 🔹 Generate Embeddings (Fixed Extraction)
    response = openai_client.embeddings.create(
        input=extracted_text + " " + extracted_annotations,
        model="text-embedding-ada-002"
    )
    embedding_vector = response.data[0].embedding  # ✅ FIXED

    # 🔹 Store in Pinecone (Annotations Included in Metadata)
    doc_id = hashlib.md5(project_name.encode()).hexdigest()
    index.upsert(vectors=[(doc_id, embedding_vector, {"text": extracted_text, "annotations": extracted_annotations})])

    # ✅ Success Message
    st.success(f"✅ {file_type} successfully processed and stored in Pinecone!")

# ✅ Streamlit UI
st.title("🚦 Traffic Study Learning")
st.write("Upload traffic study documents. The app will extract text & annotations and store them in a vector database.")

# **🔹 Upload Files**
raw_study = st.file_uploader("Upload Raw Traffic Study", type=["pdf"])
annotated_study = st.file_uploader("Upload Annotated Traffic Study", type=["pdf"])
review_letter = st.file_uploader("Upload Final Review Letter", type=["pdf"])

# **🔹 Process & Store in Pinecone**
if st.button("Store in Vector Database"):
    process_and_store(raw_study, "raw_study")
    process_and_store(annotated_study, "annotated_study")
    process_and_store(review_letter, "review_letter")
