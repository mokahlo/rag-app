import streamlit as st
import os
import openai  # OpenAI API for embeddings
import pinecone  # Pinecone for vector storage
import fitz  # PyMuPDF for PDF text and annotation extraction
import json  # For handling annotation metadata

# ✅ Load API keys from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI"]
PINECONE_API_KEY = st.secrets["PINECONE"]
PINECONE_ENV = "us-east-1"  # Pinecone region
INDEX_NAME = "ample-traffic"  # Pinecone index name

# ✅ Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ✅ Ensure the "temp" directory exists
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ✅ Function to generate a unique project name
def generate_project_name(file_names):
    """Generates a project name based on uploaded file names."""
    concatenated_names = "_".join(sorted(file_names))  # Sort to maintain consistency
    project_hash = hashlib.md5(concatenated_names.encode()).hexdigest()[:8]  # Short unique identifier
    return f"Project_{project_hash}"

# ✅ Function to extract text and annotations from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text and annotations from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    annotations = []

    for page in doc:
        text += page.get_text("text") + "\n"

        # Extract annotations
        for annot in page.annots():
            if annot:
                annot_data = {
                    "page": page.number,
                    "text": annot.info.get("content", ""),
                    "rect": annot.rect
                }
                annotations.append(annot_data)

    return text, annotations

# ✅ Function to process and store documents
def process_and_store(uploaded_file, doc_type):
    """Processes a PDF, extracts text and annotations, and stores embeddings in Pinecone."""
    
    if uploaded_file is None:
        st.warning(f"⚠️ No file uploaded for {doc_type}")
        return

    # Save uploaded file
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text and annotations
    extracted_text, annotations = extract_text_from_pdf(file_path)

    # ✅ Generate embeddings using OpenAI's new API method
    response = openai.embeddings.create(
        model="text-embedding-3-large",  # Ensure correct embedding model
        input=extracted_text
    )
    embedding_vector = response.data[0].embedding  # Extract the embedding

    # ✅ Store embeddings and annotations in Pinecone
    doc_id = f"{doc_type}-{uploaded_file.name}"
    index.upsert(vectors=[(doc_id, embedding_vector, {"text": extracted_text, "annotations": json.dumps(annotations)})])

    st.success(f"✅ {doc_type} stored in Pinecone with embeddings and annotations!")

# ✅ Streamlit UI
st.title("🚦 Traffic Study AI Database")
st.write("Upload traffic studies and their annotations for AI processing.")

# 📂 Upload files for database storage
st.header("📥 Upload Documents")

# Three files: raw study, annotated study, review letter
raw_study = st.file_uploader("Upload Raw Traffic Study (Consultant Submission)", type=["pdf"])
annotated_study = st.file_uploader("Upload Annotated Traffic Study (City Edits)", type=["pdf"])
review_letter = st.file_uploader("Upload Final Review Letter", type=["pdf"])

if st.button("📤 Process and Store Documents"):
    process_and_store(raw_study, "raw_study")
    process_and_store(annotated_study, "annotated_study")
    process_and_store(review_letter, "review_letter")

st.success("✅ All uploaded documents have been processed and stored!")

# ✅ Show extracted annotations from stored PDFs
st.header("📑 Extracted Annotations")
selected_doc = st.selectbox("Select a document to view annotations:", ["raw_study", "annotated_study", "review_letter"])

if selected_doc:
    # Retrieve stored document from Pinecone
    result = index.fetch(ids=[selected_doc])
    if result and result.vectors:
        data = result.vectors[selected_doc].metadata
        annotations = json.loads(data.get("annotations", "[]"))

        if annotations:
            st.write("### 📝 Annotations")
            for annot in annotations:
                st.write(f"**Page {annot['page']}:** {annot['text']}")
        else:
            st.write("❌ No annotations found for this document.")
    else:
        st.write("❌ Document not found in Pinecone.")

