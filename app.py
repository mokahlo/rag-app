import streamlit as st
import os
import fitz  # PyMuPDF for text extraction
import openai
import pinecone
from pdf2image import convert_from_path
from pytesseract import image_to_string  # OCR for extracting text from figures
from pinecone import Pinecone

# 🔹 Set up API keys
api_keys = {
    "OPENAI": st.secrets["OPENAI_API_KEY"],
    "PINECONE": st.secrets["PINECONE_API_KEY"]
}
pinecone_env = "us-east-1"
index_name = "ample-traffic"

# 🔹 Initialize Pinecone
pc = Pinecone(api_key=api_keys["PINECONE"])
index = pc.Index(index_name)

# Function: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function: Extract text from figures using OCR
def extract_text_from_images(pdf_path):
    images = convert_from_path(pdf_path)
    extracted_text = ""
    for image in images:
        extracted_text += image_to_string(image) + "\n"
    return extracted_text

# Function: Generate text embeddings
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding  # Ensuring 1536 dimensions

# Function: Process and upload to Pinecone
def process_and_store(pdf_path, doc_id):
    text_content = extract_text_from_pdf(pdf_path)
    figure_text = extract_text_from_images(pdf_path)
    
    full_text = text_content + "\n[Figures]\n" + figure_text
    embedding_vector = get_embedding(full_text)

    if len(embedding_vector) == 1536:
        index.upsert(vectors=[(doc_id, embedding_vector, {"text": full_text})])
        st.success(f"✅ {doc_id} successfully stored in Pinecone!")
    else:
        st.error(f"❌ Embedding size mismatch. Got {len(embedding_vector)}, expected 1536.")

# 🔹 Streamlit UI
st.title("🚦 Traffic Study Database Builder")
st.write("Upload raw studies, annotated versions, and final review letters.")

# Upload files
raw_study = st.file_uploader("📂 Upload Raw Study", type=["pdf"])
annotated_study = st.file_uploader("📂 Upload Annotated Study", type=["pdf"])
review_letter = st.file_uploader("📂 Upload Final Review Letter", type=["pdf"])

if st.button("🚀 Process and Store in Pinecone"):
    if raw_study and annotated_study and review_letter:
        process_and_store(raw_study, "raw_study")
        process_and_store(annotated_study, "annotated_study")
        process_and_store(review_letter, "review_letter")
        st.success("✅ All documents stored successfully!")
    else:
        st.error("❌ Please upload all three required files!")

