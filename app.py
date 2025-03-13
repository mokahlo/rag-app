import streamlit as st
import openai
import pinecone
import os
import PyPDF2

# ✅ Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ample-traffic"

# ✅ Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# ✅ Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ✅ Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# ✅ Function to generate embeddings using OpenAI
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # ✅ Using the latest embedding model
    )
    return response.data[0].embedding  # ✅ Extract embedding vector

# ✅ Streamlit UI
st.title("🚦 Traffic Study Database")
st.write("Upload past traffic studies to store them in Pinecone for easy retrieval.")

# 📂 Upload PDF File
uploaded_file = st.file_uploader("Upload a Traffic Study (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from the PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)

    if extracted_text:
        with st.spinner("Generating embeddings..."):
            embedding_vector = get_embedding(extracted_text)

        # ✅ Store in Pinecone
        with st.spinner("Storing in Pinecone..."):
            doc_id = f"doc-{uploaded_file.name}"
            index.upsert(vectors=[(doc_id, embedding_vector, {"text": extracted_text})])

        st.success("✅ Document stored successfully in Pinecone!")
    else:
        st.error("❌ Could not extract text from the PDF. Try another file.")
