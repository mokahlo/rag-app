import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import os
import tempfile
import pinecone
import openai

# ✅ Set up API keys (Ensure these are set in Streamlit Secrets)
OPENAI_API_KEY = st.secrets["OPENAI"]
PINECONE_API_KEY = st.secrets["PINECONE"]
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ample-traffic"

# ✅ Initialize Pinecone
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pinecone_client.list_indexes().names():
    st.error(f"⚠️ Pinecone index '{INDEX_NAME}' not found. Check Pinecone setup.")
index = pinecone_client.Index(INDEX_NAME)

# ✅ Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF uploaded via Streamlit."""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            doc = fitz.open(temp_file_path)
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()
            os.remove(temp_file_path)
            return text
        except Exception as e:
            st.error(f"⚠️ Error processing PDF: {str(e)}")
            return None
    return None

# ✅ Function to get text embeddings
def get_text_embedding(text):
    """Generate text embeddings using OpenAI's 'text-embedding-3-large' model."""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY
    )
    return response.data[0].embedding

# ✅ Streamlit Web UI
st.title("📄 Traffic Study Database with Pinecone")
st.write("Upload traffic study documents to extract text, generate embeddings, and store them in Pinecone.")

# 📂 Upload raw, annotated study, and final review letter
raw_study = st.file_uploader("Upload Raw Traffic Study (PDF)", type=["pdf"])
annotated_study = st.file_uploader("Upload Annotated Study (PDF)", type=["pdf"])
review_letter = st.file_uploader("Upload Traffic Review Letter (PDF)", type=["pdf"])

# ✅ Process and store in Pinecone
if st.button("📤 Store in Pinecone"):

    for doc_name, doc_file in [("raw_study", raw_study), 
                               ("annotated_study", annotated_study), 
                               ("review_letter", review_letter)]:
        
        if doc_file:
            st.write(f"Processing {doc_name}...")

            # ✅ Extract text
            text_content = extract_text_from_pdf(doc_file)
            if text_content:
                st.success(f"✅ Extracted text from {doc_name}")

                # ✅ Generate embeddings
                embedding_vector = get_text_embedding(text_content)
                if len(embedding_vector) == 1536:  # Ensure correct embedding size
                    st.success(f"✅ Embedding generated for {doc_name}")

                    # ✅ Store in Pinecone
                    doc_id = f"{doc_name}-{doc_file.name}"
                    index.upsert(vectors=[(doc_id, embedding_vector, {"text": text_content})])
                    st.success(f"✅ Stored '{doc_name}' in Pinecone!")
                else:
                    st.error("❌ Embedding size mismatch. Expected 1536 dimensions.")
            else:
                st.error(f"⚠️ Could not extract text from {doc_name}.")
        else:
            st.warning(f"⚠️ No file uploaded for {doc_name}.")
