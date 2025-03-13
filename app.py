import streamlit as st
import os
import uuid
from pinecone import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader

# 🔹 Pinecone Configuration
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ample-traffic"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize OpenAI Embeddings (needed for storing vector representations)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=st.secrets["OPENAI_API_KEY"])

# 🔹 Streamlit UI Setup
st.title("🚦 Traffic Study Database Builder")
st.write("Upload a traffic study, its annotated version, and the review letter to build the AI training dataset.")

# 📂 User Inputs for Project Name
project_name = st.text_input("Enter Project Name:")
project_id = str(uuid.uuid4()) if project_name else None

# 📂 File Uploads for Study Documents
raw_study = st.file_uploader("Upload Raw Study (Consultant Submission)", type=["pdf"])
annotated_study = st.file_uploader("Upload Annotated Study (City Comments)", type=["pdf"])
review_letter = st.file_uploader("Upload Review Letter (City Response)", type=["pdf"])

# 🔹 Function to Extract Text from PDFs
def extract_text_from_pdf(file):
    if file is not None:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return ""

# 🔹 Store Documents in Pinecone
if st.button("Store in Database") and project_name and raw_study and annotated_study and review_letter:
    st.write("📄 Extracting text from PDFs...")
    
    raw_text = extract_text_from_pdf(raw_study)
    annotated_text = extract_text_from_pdf(annotated_study)
    review_text = extract_text_from_pdf(review_letter)
    
    # Convert text to embeddings
    raw_vector = embeddings.embed_query(raw_text)
    annotated_vector = embeddings.embed_query(annotated_text)
    review_vector = embeddings.embed_query(review_text)
    
    # Ensure correct dimensions
    if len(raw_vector) == 1536 and len(annotated_vector) == 1536 and len(review_vector) == 1536:
        
        # Upsert into Pinecone
        index.upsert(
            vectors=[
                (f"{project_id}_raw", raw_vector, {"text": raw_text, "type": "raw_study", "project": project_name}),
                (f"{project_id}_annotated", annotated_vector, {"text": annotated_text, "type": "annotated_study", "project": project_name}),
                (f"{project_id}_review", review_vector, {"text": review_text, "type": "review_letter", "project": project_name})
            ]
        )
        
        st.success(f"✅ Project '{project_name}' successfully stored in Pinecone!")
    else:
        st.error("❌ Embedding size mismatch. Ensure text embeddings have 1536 dimensions.")
