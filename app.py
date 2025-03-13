import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import Pinecone  # ✅ Corrected Import
from langchain_openai.embeddings import OpenAIEmbeddings
import pinecone

# ✅ Secure API Key Handling
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", None)
pinecone_index_name = "ample-traffic"
pinecone_env = "us-east-1"

if not openai_api_key or not pinecone_api_key:
    st.error("❌ Missing API keys! Set them in Streamlit Secrets.")
    st.stop()

# ✅ Initialize Pinecone
pc = pinecone.Pinecone(api_key=pinecone_api_key)

# ✅ Connect to Existing Pinecone Index
if pinecone_index_name not in [idx.name for idx in pc.list_indexes()]:
    st.error(f"❌ Pinecone index '{pinecone_index_name}' not found! Create it first.")
    st.stop()

index = pc.Index(pinecone_index_name)

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# ✅ Streamlit UI
st.title("🚦 Traffic Review AI Assistant")
st.write("Upload past studies to train AI or upload a new study for automated review.")

# **🔹 Section 1: AI Learning Area**
st.header("📚 AI Learning Area (Upload Past Studies)")
st.write("Train AI by uploading past consultant reports, review comments, and response letters.")

past_project_name = st.text_input("Enter past project name for AI training:")
if past_project_name:
    project_folder = os.path.join("rag_learning", past_project_name)
    os.makedirs(project_folder, exist_ok=True)

    # ✅ Upload Files
    files = {
        "Raw Study (Consultant Submission)": os.path.join(project_folder, "raw_study.pdf"),
        "Annotated Study (City Edits)": os.path.join(project_folder, "annotated_study.pdf"),
        "Traffic Review Response Letter": os.path.join(project_folder, "review_letter.pdf")
    }

    uploaded_files = {}

    for category, file_path in files.items():
        uploaded_file = st.file_uploader(f"Upload '{category}'", type=["pdf"])
        if uploaded_file:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_files[category] = file_path
            st.success(f"✅ Uploaded '{category}' for project '{past_project_name}'")

    # ✅ Process & Train AI
    if len(uploaded_files) == 3:
        all_docs = []
        for file_path in uploaded_files.values():
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        # ✅ Store in Pinecone
        vectorstore = Pinecone(index, embeddings, "text")
        vectorstore.add_documents(all_docs)
        st.success(f"✅ Project '{past_project_name}' indexed in Pinecone!")

# **🔹 Section 2: New Study Review**
st.header("📝 New Study Review")
st.write("Upload a new study and let AI generate review comments.")

new_project_name = st.text_input("Enter new project name:")
if new_project_name:
    new_project_folder = os.path.join("new_studies", new_project_name)
    os.makedirs(new_project_folder, exist_ok=True)

    raw_study_file = st.file_uploader("Upload 'Raw Study (Consultant Submission)'", type=["pdf"])
    if raw_study_file:
        raw_study_path = os.path.join(new_project_folder, "raw_study.pdf")
        with open(raw_study_path, "wb") as f:
            f.write(raw_study_file.getbuffer())
        st.success(f"✅ Uploaded raw study for project '{new_project_name}'")
