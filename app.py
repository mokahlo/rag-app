import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
import openai

# ✅ Secure OpenAI API Key Handling
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
if not openai_api_key:
    st.error("❌ OpenAI API key is missing! Set it in Streamlit Secrets.")
    st.stop()

# ✅ Define Storage Directories
LEARNING_DIR = "rag_learning"
NEW_STUDY_DIR = "new_studies"

# Ensure Directories Exist
os.makedirs(LEARNING_DIR, exist_ok=True)
os.makedirs(NEW_STUDY_DIR, exist_ok=True)

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# ✅ Streamlit UI
st.title("🚦 Traffic Review AI Assistant")
st.write("Upload past studies to train AI or upload a new study for automated review.")

# **🔹 Section 1: RAG Learning Area**
st.header("📚 AI Learning Area (Upload Past Studies)")
st.write("Train AI by uploading past consultant reports, review comments, and response letters.")

# 📂 Enter Past Project Name
past_project_name = st.text_input("Enter past project name for AI training:")
if past_project_name:
    project_folder = os.path.join(LEARNING_DIR, past_project_name)
    os.makedirs(project_folder, exist_ok=True)

    # Define File Paths
    past_files = {
        "Raw Study (Consultant Submission)": os.path.join(project_folder, "raw_study.pdf"),
        "Annotated Study (City Edits)": os.path.join(project_folder, "annotated_study.pdf"),
        "Traffic Review Response Letter": os.path.join(project_folder, "review_letter.pdf")
    }

    uploaded_past_files = {}

    # ✅ Upload Files
    for category, file_path in past_files.items():
        uploaded_file = st.file_uploader(f"Upload '{category}'", type=["pdf"])
        if uploaded_file:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_past_files[category] = file_path
            st.success(f"✅ Uploaded '{category}' for project '{past_project_name}'")

    # ✅ Process & Train AI
    if len(uploaded_past_files) == 3:
        all_docs = []
        for file_path in uploaded_past_files.values():
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        if all_docs:
            # ✅ Initialize ChromaDB Correctly
            vectorstore = Chroma(
                collection_name="traffic_reviews",
                embedding_function=embeddings,
                persist_directory=project_folder
            )
            vectorstore.add_documents(all_docs)
            st.success(f"✅ Project '{past_project_name}' has been indexed for AI learning!")

# **🔹 Section 2: New Study Review**
st.header("📝 New Study Review")
st.write("Upload a new raw consultant study and let AI generate review comments and a traffic response letter.")

# 📂 Enter New Project Name
new_project_name = st.text_input("Enter new project name:")
if new_project_name:
    new_project_folder = os.path.join(NEW_STUDY_DIR, new_project_name)
    os.makedirs(new_project_folder, exist_ok=True)

    # ✅ Upload Raw Study
    raw_study_file = st.file_uploader("Upload 'Raw Study (Consultant Submission)'", type=["pdf"])
    if raw_study_file:
        raw_study_path = os.path.join(new_project_folder, "raw_study.pdf")
        with open(raw_study_path, "wb") as f:
            f.write(raw_study_file.getbuffer())
        st.success(f"✅ Uploaded raw study for project '{new_project_name}'")

# **🔹 AI-Generated Comments & Review Letter**
st.subheader("🚀 Generate AI Review")
available_projects = os.listdir(NEW_STUDY_DIR) if os.listdir(NEW_STUDY_DIR) else ["No projects available"]
selected_project = st.selectbox("Select a project for AI review:", available_projects)

if selected_project and selected_project != "No projects available":
    project_folder = os.path.join(NEW_STUDY_DIR, selected_project)

    # ✅ Load AI Knowledge Base
    vectorstore = Chroma(persist_directory=LEARNING_DIR, embedding_function=embeddings)

    st.subheader("1️⃣ AI-Generated Comments on Consultant’s Study")
    if st.button("Generate AI Comments"):
        comments_prompt = """
        You are a City Traffic Engineer reviewing a private consultant’s traffic impact analysis.
        Based on past projects, generate professional review comments identifying necessary improvements.
        """
        docs = vectorstore.similarity_search(comments_prompt)
        if docs:
            ai_comments = OpenAI(api_key=openai_api_key).complete(comments_prompt + "\n\n" + docs[0].page_content)
            st.write("### ✍️ AI-Generated Comments:")
            st.write(ai_comments)
        else:
            st.write("❌ No relevant information found.")

    st.subheader("2️⃣ AI-Generated Traffic Review Letter")
    if st.button("Generate Traffic Review Letter"):
        review_letter_prompt = """
        You are a City Traffic Engineer drafting a formal traffic review response letter.
        Based on past projects, generate a structured and professional letter addressing key findings, improvements, and City policy compliance.
        """
        docs = vectorstore.similarity_search(review_letter_prompt)
        if docs:
            ai_review_letter = OpenAI(api_key=openai_api_key).complete(review_letter_prompt + "\n\n" + docs[0].page_content)
            st.write("### 📄 AI-Generated Traffic Review Letter:")
            st.write(ai_review_letter)
        else:
            st.write("❌ No relevant information found.")
