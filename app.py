import streamlit as st
import os
import shutil
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import openai

# Set OpenAI API Key (Replace with your actual key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define main storage directory
PROJECTS_DIR = "projects"

# Ensure directory exists
os.makedirs(PROJECTS_DIR, exist_ok=True)

# Streamlit UI
st.title("Traffic Review AI Assistant")
st.write("Upload a raw consultant report, and the AI will generate review comments and a traffic review letter based on past projects.")

# **Step 1: Enter Project Name**
project_name = st.text_input("Enter project name:")
if project_name:
    project_folder = os.path.join(PROJECTS_DIR, project_name)
    os.makedirs(project_folder, exist_ok=True)

    # Define subfolders for each document type
    file_paths = {
        "Raw Consultant Study": os.path.join(project_folder, "raw_study.pdf"),
        "Annotated Study (City Edits)": os.path.join(project_folder, "annotated_study.pdf"),
        "Traffic Review Response Letter": os.path.join(project_folder, "review_letter.pdf")
    }

    uploaded_files = {}

    # **Step 2: Upload Files**
    for category, file_path in file_paths.items():
        uploaded_file = st.file_uploader(f"Upload '{category}'", type=["pdf"])
        if uploaded_file:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_files[category] = file_path
            st.success(f"Uploaded '{category}' for project '{project_name}'")

    # **Step 3: Process Files and Train AI**
    if len(uploaded_files) == 3:  # Ensure all three files are uploaded
        all_docs = []
        
        for category, file_path in uploaded_files.items():
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        if all_docs:
            # Create a vector store and save it for future queries
            vectorstore = Chroma.from_documents(all_docs, OpenAIEmbeddings(), persist_directory=project_folder)
            st.success(f"Project '{project_name}' has been indexed for AI review!")

# **Step 4: Generate AI Review Comments & Letter**
st.subheader("Generate AI-Based Traffic Review")

# Select a project to generate a review
available_projects = [p for p in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, p))]
selected_project = st.selectbox("Select a project for AI review:", available_projects if available_projects else ["No projects available"])

if selected_project and selected_project != "No projects available":
    project_folder = os.path.join(PROJECTS_DIR, selected_project)
    
    # Load the stored vector database
    vectorstore = Chroma(persist_directory=project_folder, embedding_function=OpenAIEmbeddings())
    
    st.subheader("1️⃣ AI-Generated Comments on Consultant’s Study")
    if st.button("Generate AI Comments"):
        comments_prompt = """
        You are a City Traffic Engineer reviewing a private consultant’s traffic impact analysis.
        Based on past projects, generate review comments for this study, identifying necessary improvements.
        Keep the feedback professional and aligned with City of Phoenix policies.
        """
        docs = vectorstore.similarity_search(comments_prompt)
        if docs:
            ai_comments = OpenAI().complete(comments_prompt + "\n\n" + docs[0].page_content)
            st.write("### AI-Generated Comments:")
            st.write(ai_comments)
        else:
            st.write("No relevant information found.")

    st.subheader("2️⃣ AI-Generated Traffic Review Letter")
    if st.button("Generate Traffic Review Letter"):
        review_letter_prompt = """
        You are a City Traffic Engineer drafting a formal traffic review response letter.
        Based on past projects, generate a structured and professional letter addressing key findings, improvements, and City policy compliance.
        """
        docs = vectorstore.similarity_search(review_letter_prompt)
        if docs:
            ai_review_letter = OpenAI().complete(review_letter_prompt + "\n\n" + docs[0].page_content)
            st.write("### AI-Generated Traffic Review Letter:")
            st.write(ai_review_letter)
        else:
            st.write("No relevant information found.")
