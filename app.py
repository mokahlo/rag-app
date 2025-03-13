import streamlit as st
import os
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import openai

# Set OpenAI API Key (Ensure this is set in Streamlit Cloud secrets)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define directories for past and new projects
LEARNING_DIR = "rag_learning"
NEW_STUDY_DIR = "new_studies"

# Ensure directories exist
os.makedirs(LEARNING_DIR, exist_ok=True)
os.makedirs(NEW_STUDY_DIR, exist_ok=True)

# Streamlit UI
st.title("Traffic Review AI Assistant")
st.write("Upload past traffic studies to train AI or upload a new study for automatic review.")

# **Section 1: RAG Learning Area**
st.header("📚 AI Learning Area (Upload Past Studies)")
st.write("Upload past consultant studies, annotated reviews, and response letters to improve AI learning.")

# Enter project name for training data
past_project_name = st.text_input("Enter past project name for AI training:")
if past_project_name:
    project_folder = os.path.join(LEARNING_DIR, past_project_name)
    os.makedirs(project_folder, exist_ok=True)

    # Define subfolders for past documents
    past_files = {
        "Raw Study (Consultant Submission)": os.path.join(project_folder, "raw_study.pdf"),
        "Annotated Study (City Edits)": os.path.join(project_folder, "annotated_study.pdf"),
        "Traffic Review Response Letter": os.path.join(project_folder, "review_letter.pdf")
    }

    uploaded_past_files = {}

    # Upload files
    for category, file_path in past_files.items():
        uploaded_file = st.file_uploader(f"Upload '{category}'", type=["pdf"])
        if uploaded_file:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_past_files[category] = file_path
            st.success(f"Uploaded '{category}' for past project '{past_project_name}'")

    # Process and train AI on past data
    if len(uploaded_past_files) == 3:
        all_docs = []
        for file_path in uploaded_past_files.values():
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        if all_docs:
            # Corrected OpenAIEmbeddings initialization
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
            vectorstore = Chroma.from_documents(all_docs, embeddings, persist_directory=project_folder)
            st.success(f"Past project '{past_project_name}' has been indexed for AI learning!")

# **Section 2: New Study Review Area**
st.header("📝 New Study Review")
st.write("Upload a new raw consultant study and let AI generate comments and a traffic review letter.")

# Enter new project name
new_project_name = st.text_input("Enter new project name:")
if new_project_name:
    new_project_folder = os.path.join(NEW_STUDY_DIR, new_project_name)
    os.makedirs(new_project_folder, exist_ok=True)

    # Upload raw study
    raw_study_file = st.file_uploader("Upload 'Raw Study (Consultant Submission)'", type=["pdf"])
    if raw_study_file:
        raw_study_path = os.path.join(new_project_folder, "raw_study.pdf")
        with open(raw_study_path, "wb") as f:
            f.write(raw_study_file.getbuffer())
        st.success(f"Uploaded raw study for project '{new_project_name}'")

# **AI-Generated Comments & Review Letter**
st.subheader("🚀 Generate AI Review")
selected_project = st.selectbox("Select a project for AI review:", os.listdir(NEW_STUDY_DIR) if os.listdir(NEW_STUDY_DIR) else ["No projects available"])

if selected_project and selected_project != "No projects available":
    project_folder = os.path.join(NEW_STUDY_DIR, selected_project)

    # Load stored knowledge base
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = Chroma(persist_directory=LEARNING_DIR, embedding_function=embeddings)

    st.subheader("1️⃣ AI-Generated Comments on Consultant’s Study")
    if st.button("Generate AI Comments"):
        comments_prompt = """
        You are a City Traffic Engineer reviewing a private consultant’s traffic impact analysis.
        Based on past projects, generate professional review comments identifying necessary improvements.
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
