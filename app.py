import streamlit as st
import os
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import openai

# Set OpenAI API Key (replace with your actual key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define file storage paths for each category
UPLOAD_DIRS = {
    "Raw Study (Consultant Submission)": "uploads/raw_study",
    "Annotated Study (City Edits)": "uploads/annotated_study",
    "Traffic Review Response Letter": "uploads/response_letter"
}

# Create directories if they don’t exist
for dir_path in UPLOAD_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Streamlit UI
st.title("City of Phoenix Traffic Study RAG System")
st.write("Upload and query traffic study documents.")

# Select project file bin
selected_bin = st.selectbox("Select a document category:", list(UPLOAD_DIRS.keys()))

# Upload PDF
uploaded_file = st.file_uploader(f"Upload a file to '{selected_bin}'", type=["pdf"])

if uploaded_file is not None:
    # Save file to selected bin
    file_path = os.path.join(UPLOAD_DIRS[selected_bin], uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded to '{selected_bin}' successfully!")

# Query Section
st.subheader("Ask a question based on the selected document bin")

query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if selected_bin in UPLOAD_DIRS:
        folder_path = UPLOAD_DIRS[selected_bin]

        # Load and process the documents in the selected bin
        all_docs = []
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, file))
                all_docs.extend(loader.load())

        if all_docs:
            # Create vector store
            vectorstore = Chroma.from
