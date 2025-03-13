import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import openai
import os

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("Traffic Engineering RAG System")
st.write("Upload a Traffic Impact Analysis (TIA) report and ask questions.")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save file
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process the document
    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    # Create vector store
    vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())

    # Query interface
    query = st.text_input("Ask a question about the report:")
    if st.button("Get Answer"):
        docs = vectorstore.similarity_search(query)
        st.write(docs[0].page_content)
