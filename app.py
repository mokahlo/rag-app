import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import openai

# ✅ Load API Keys from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_index_name = st.secrets["PINECONE_INDEX"]
pinecone_region = st.secrets["PINECONE_ENV"]
pinecone_host = st.secrets["PINECONE_HOST"]

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# ✅ Initialize Pinecone Client
pc = Pinecone(api_key=pinecone_api_key)

# ✅ Ensure Pinecone Index Exists
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=pinecone_region
        )
    )

# ✅ Connect to Pinecone Index
index = pc.Index(pinecone_index_name)

# ✅ Streamlit App UI
st.title("🚦 Traffic Review AI Assistant")
st.write("Upload past studies to train AI or upload a new study for automated review.")

# 🔹 Upload & Process Files for AI Learning
st.header("📚 AI Learning Area (Upload Past Studies)")
uploaded_file = st.file_uploader("Upload PDF Study", type=["pdf"])

if uploaded_file:
    file_path = f"/tmp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ Process PDF and Embed in Pinecone
    with st.spinner("Processing document..."):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # ✅ Store document embeddings in Pinecone
        Pinecone.from_documents(docs, embeddings, index)

    st.success("✅ Document successfully indexed in Pinecone!")

# 🔹 AI-Generated Review
st.header("📝 New Study Review")
st.write("Upload a new study and let AI generate review comments.")

if st.button("Generate AI Review"):
    query = "Generate traffic review comments for a consultant study."
    results = index.query(query, top_k=3, include_metadata=True)

    if results["matches"]:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a city traffic engineer reviewing a study."},
                {"role": "user", "content": f"Summarize these studies: {results['matches']}"},
            ],
        )
        st.write(response["choices"][0]["message"]["content"])
    else:
        st.write("❌ No relevant studies found in the database.")
