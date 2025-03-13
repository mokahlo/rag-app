import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import openai

# ✅ Load API Keys from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_index_name = st.secrets["PINECONE_INDEX"]
pinecone_region = st.secrets["PINECONE_ENV"]

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
st.write("Train AI by uploading the raw study, annotated study with comments, and the final traffic review letter.")

# 📂 Upload 3 files
raw_study = st.file_uploader("Upload Raw Study (Consultant Submission)", type=["pdf"])
annotated_study = st.file_uploader("Upload Annotated Study (City Comments)", type=["pdf"])
traffic_review_letter = st.file_uploader("Upload Traffic Review Letter", type=["pdf"])

if raw_study and annotated_study and traffic_review_letter:
    with st.spinner("Processing documents..."):
        files = {
            "Raw Study": raw_study,
            "Annotated Study": annotated_study,
            "Traffic Review Letter": traffic_review_letter
        }

        all_docs = []
        for name, uploaded_file in files.items():
            file_path = f"/tmp/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ✅ Process PDF and Load Documents
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        # ✅ Store document embeddings in Pinecone
        vectorstore = LangchainPinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)

        st.success("✅ All documents successfully indexed in Pinecone!")

# 🔹 AI-Generated Review
st.header("📝 New Study Review")
st.write("Upload a new raw study and let AI generate review comments and a response letter.")

new_study = st.file_uploader("Upload New Study (Consultant Submission)", type=["pdf"])

if new_study and st.button("Generate AI Review"):
    with st.spinner("Analyzing the study..."):
        file_path = f"/tmp/{new_study.name}"
        with open(file_path, "wb") as f:
            f.write(new_study.getbuffer())

        # ✅ Process and search relevant past studies
        query = "Generate traffic review comments for a consultant study."
        results = vectorstore.similarity_search(query, k=3)

        if results:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a city traffic engineer reviewing a study."},
                    {"role": "user", "content": f"Summarize these studies: {results}"},
                ],
            )
            st.subheader("🚀 AI-Generated Comments")
            st.write(response["choices"][0]["message"]["content"])

            review_letter_prompt = "Write a professional traffic review letter based on these studies."
            letter_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a city traffic engineer drafting a traffic review letter."},
                    {"role": "user", "content": f"Use these studies to draft a review letter: {results}"},
                ],
            )
            st.subheader("📄 AI-Generated Traffic Review Letter")
            st.write(letter_response["choices"][0]["message"]["content"])
        else:
            st.write("❌ No relevant past studies found in the database.")
