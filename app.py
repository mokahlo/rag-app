import os
import json
import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# ✅ Load API keys and fallback models
api_keys = json.loads(st.secrets["OPENAI_API_KEYS"])  # Store multiple API keys
fallback_models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
current_key_index = 0

# ✅ Pinecone Configuration
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ample-traffic"

# ✅ Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(INDEX_NAME)

# ✅ Function to Get Embeddings with API Key Rotation
def get_embedding_function():
    global current_key_index
    for _ in range(len(api_keys)):  # Try all available keys
        try:
            return OpenAIEmbeddings(openai_api_key=api_keys[current_key_index])
        except openai.error.RateLimitError:
            st.warning(f"🚨 API key {current_key_index+1} exceeded quota. Switching to next key...")
            current_key_index = (current_key_index + 1) % len(api_keys)
    st.error("🚨 All API keys have exceeded their quota!")
    return None

embeddings = get_embedding_function()

# ✅ Function to Get AI Response with Model Fallback
def get_ai_response(prompt):
    for model in fallback_models:  # Step down through models if one fails
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                api_key=api_keys[current_key_index]
            )
            return response["choices"][0]["message"]["content"]
        except openai.error.RateLimitError:
            st.warning(f"🚨 Model {model} exceeded quota. Trying next model...")
    st.error("🚨 All models are at capacity. Try again later.")
    return None

# ✅ Streamlit UI
st.title("🚦 Traffic Review AI Assistant")
st.write("Upload past studies and generate AI-powered reviews.")

# 🔹 Section 1: Upload Past Studies
st.header("📚 AI Learning Area (Upload Past Studies)")
st.write("Train AI with past consultant reports, annotated comments, and response letters.")

past_project_name = st.text_input("Enter past project name for AI training:")
if past_project_name:
    project_folder = os.path.join("rag_learning", past_project_name)
    os.makedirs(project_folder, exist_ok=True)

    uploaded_files = {
        "Raw Study (Consultant Submission)": st.file_uploader("Upload 'Raw Study'", type=["pdf"]),
        "Annotated Study (City Edits)": st.file_uploader("Upload 'Annotated Study'", type=["pdf"]),
        "Traffic Review Letter": st.file_uploader("Upload 'Traffic Review Letter'", type=["pdf"])
    }

    # ✅ Process & Store Files
    if all(uploaded_files.values()):
        all_docs = []
        for label, file in uploaded_files.items():
            file_path = os.path.join(project_folder, f"{label.replace(' ', '_').lower()}.pdf")
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        # ✅ Store document embeddings in Pinecone
        vectorstore = LangchainPinecone.from_documents(all_docs, embeddings, index_name=INDEX_NAME)
        st.success("✅ All documents successfully indexed in Pinecone!")

# 🔹 Section 2: New Study Review
st.header("📝 New Study Review")
st.write("Upload a new raw study and let AI generate review comments and a traffic response letter.")

new_study = st.file_uploader("Upload New Study (Consultant Submission)", type=["pdf"])
if new_study:
    new_study_path = os.path.join("new_studies", "raw_study.pdf")
    with open(new_study_path, "wb") as f:
        f.write(new_study.getbuffer())

    st.success("✅ Uploaded new study for AI review.")

# 🔹 AI-Generated Comments & Review Letter
st.subheader("🚀 Generate AI Review")
if st.button("Generate AI Comments"):
    query = "Generate professional review comments for this traffic study."
    vectorstore = LangchainPinecone(index, embeddings, text_key="text")
    docs = vectorstore.similarity_search(query)

    if docs:
        ai_response = get_ai_response(query + "\n\n" + docs[0].page_content)
        if ai_response:
            st.write("### ✍️ AI-Generated Comments:")
            st.write(ai_response)
    else:
        st.write("❌ No relevant information found.")

if st.button("Generate Traffic Review Letter"):
    query = "Generate a structured traffic review letter based on this study."
    docs = vectorstore.similarity_search(query)

    if docs:
        ai_response = get_ai_response(query + "\n\n" + docs[0].page_content)
        if ai_response:
            st.write("### 📄 AI-Generated Traffic Review Letter:")
            st.write(ai_response)
    else:
        st.write("❌ No relevant information found.")
