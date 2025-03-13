import os
import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# ✅ Load API Keys from Streamlit Secrets
api_providers = [
    {"name": "OpenAI", "key": st.secrets["OPENAI_API_KEY"], "model": "gpt-4"},
    {"name": "OpenRouter", "key": st.secrets["OPENROUTER_API_KEY"], "model": "mistral-7b"},
    {"name": "Claude", "key": st.secrets["CLAUDE_API_KEY"], "model": "claude-3-haiku"},
]
current_provider_index = 0

# ✅ Pinecone Configuration
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ample-traffic"

# ✅ Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ✅ Function to Rotate API Providers
def get_embedding_function():
    global current_provider_index
    for _ in range(len(api_providers)):  # Try all providers
        try:
            return OpenAIEmbeddings(openai_api_key=api_providers[current_provider_index]["key"])
        except openai.error.RateLimitError:
            st.warning(f"🚨 {api_providers[current_provider_index]['name']} API hit quota. Switching...")
            current_provider_index = (current_provider_index + 1) % len(api_providers)
    st.error("🚨 All API providers are exhausted!")
    return None

embeddings = get_embedding_function()

# ✅ Function to Call AI with Automatic API Switching
def get_ai_response(prompt):
    global current_provider_index

    for _ in range(len(api_providers)):  # Try all API providers
        provider = api_providers[current_provider_index]
        try:
            response = openai.ChatCompletion.create(
                model=provider["model"],
                messages=[{"role": "system", "content": prompt}],
                api_key=provider["key"]
            )
            return response["choices"][0]["message"]["content"]
        except openai.error.RateLimitError:
            st.warning(f"🚨 {provider['name']} API exceeded quota. Switching...")
            current_provider_index = (current_provider_index + 1) % len(api_providers)

    st.error("🚨 All API providers have hit quota limits!")
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
        "Raw Study": st.file_uploader("Upload 'Raw Study'", type=["pdf"]),
        "Annotated Study": st.file_uploader("Upload 'Annotated Study'", type=["pdf"]),
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
