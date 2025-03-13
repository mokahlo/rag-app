import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

# ✅ Secure API Key Handling
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", None)
pinecone_env = st.secrets.get("PINECONE_ENV", "us-west1-gcp")  # Default to US region

if not openai_api_key or not pinecone_api_key:
    st.error("❌ Missing API keys! Set OPENAI and PINECONE API keys in Streamlit Secrets.")

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# ✅ Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Define Pinecone index name
index_name = "traffic-reviews"

# ✅ Ensure Pinecone Index Exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # Adjust dimension based on OpenAI embeddings

# ✅ Connect to Pinecone Index
vectorstore = Pinecone(index_name=index_name, embedding_function=embeddings)

# ✅ Streamlit UI
st.title("🚦 Traffic Review AI Assistant with Pinecone")
st.write("Upload past studies to train AI or upload a new study for automated review.")

# **🔹 Upload Documents & Add to Pinecone**
st.header("📚 AI Learning Area (Upload Past Studies)")
uploaded_file = st.file_uploader("Upload a past study (PDF)", type=["pdf"])

if uploaded_file:
    text_content = uploaded_file.read().decode("utf-8")  # Convert PDF content to text
    doc_embedding = embeddings.embed_query(text_content)

    # Store document in Pinecone
    vectorstore.add_texts([text_content])

    st.success("✅ Document indexed in Pinecone!")

# **🔹 Query AI Knowledge Base**
st.header("🔍 Search AI Knowledge Base")
query = st.text_input("Enter search query:")

if query:
    docs = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 similar docs
    st.subheader("🔎 AI-Generated Results")
    for i, doc in enumerate(docs):
        st.write(f"**Result {i+1}:** {doc.page_content}")

