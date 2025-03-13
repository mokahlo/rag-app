import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeStore  # ✅ Fixed Import
from langchain_community.embeddings import OpenAIEmbeddings  # ✅ Fixed Import
import openai

# ✅ Secure API Key Handling
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", None)
pinecone_region = "us-east-1"  # ✅ Your region
index_name = "ample-parking"  # ✅ Your Pinecone index name

if not openai_api_key or not pinecone_api_key:
    st.error("❌ Missing API keys! Set them in Streamlit Secrets.")
    st.stop()  # Stop execution if API keys are missing

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# ✅ Correct Pinecone Initialization
pc = Pinecone(api_key=pinecone_api_key)

# ✅ Ensure Pinecone Index Exists
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=pinecone_region
        )
    )

# ✅ Connect to Pinecone Index
index = pc.Index(index_name)

# ✅ Create LangChain Pinecone Wrapper Properly
vectorstore = PineconeStore(
    index=index,  # ✅ Correct way to pass Pinecone Index
    embedding_function=embeddings.embed_query,  # ✅ Pass embeddings function
)

# ✅ Streamlit UI
st.title("🚦 Traffic Review AI Assistant with Pinecone (`ample-parking`)")
st.write("Upload past studies to train AI or upload a new study for automated review.")

# **🔹 Upload Documents & Add to Pinecone**
st.header("📚 AI Learning Area (Upload Past Studies)")
uploaded_file = st.file_uploader("Upload a past study (TXT or PDF)", type=["txt", "pdf"])

if uploaded_file:
    text_content = uploaded_file.read().decode("utf-8")  # Convert to text
    doc_embedding = embeddings.embed_query(text_content)

    # Store document in Pinecone (`ample-parking` index)
    vectorstore.add_texts([text_content], embeddings=[doc_embedding])

    st.success("✅ Document indexed in Pinecone!")

# **🔹 Query AI Knowledge Base**
st.header("🔍 Search AI Knowledge Base")
query = st.text_input("Enter search query:")

if query:
    docs = vectorstore.similarity_search(query, k=3)  # Retrieve top 3 similar docs
    st.subheader("🔎 AI-Generated Results")

    for i, doc in enumerate(docs):
        st.write(f"**Result {i+1}:** {doc.page_content}")
