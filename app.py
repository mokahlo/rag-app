import streamlit as st
import os
import openai
import anthropic
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from PyPDF2 import PdfReader

# 🔹 Load API Keys from Streamlit Secrets
api_keys = {
    "OPENAI": st.secrets.get("OPENAI_API_KEY"),
    "OPENROUTER": st.secrets.get("OPENROUTER_API_KEY"),
    "CLAUDE": st.secrets.get("CLAUDE_API_KEY"),
    "PINECONE": st.secrets.get("PINECONE_API_KEY"),
}
current_provider_index = 0  # Tracks which API is in use

# 🔹 Pinecone Setup
INDEX_NAME = "ample-traffic"
pc = Pinecone(api_key=api_keys["PINECONE"])

# Check if the index exists, if not, create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(INDEX_NAME)  # Correct way to access the index
embeddings = OpenAIEmbeddings(api_key=api_keys["OPENAI"])

# 🔹 Streamlit UI
st.set_page_config(page_title="Traffic Review AI", layout="wide")
st.title("🚦 Traffic Review AI Assistant")
st.write("Upload traffic studies & let AI generate review comments.")

# 🔹 Model Selection in Sidebar
st.sidebar.header("🔍 Select AI Models")
use_openai = st.sidebar.checkbox("OpenAI (GPT-4)", True)
use_openrouter = st.sidebar.checkbox("OpenRouter", True)
use_claude = st.sidebar.checkbox("Claude 3.5 Haiku", True)

# 🔹 File Upload Section
st.header("📂 Upload Traffic Studies")
col1, col2, col3 = st.columns(3)

with col1:
    raw_study = st.file_uploader("📄 Raw Study", type=["pdf"])
with col2:
    annotated_study = st.file_uploader("📝 Study with Comments", type=["pdf"])
with col3:
    review_letter = st.file_uploader("📃 Traffic Review Letter", type=["pdf"])

# 🔹 Function to Extract Text from PDFs
def extract_text_from_pdf(uploaded_file):
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return ""

raw_text = extract_text_from_pdf(raw_study)
annotated_text = extract_text_from_pdf(annotated_study)
review_text = extract_text_from_pdf(review_letter)

# 🔹 AI Query Function
def get_ai_response(prompt):
    global current_provider_index
    providers = [("OPENAI", "gpt-4"), ("OPENROUTER", "gpt-4"), ("CLAUDE", "claude-3.5-haiku")]

    # Filter enabled providers
    enabled_providers = [
        (key, model) for key, model in providers if api_keys[key] and st.session_state.get(key, True)
    ]

    for _ in range(len(enabled_providers)):
        key, model = enabled_providers[current_provider_index]
        try:
            if key == "CLAUDE":
                client = anthropic.Anthropic(api_key=api_keys[key])
                response = client.messages.create(
                    model=model, max_tokens=500, messages=[{"role": "user", "content": prompt}]
                )
                return response["content"]
            else:
                response = openai.ChatCompletion.create(
                    model=model, messages=[{"role": "system", "content": prompt}],
                    api_key=api_keys[key]
                )
                return response["choices"][0]["message"]["content"]
        except Exception as e:
            st.warning(f"🚨 {key} API failed. Trying next provider...")
            current_provider_index = (current_provider_index + 1) % len(enabled_providers)

    return "❌ No AI models available or quota exceeded."

# 🔹 AI Insights Before Storing in Pinecone
st.header("🔍 AI Insights Before Storing")
ai_insights = {}

if st.button("📊 Generate AI Insights for Each Document"):
    if raw_text:
        ai_insights["Raw Study"] = get_ai_response(f"Summarize this traffic study:\n\n{raw_text}")
    if annotated_text:
        ai_insights["Annotated Study"] = get_ai_response(f"Summarize the annotations and comments:\n\n{annotated_text}")
    if review_text:
        ai_insights["Traffic Review Letter"] = get_ai_response(f"Summarize this traffic review letter:\n\n{review_text}")

    # Display AI-generated insights
    for doc_name, insight in ai_insights.items():
        st.subheader(f"📄 AI Insights: {doc_name}")
        st.write(insight)

# 🔹 Store Document Embeddings in Pinecone
if st.button("📌 Store Study in Pinecone"):
    if raw_text and annotated_text and review_text:
        all_texts = [raw_text, annotated_text, review_text]
        vectorstore = LangchainPinecone.from_texts(
            texts=all_texts, embedding=embeddings, index_name=INDEX_NAME
        )
        st.success("✅ Traffic study stored in Pinecone!")
    else:
        st.warning("⚠️ Please upload all three files before proceeding.")

# 🔹 AI Review Generation
st.header("📝 Generate AI Review Comments")
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Generate AI Comments"):
        if raw_text:
            prompt = f"Review the following traffic study and provide detailed comments:\n\n{raw_text}"
            ai_response = get_ai_response(prompt)
            st.write("### 📝 AI Review Comments:")
            st.write(ai_response)
        else:
            st.warning("⚠️ Please upload a raw study first.")

with col2:
    if st.button("📄 Generate AI Review Letter"):
        if review_text:
            prompt = f"Draft a formal traffic review letter based on these comments:\n\n{review_text}"
            ai_response = get_ai_response(prompt)
            st.write("### 📄 AI Review Letter:")
            st.write(ai_response)
        else:
            st.warning("⚠️ Please upload a review letter first.")
