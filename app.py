import streamlit as st
import os
import requests
import openai
import anthropic
import pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from PyPDF2 import PdfReader

# 🔹 Load API Keys from Streamlit Secrets
api_keys = {
    "OPENAI": st.secrets.get("OPENAI_API_KEY"),
    "OPENROUTER": st.secrets.get("OPENROUTER_API_KEY"),
    "CLAUDE": st.secrets.get("CLAUDE_API_KEY"),
}
current_provider_index = 0  # Tracks which API is in use

# 🔹 Pinecone Setup
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ample-traffic"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
pinecone_index = pinecone.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(api_key=api_keys["OPENAI"])

# 🔹 Streamlit UI
st.title("🚦 Traffic Review AI Assistant")
st.write("Upload traffic studies & let AI generate review comments.")

# 🔹 Model Selection
st.sidebar.header("🔍 Select AI Models")
use_openai = st.sidebar.checkbox("OpenAI (GPT-4)", True)
use_openrouter = st.sidebar.checkbox("OpenRouter", True)
use_claude = st.sidebar.checkbox("Claude 3.5 Haiku", True)

# 🔹 File Upload
st.header("📂 Upload Traffic Studies")
raw_study = st.file_uploader("Upload Raw Study", type=["pdf"])
annotated_study = st.file_uploader("Upload Study with Comments", type=["pdf"])
review_letter = st.file_uploader("Upload Traffic Review Letter", type=["pdf"])

# 🔹 Process PDFs
def extract_text_from_pdf(uploaded_file):
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return ""

raw_text = extract_text_from_pdf(raw_study)
annotated_text = extract_text_from_pdf(annotated_study)
review_text = extract_text_from_pdf(review_letter)

# 🔹 Store Document Embeddings
if st.button("📌 Store Study in Pinecone"):
    if raw_text and annotated_text and review_text:
        all_texts = [raw_text, annotated_text, review_text]
        vectorstore = LangchainPinecone.from_texts(
            texts=all_texts, embedding=embeddings, index_name=INDEX_NAME
        )
        st.success("✅ Traffic study stored in Pinecone!")
    else:
        st.warning("⚠️ Please upload all three files before proceeding.")

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

# 🔹 Generate AI Review
st.header("📝 Generate AI Review Comments")
if st.button("🚀 Generate AI Comments"):
    if raw_text:
        prompt = f"Review the following traffic study and provide detailed comments:\n\n{raw_text}"
        ai_response = get_ai_response(prompt)
        st.write("### 📝 AI Review Comments:")
        st.write(ai_response)
    else:
        st.warning("⚠️ Please upload a raw study first.")

if st.button("📄 Generate AI Review Letter"):
    if review_text:
        prompt = f"Draft a formal traffic review letter based on these comments:\n\n{review_text}"
        ai_response = get_ai_response(prompt)
        st.write("### 📄 AI Review Letter:")
        st.write(ai_response)
    else:
        st.warning("⚠️ Please upload a review letter first.")
