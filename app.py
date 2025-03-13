import streamlit as st
import os
import requests
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone

# ✅ Load API Keys from Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
CLAUDE_API_KEY = st.secrets.get("CLAUDE_API_KEY")

# ✅ Pinecone Setup
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ample-traffic"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

# ✅ Available AI Providers
api_providers = {
    "OpenAI": {"key": OPENAI_API_KEY, "url": "https://api.openai.com/v1/chat/completions"},
    "OpenRouter": {"key": OPENROUTER_API_KEY, "url": "https://openrouter.ai/api/v1/chat/completions"},
    "Claude": {"key": CLAUDE_API_KEY, "url": "https://api.anthropic.com/v1/messages"}
}
selected_providers = {}

# ✅ UI Setup
st.title("🚦 Traffic Review AI Assistant")
st.write("Upload past studies to train AI or upload a new study for automated review.")

# 🔹 Model Selection (Checkboxes)
st.subheader("Select AI Models")
for provider in api_providers.keys():
    selected_providers[provider] = st.checkbox(provider, value=True)

# 🔹 PDF Text Extraction
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# 🔹 API Query Function (Fallback Enabled)
def query_ai(prompt, uploaded_file):
    extracted_text = extract_text_from_pdf(uploaded_file)
    for provider, details in api_providers.items():
        if selected_providers.get(provider):
            st.write(f"🤖 Using {provider} API...")
            headers = {
                "Authorization": f"Bearer {details['key']}",
                "Content-Type": "application/json"
            }
            if provider == "Claude":
                headers["anthropic-version"] = "2023-06-01"

            payload = {
                "model": "claude-3-5-haiku-20241022" if provider == "Claude" else "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a City Traffic Engineer reviewing a traffic study."},
                    {"role": "user", "content": f"{prompt}\n\n### Study Contents:\n{extracted_text}"}
                ],
                "max_tokens": 500
            }

            response = requests.post(details["url"], headers=headers, json=payload)
            if response.status_code == 200:
                return response.json().get("content", response.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
            else:
                st.warning(f"🚨 {provider} API Error: {response.text}")
    return "❌ All selected AI providers failed. Please check API keys and quotas."

# 🔹 Study Upload Section
st.header("📝 New Study Review")
st.write("Upload a raw study, and AI will generate review comments.")

uploaded_study = st.file_uploader("Upload New Study (PDF)", type=["pdf"])
if uploaded_study:
    study_text = extract_text_from_pdf(uploaded_study)
    st.text_area("Extracted Study Text", study_text, height=200)

    if st.button("Analyze with AI"):
        response = query_ai("Provide feedback on this traffic study.", uploaded_study)
        if response:
            st.write("### 🚦 AI Review Comments")
            st.write(response)
