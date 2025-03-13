import streamlit as st
import openai
import os
import time
import logging
import requests  # Required for Claude API Calls
from openai import OpenAIError

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load API keys from Streamlit Secrets
api_providers = [
    {"name": "OpenAI", "key": st.secrets.get("OPENAI_API_KEY"), "model": "gpt-4", "base_url": "https://api.openai.com/v1"},
    {"name": "OpenRouter", "key": st.secrets.get("OPENROUTER_API_KEY"), "model": "gpt-3.5-turbo", "base_url": "https://openrouter.ai/api/v1"},
    {"name": "Claude", "key": st.secrets.get("CLAUDE_API_KEY"), "model": "claude-3-opus", "base_url": "https://api.anthropic.com/v1"}
]

# ✅ Streamlit UI: Checkboxes to enable/disable providers
st.sidebar.title("⚙️ AI Model Selection")
use_openai = st.sidebar.checkbox("Use OpenAI", value=True)
use_openrouter = st.sidebar.checkbox("Use OpenRouter", value=True)
use_claude = st.sidebar.checkbox("Use Claude", value=True)

# Filter out disabled providers
enabled_providers = [
    provider for provider in api_providers if (
        (provider["name"] == "OpenAI" and use_openai) or
        (provider["name"] == "OpenRouter" and use_openrouter) or
        (provider["name"] == "Claude" and use_claude)
    )
]

if not enabled_providers:
    st.sidebar.warning("⚠️ At least one AI provider must be selected.")

current_provider_index = 0  # Start with the first enabled provider

# ✅ Function to get AI response with automatic fallback
def get_ai_response(prompt):
    """Tries different API providers if one exceeds quota or fails authentication."""
    global current_provider_index

    for _ in range(len(enabled_providers)):  # Try all enabled providers
        provider = enabled_providers[current_provider_index]

        if not provider["key"]:
            logger.warning(f"⚠️ API key for {provider['name']} is missing. Skipping...")
            current_provider_index = (current_provider_index + 1) % len(enabled_providers)
            continue

        try:
            st.info(f"🔄 Using {provider['name']} API...")

            # ✅ OpenAI & OpenRouter Use OpenAI-Compatible API
            if provider["name"] in ["OpenAI", "OpenRouter"]:
                client = openai.OpenAI(api_key=provider["key"], base_url=provider["base_url"])  # ✅ Fix API URL
                response = client.chat.completions.create(
                    model=provider["model"],
                    messages=[{"role": "system", "content": prompt}]
                )
                return response.choices[0].message.content

            # ✅ Claude API Needs Custom HTTP Request
            elif provider["name"] == "Claude":
                headers = {
                    "x-api-key": provider["key"],
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": provider["model"],
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                }
                response = requests.post(f"{provider['base_url']}/messages", headers=headers, json=payload)

                if response.status_code == 200:
                    return response.json()["content"]
                else:
                    raise ValueError(f"Claude API Error: {response.text}")

        except (OpenAIError, requests.exceptions.RequestException) as e:
            logger.warning(f"🚨 {provider['name']} API failed. Switching to next provider... Error: {e}")
            current_provider_index = (current_provider_index + 1) % len(enabled_providers)
            time.sleep(2)  # Wait before switching

    st.error("🚨 All selected API providers failed! Please update API keys.")
    return "Error: No available API providers."

# ✅ Streamlit UI
st.title("🚦 Traffic Review AI Assistant")

st.write("Upload traffic studies and generate AI-powered review comments.")

# File upload section for three documents
st.header("📂 Upload Traffic Study Documents")
raw_study = st.file_uploader("Upload Raw Study (Consultant Submission)", type=["pdf"])
annotated_study = st.file_uploader("Upload Study with City Comments", type=["pdf"])
review_letter = st.file_uploader("Upload Resulting Traffic Review Letter", type=["pdf"])

if st.button("Generate AI Review"):
    if raw_study and annotated_study and review_letter:
        st.success("✅ Files uploaded successfully! Generating AI-powered review...")

        # Generate AI response based on study contents
        prompt = "Analyze the uploaded studies and generate AI-based review comments."
        ai_response = get_ai_response(prompt)

        st.subheader("📄 AI-Generated Review:")
        st.write(ai_response)

    else:
        st.warning("⚠️ Please upload all three required documents.")
