import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader  # ✅ Fixed Import
from langchain_community.vectorstores import Chroma  # ✅ Fixed Import
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import openai

# ✅ Secure OpenAI API Key Handling
openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
if not openai_api_key:
    st.error("❌ OpenAI API key is missing! Set it in Streamlit Secrets.")

# ✅ Define Storage Directories
LEARNING_DIR = "rag_learning"
NEW_STUDY_DIR = "new_studies"

# Ensure Directories Exist
os.make
