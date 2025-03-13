import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile

# ✅ Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF uploaded via Streamlit."""
    if uploaded_file is not None:
        # ✅ Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name  # ✅ Get the correct file path
        
        try:
            doc = fitz.open(temp_file_path)  # ✅ Open the file correctly
            text = ""
            for page in doc:
                text += page.get_text("text")  # ✅ Extract text
            doc.close()
            
            os.remove(temp_file_path)  # ✅ Clean up temp file after extraction
            return text
        except Exception as e:
            st.error(f"⚠️ Error processing PDF: {str(e)}")
            return None
    return None

# ✅ Streamlit Web UI
st.title("📄 Traffic Review AI Database")
st.write("Upload traffic study documents to store them in the database.")

# 📂 Upload raw, annotated study, and final review letter
raw_study = st.file_uploader("Upload Raw Traffic Study (PDF)", type=["pdf"])
annotated_study = st.file_uploader("Upload Annotated Study (PDF)", type=["pdf"])
review_letter = st.file_uploader("Upload Traffic Review Letter (PDF)", type=["pdf"])

# ✅ Process and display extracted text
if raw_study:
    st.subheader("📜 Extracted Text from Raw Study")
    raw_text = extract_text_from_pdf(raw_study)
    st.text_area("Raw Study Text", raw_text, height=200)

if annotated_study:
    st.subheader("📜 Extracted Text from Annotated Study")
    annotated_text = extract_text_from_pdf(annotated_study)
    st.text_area("Annotated Study Text", annotated_text, height=200)

if review_letter:
    st.subheader("📜 Extracted Text from Traffic Review Letter")
    review_text = extract_text_from_pdf(review_letter)
    st.text_area("Review Letter Text", review_text, height=200)

