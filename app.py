import openai
import streamlit as st
import json

# Load API keys from Streamlit Secrets
api_keys = json.loads(st.secrets["OPENAI_API_KEYS"])
current_key_index = 0  # Start with the first key

def call_openai_api(prompt):
    global current_key_index
    
    for _ in range(len(api_keys)):  # Try all available keys
        openai.api_key = api_keys[current_key_index]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}]
            )
            return response  # Success!

        except openai.error.RateLimitError:
            st.warning(f"🚨 API key {current_key_index+1} exceeded quota. Switching to next key...")
            current_key_index = (current_key_index + 1) % len(api_keys)  # Rotate to next key

        except openai.error.OpenAIError as e:
            st.error(f"❌ OpenAI Error: {e}")
            break  # Stop if it's another OpenAI error

    st.error("🚨 All API keys have exceeded their quota!")
    return None

# Example Streamlit UI
st.title("🔑 OpenAI API Backup Key Rotation")
prompt = st.text_input("Enter a prompt:")
if st.button("Generate Response"):
    result = call_openai_api(prompt)
    if result:
        st.write(result["choices"][0]["message"]["content"])
