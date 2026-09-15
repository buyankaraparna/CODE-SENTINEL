import streamlit as st
from reviewer import review_code

st.set_page_config(page_title="AI Code Review Agent", layout="wide")

st.title("🔍 AI Code Review & Debugging Agent")
st.caption("Runs entirely locally using Ollama — no cloud, no API keys.")

uploaded_file = st.file_uploader("Upload a Python file (.py)", type=["py"])

pasted_code = st.text_area("...or paste code directly here", height=250)

if st.button("Review Code"):
    if uploaded_file is not None:
        code = uploaded_file.read().decode("utf-8", errors="ignore")
    elif pasted_code.strip():
        code = pasted_code
    else:
        st.warning("Please upload a file or paste some code first.")
        code = None

    if code:
        with st.spinner("Reviewing code with local LLM... this may take 10-30 seconds"):
            result = review_code(code)
        st.subheader("Review Result")
        st.text(result)