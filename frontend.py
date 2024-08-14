import streamlit as st
import requests
import time

st.title("QnA on PDF Document📝")

# File uploader for PDF document
uploaded_file = st.file_uploader("Upload a PDF file", type=("pdf"))

if uploaded_file:
    files = {'file': (uploaded_file.name,
                      uploaded_file.getvalue(), uploaded_file.type)}
    try:
        response = requests.post(
            "http://localhost:5001/docprocess", files=files)
        if response.status_code == 200:
            st.write(response.json().get("message"))
        else:
            st.error(
                f"Failed to process document: {response.status_code} - {response.text}")
    except requests.ConnectionError as e:
        st.error(f"Connection error: {e}")

# Text input for asking questions about the document
question = st.text_input("Ask something about the document",
                         placeholder="Can you give me an insurance plan?")

if question:
    try:

        res = requests.post("http://localhost:5000/qna",
                            json={"question": question})
        if res.status_code == 200:
            st.write("### Question:")
            st.write(question)

            st.write("### Answer:")
            st.write(res.json().get("Answer"))
        else:
            st.error(f"Failed to get answer: {res.status_code} - {res.text}")
    except requests.ConnectionError as e:
        st.error(f"Connection error: {e}")
