# tesla_ui.py

import streamlit as st
from tesla_chatbot import get_qa_chain

st.set_page_config(page_title="Tesla Manual Chatbot")

st.title("🤖 Tesla Manual Q&A Bot")

# Cache the chain so it's not rebuilt every time
@st.cache_resource
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

question = st.text_input("Ask a question about the Tesla Owner's Manual:")

if question:
    with st.spinner("Searching..."):
        response = qa_chain.invoke(question)
        st.success(response['result'])