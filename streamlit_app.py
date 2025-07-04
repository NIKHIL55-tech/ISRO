# streamlit_app.py

import streamlit as st
import requests

st.title("🚀 MOSDAC AI HelpBot")
st.markdown("Ask any question related to MOSDAC satellite data!")

user_query = st.text_input("Enter your question")

if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/vector-query",
                    json={"query": user_query}
                )
                st.success(response.json()["response"])
            except Exception as e:
                st.error(f"Error: {e}")
