from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from openai_core import graph

st.title("💬 Chatbot")

prompt = st.text_input("Enter your question here:")

if prompt:
        user_message = st.chat_message("human")
        user_message.write(prompt)

        with st.spinner("Generating response..."):
                ai_message = st.chat_message("ai")
                ai_reply = graph.invoke({"question": prompt})
                ai_message.write(ai_reply["generation"])


# Add a footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")