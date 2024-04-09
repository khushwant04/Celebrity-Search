import os
from langchain_community.llms import Ollama

import streamlit as st

# strealit framework
st.title("Langchain Demo With Ollama")
input_text = st.text_input("Enter the topic")

## LLMs
llm = Ollama(base_url = 'http://localhost:11434', model = "llama2",temperature=0.6)

if input_text:
    st.write(llm(input_text))
    