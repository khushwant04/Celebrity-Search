import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import streamlit as st
from langchain.memory import ConversationBufferMemory

# strealit framework
st.title("Celebrity Search Results")
input_text = st.text_input("Enter the topic")

## Prompt Templates

first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

## Memory 
person_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob',memory_key='chat_history')

## Ollama LLMs
llm = Ollama(base_url = 'http://localhost:11434', model = "llama2",temperature=0.6)
chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key="person",memory=person_memory)

## Prompt Templates
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born"
)

chain2 = LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key="dob",memory=dob_memory)

## Prompt Templates

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 Major events happend around {dob} in the world"
)

chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key="description",memory=descr_memory)

parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))
    
    with st.expander('Person Nmae'):
        st.info(person_memory.buffer)
        
    with st.expander('Major Events'):
        st.info(descr_memory)    
    