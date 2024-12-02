from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
import pandas as pd
import joblib
import torch
from langchain_nomic import NomicEmbeddings
import uuid
import json
import streamlit as st

import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore

from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# La lecture des pdfs  et l'extraction du texte





def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
   



    




    


def get_simple_conversation(model = "llama3.1"):
    # template_messages = [
    #     SystemMessage(content="You are a helpfull assistant."),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     HumanMessagePromptTemplate.from_template("{text}"),
    # ]
    # prompt_template = ChatPromptTemplate.from_messages(template_messages)
        
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # chain = LLMChain(llm = Ollama(model=model), prompt=prompt_template, memory=memory)


    # prompt_template = PromptTemplate(...)
    prompt_template = PromptTemplate(input_variables=["text"], template= "{text}")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = OllamaLLM(model=model)
    chain = LLMChain(llm=llm, memory=memory, prompt=prompt_template)
    return chain






def get_chroma_vectorstore(embed_model,text_chunks):
    persist_dir = './Chroma'
    vector_store = Chroma.from_documents(
        text_chunks, embed_model, persist_directory=persist_dir
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return vector_store

def get_conversation_chain(vectorstore, model="llama3.1"):
    llm = OllamaLLM(model=model)
    formatted_prompt = """
    Si une réponse peut être trouvée à partir des documents fournis, donnez-la. 
    Si aucune réponse pertinente n'est disponible dans les documents, répondez en utilisant le modèle LLM uniquement.
    Contexte: {context}
    Question: {question}
    Réponse:
    """.strip()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"], 
        template=formatted_prompt
    )

    # Create retrieval chain
    combine_docs_chain = create_stuff_documents_chain(
        llm, prompt_template
    )
    retrieval_chain = create_retrieval_chain(vectorstore, combine_docs_chain)
    return retrieval_chain

    


def handle_user_input():
    user_question = st.session_state.fixed_input
    assistant_response = st.session_state.chain({"text": user_question})
    
    # Extract the text part of the assistant's response
    assistant_text = assistant_response['text']
    
    # Append the conversation history
    st.session_state.conversation.append({"role": "user", "content": user_question})
    st.session_state.conversation.append({"role": "assistant", "content": assistant_text})
    
