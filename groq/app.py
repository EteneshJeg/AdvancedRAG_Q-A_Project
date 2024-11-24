import streamlit as st
import os 
from langchain_groq import ChatGroq 
from langchain_community.document_loaders import WebBaseLoader
import ollama

response = ollama.embeddings(
    model="mxbai-embed-large",
    prompt="Llamas are members of the camelid family."
)
print(response)

from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain 
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embedding = OllamaEmbeddings(model="mxbai-embed-large")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:5])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embedding)
st.markdown("<h1 style='text-align: center; color: blue;'>AI-Powered Q&A Assistant</h1>", unsafe_allow_html=True)
st.title("Ask a Question based on the Your Website content")
llm=ChatGroq(groq_api_key=groq_api_key,
            model_name="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the Questions based on the provided context only
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

document_chain= create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain= create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input Your Prompt Here!")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input": prompt})
    print("Respones time :" ,time.process_time() - start)
    st.write(response['answer'])

    # with a streamlit expander 
    with st.expander("Document Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------")


