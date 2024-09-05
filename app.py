import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os
from langchain.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'memory' not in st.session_state:
    st.session_state.memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
load_dotenv()
groq_api_key=os.getenv('apikey')

def conversation_chain(vectorstore):
   retriever=vectorstore.as_retriever()
   llm=ChatGroq(model_name='llama3-8b-8192',groq_api_key=groq_api_key,temperature=0.6)
   chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,memory=st.session_state.memory)
   return chain 

def textRetriever(pdfs):
    
    text=''
    for pdf in pdfs:
       pdf_read=PdfReader(pdf)
       for page in pdf_read.pages:
          text+=page.extract_text()
    return text


def chunker(text):
   text_chunk=RecursiveCharacterTextSplitter(separators=['\n',' '],chunk_size=1000,chunk_overlap=200,length_function=len )
   chunks=text_chunk.split_text(text)
   return chunks

def embeds(chunks):
   model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
   vectorstore = FAISS.from_texts(texts=chunks, embedding=model)
   return vectorstore
   
st.set_page_config(page_title='pdf chatbot')
st.header("chat with multiple pdf")
prompt=st.text_input("ask a question about your pdf")
with st.sidebar:
    st.subheader('your document')
    pdfs= st.file_uploader('enter your pdf here',accept_multiple_files=True)
    if st.button('process'):
        with st.spinner('Processing'):
          #extract pdf text
          text= textRetriever(pdfs)
          st.write(text)
          #split into chunks
          chunks=chunker(text)
          st.write(chunks)
          #embedding
          vectorstore=embeds(chunks)
          st.write(vectorstore)
          #conversation chain
          st.session_state.conversation=conversation_chain(vectorstore) 

if prompt:
   response = st.session_state.conversation({'question': prompt, 'chat_history': st.session_state['chat_history']})
   st.write(response['answer'])
   st.write(response)

        
          

        
        
    