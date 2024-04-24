import os
import streamlit as st
import pickle
import faiss
import time
from langchain_community.llms.openai import OpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever 

from dotenv import load_dotenv
load_dotenv()  

st.set_page_config(page_title="News Data Summariser Tool 📰📰📰", page_icon=":newspaper:", layout="wide")
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #00b4d8;
            padding: 2rem;
        }
        .st-bd {
            background-color: #f6f5f5;
        }
        .st-br {
            border-radius: 15px;
        }
        .stHeader {
            color: #00b4d8 !important;
        }
        .stTextInput>div>div>input {
            background-color: #f6f5f5;
            border: 2px solid #00b4d8;
            border-radius: 10px;
            color: #222831;
        }
        .stButton>button {
            background-color: #00b4d8;
            color: white;
            border-radius: 10px;
            padding: 0.375rem 0.75rem;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #0077b6;
            transform: scale(1.05);
        }
        .result-container {
            padding: 20px;
            background-color: #f6f5f5;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# UI layout
st.title("News Data Summariser Tool 📰📰📰")

# Sidebar section for URL input and process button
st.sidebar.header("Input Section")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

file_path =  "faiss_store_openai.index"
llm = OpenAI(temperature=0.9, max_tokens=500)

main_placeholder = st.empty()

process_url_clicked = st.sidebar.button("Process URLs")

# Result section on the main page
st.header("Result Section")
query = st.text_input("Ask a Question:")
if process_url_clicked:
    with st.spinner('Processing...'):
        if any(urls):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            main_placeholder.text("Data Loading...Started...✅✅✅")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            embeddings = OpenAIEmbeddings()
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            vectorstore_openai = FAISS.from_documents(docs, embeddings)

            file_path = "faiss_store_openai.index"
            faiss.write_index(vectorstore_openai.index, file_path)

if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        faiss_index = faiss.read_index(file_path)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faiss_index.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.subheader("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
