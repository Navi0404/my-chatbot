
# pip install --upgrade streamlit
# python -m pip install --upgrade streamlit-extras
# pip install --upgrade openai
# pip install PyMuPDF

import streamlit as st
from dotenv import load_dotenv
import pickle
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai

# Sidebar contents
## Streamlit UI

st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

OPENAI_API_KEY=''


def main():
    # calling openai
    load_dotenv()

    st.header("Chat with PDF")
    # store_path = os.path.abspath(r"pkl_file")
    store_path = os.path.abspath(r"pkl_file")
    vector_stores = []

    if os.path.exists(store_path):
        for filename in os.listdir(store_path):
            if filename.endswith(".pkl"):
                file_path = os.path.join(store_path, filename)
                with open(file_path, "rb") as f:
                    vector_store = pickle.load(f)
                    vector_stores.append(vector_store)
    else:
        st.warning("Vector data not found. Please upload PDFs first.")

    

    # Accept user questions
    query = st.text_input("Ask a question about your PDF file:")

    if query and vector_stores:
        for selected_vector_store in vector_stores:
            # Search for relevant documents using semantic similarity
            docs = selected_vector_store.similarity_search(query=query, k=3)

            # Use OpenAI for question-answering
            llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

        st.write(response)

if __name__ == '__main__':
    main()
    