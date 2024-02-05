# import streamlit as st
# from dotenv import load_dotenv
# import pickle
# import PyPDF2
# import fitz  # PyMuPDF
# from PyPDF2 import PdfFileReader
# from streamlit_extras.add_vertical_space import add_vertical_space
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
# import os
# import openai




# directory_path = r'input_file'
# directory_contents = os.listdir(directory_path)
# text = " "

# for item in directory_contents:
#     pdf_path = os.path.join(directory_path, item)

#     if item.endswith('.pdf'):
#         with open(pdf_path, 'rb') as pdf_file:
#             pdf_reader = PyPDF2.PdfReader(pdf_file)
#             num_pages = len(pdf_reader.pages)

#             for page_num in range(num_pages):
#                 page = pdf_reader.pages[page_num]
#                 text += page.extract_text()
#             pdf_file.close()  # Close the PDF file after use

#         save_directory = r'input_file_text'

#         if not os.path.exists(save_directory):
#             os.makedirs(save_directory)
#         # Define the path to save the .txt file
#         save_txt_path = os.path.join(save_directory, f"{item.replace('.pdf', '')}.txt")
#         # Save the extracted text to the .txt file
#         with open(save_txt_path, 'w', encoding='utf-8') as file:
#             file.write(text)

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )

#         chunks = text_splitter.split_text(text=text)

#         # EMBEDDING
#         store_name, _ = os.path.splitext(item)
#         store_path = os.path.join(r'pkl_file', f"{store_name}.pkl")

#         if os.path.exists(store_path):
#             with open(store_path, "rb") as f:
#                 VectorStore = pickle.load(f)
#         else:
#             embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#             VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#             with open(store_path, "wb") as f:
#                 pickle.dump(VectorStore, f)





import streamlit as st
from dotenv import load_dotenv
import pickle
import PyPDF2
from PyPDF2 import PdfFileReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import openai

# Load environment variables from a .env file if present
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

directory_path = r'input_file'
directory_contents = os.listdir(directory_path)
text = " "

for item in directory_contents:
    pdf_path = os.path.join(directory_path, item)

    if item.endswith('.pdf'):
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            pdf_file.close()  # Close the PDF file after use

        save_directory = r'input_file_text'

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # Define the path to save the .txt file
        save_txt_path = os.path.join(save_directory, f"{item.replace('.pdf', '')}.txt")
        # Save the extracted text to the .txt file
        with open(save_txt_path, 'w', encoding='utf-8') as file:
            file.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # EMBEDDING
        store_name, _ = os.path.splitext(item)
        store_path = os.path.join(r'pkl_file', f"{store_name}.pkl")

        if os.path.exists(store_path):
            with open(store_path, "rb") as f:
                VectorStore = pickle.load(f)
        else:
            # Pass the API key as a named parameter to OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(store_path, "wb") as f:
                pickle.dump(VectorStore, f)