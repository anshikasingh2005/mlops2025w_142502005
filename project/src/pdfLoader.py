#!/home/shivani/mlopsPro/mlops2025w_142502005/.venv/bin/python



import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


pdfstr = "/home/shivani/mlopsPro/mlops2025w_142502005/project/data/NCERT-Class-6-History.pdf"
loader = PyPDFLoader(pdfstr)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
splits = text_splitter.split_documents(docs)
print(splits[0])
vectorstore = Chroma.from_documents(documents = splits, embedding = OpenAIEmbeddings())
