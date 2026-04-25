# embedding.py
# Responsible for creating embeddings and storing in vector DB

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_db(chunks):
    embeddings = OpenAIEmbeddings()

    vector_db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="./db"
    )

    return vector_db