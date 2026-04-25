# retriever.py
# Responsible for retrieving relevant documents

def get_retriever(vector_db):
    return vector_db.as_retriever()