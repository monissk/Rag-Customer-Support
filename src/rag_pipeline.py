# rag_pipeline.py
# Core RAG logic

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

def generate_answer(query, retriever):

    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a customer support assistant.

    Answer ONLY from the context.
    If answer is not found, say "Not found".

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return response.content