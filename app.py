import os

# SET API KEYS (DO NOT PUSH REAL KEYS TO GITHUB)

os.environ["OPENAI_API_KEY"] = input("Enter OpenAI Key: ")
os.environ["LANGCHAIN_API_KEY"] = input("Enter LangSmith Key: ")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# LOAD PDF
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("knowledge_base.pdf")  # Put your PDF in same folder
documents = loader.load()

# CHUNKING
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# EMBEDDING + VECTOR DB

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./db"
)

retriever = vector_db.as_retriever()

# LLM
import re

def simple_llm(context, query):
    """
    Improved answer extraction:
    - Selects multiple relevant sentences
    - Avoids short/incomplete answers
    """

    sentences = re.split(r'[.\n]', context)

    matched = []

    for sentence in sentences:
        sentence = sentence.strip()

        # skip very short lines
        if len(sentence) < 20:
            continue

        # match keywords
        if any(word.lower() in sentence.lower() for word in query.split()):
            matched.append(sentence)

    if not matched:
        return "Not found"

    # remove duplicates
    matched = list(dict.fromkeys(matched))

    # return top 3 sentences
    return ". ".join(matched[:3]) + "."

# RAG FUNCTION
def rag_answer(query):
    """
    RAG Pipeline:
    Query → Retrieve → Generate Answer
    """

    docs = retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n".join([doc.page_content for doc in docs])

    # Generate answer
    answer = simple_llm(context, query)

    return answer

# HITL (HUMAN ESCALATION)
def check_escalation(answer):
    def check_escalation(answer):
        if "not found" in answer.lower():
            return True
        return False

def human_intervention(query):
    return f"⚠️ Escalated to human agent for query: {query}"

# LANGGRAPH FLOW
from langgraph.graph import StateGraph

def process_node(state):
    answer = rag_answer(state["query"])
    state["answer"] = answer
    return state

def decision_node(state):
    if check_escalation(state["answer"]):
        state["answer"] = human_intervention(state["query"])
    return state

# Build Graph
graph = StateGraph(dict)

graph.add_node("process", process_node)
graph.add_node("decision", decision_node)

graph.set_entry_point("process")
graph.add_edge("process", "decision")

app = graph.compile()

# RUN SYSTEM
print("\n=== RAG CUSTOMER SUPPORT SYSTEM ===\n")

while True:
    query = input("Ask your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    result = app.invoke({"query": query})

    print("\nAnswer:")
    print(result["answer"])
    print("\n---------------------------\n")