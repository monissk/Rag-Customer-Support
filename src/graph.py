# graph.py
# LangGraph workflow

from langgraph.graph import StateGraph
from src.rag_pipeline import generate_answer
from src.hitl import check_escalation, escalate_to_human

def build_graph(retriever):

    def process_node(state):
        answer = generate_answer(state["query"], retriever)
        state["answer"] = answer
        return state

    def decision_node(state):
        if check_escalation(state["answer"]):
            state["answer"] = escalate_to_human(state["query"])
        return state

    graph = StateGraph(dict)

    graph.add_node("process", process_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("process")
    graph.add_edge("process", "decision")

    return graph.compile()