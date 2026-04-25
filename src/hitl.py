# hitl.py
# Human-in-the-loop logic

def check_escalation(answer):
    if "not found" in answer.lower() or len(answer) < 30:
        return True
    return False

def escalate_to_human(query):
    return f"⚠️ Escalated to human agent for query: {query}"