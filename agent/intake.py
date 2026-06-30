from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

# ── State Schema ──────────────────────────────────────────────
# Defined once here, all future nodes will import and extend this

class QuoteFlowState(TypedDict):
    messages: list                  # full conversation history
    lead_name: Optional[str]        # extracted by intake
    contact: Optional[str]          # phone or email
    service_type: Optional[str]     # classified by router (later)
    in_service_area: Optional[bool] # checked by router (later)
    needs_site_visit: Optional[bool]
    qualify_answers: Optional[dict]
    price_range: Optional[tuple]
    decision: Optional[str]         # accepted | more_info | escalate
    lead_object: Optional[dict]     # final payload sent to handoff
    memory: Optional[dict]          # carries context across loops
    reclassified: Optional[bool]    # flags router second pass


# ── LLM ───────────────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ── Intake Node ───────────────────────────────────────────────

INTAKE_SYSTEM_PROMPT = """You are a friendly assistant for a home service business.
Your job in this step is to warmly greet the customer and collect two things:
1. Their name
2. Their contact information (phone number or email)

Keep it conversational and brief. Once you have both, confirm them back to the customer.
Do not ask about the job yet — that comes next.
If the customer volunteers job information, acknowledge it briefly but stay focused on getting their name and contact first."""


def intake_node(state: QuoteFlowState) -> QuoteFlowState:
    """Greet the lead and collect name + contact info."""

    messages = state.get("messages", [])

    # Build message list for Claude
    claude_messages = [SystemMessage(content=INTAKE_SYSTEM_PROMPT)]
    claude_messages.extend(messages)

    # Call Claude
    response = llm.invoke(claude_messages)

    # Append Claude's response to history
    updated_messages = messages + [AIMessage(content=response.content)]

    return {
        **state,
        "messages": updated_messages,
    }


# ── Graph ─────────────────────────────────────────────────────

def build_intake_graph():
    graph = StateGraph(QuoteFlowState)
    graph.add_node("intake", intake_node)
    graph.set_entry_point("intake")
    graph.add_edge("intake", END)
    return graph.compile()


# ── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    graph = build_intake_graph()

    # Simulate a customer's first message
    result = graph.invoke({
        "messages": [HumanMessage(content="Hi, I need to get a quote for cleaning my gutters")],
        "lead_name": None,
        "contact": None,
        "service_type": None,
        "in_service_area": None,
        "needs_site_visit": None,
        "qualify_answers": None,
        "price_range": None,
        "decision": None,
        "lead_object": None,
        "memory": {},
        "reclassified": False,
    })

    print("\n── Agent Response ──")
    print(result["messages"][-1].content)