from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import re
import os

load_dotenv()

# ── State Schema ──────────────────────────────────────────────

class QuoteFlowState(TypedDict):
    messages: list
    lead_name: Optional[str]
    contact: Optional[str]
    service_type: Optional[str]
    in_service_area: Optional[bool]
    needs_site_visit: Optional[bool]
    qualify_answers: Optional[dict]
    price_range: Optional[tuple]
    decision: Optional[str]
    lead_object: Optional[dict]
    memory: Optional[dict]
    reclassified: Optional[bool]


# ── LLM ───────────────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ── Name/contact extraction ───────────────────────────────────

NOT_NAMES = {"in", "a", "an", "the", "from", "at", "looking", "here", "not", "so", "just"}

def _extract_name_contact(messages: list):
    """Extract name and email/phone from human messages."""
    name = None
    contact = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content

            # Email
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', content)
            if email_match:
                contact = email_match.group()

            # Phone
            phone_match = re.search(r'\b\d[\d\s\-().]{7,}\d\b', content)
            if phone_match and not contact:
                contact = phone_match.group().strip()

            # Name — "my name is X" or "this is X" (most reliable)
            name_match = re.search(
                r"(?:my name is|this is)\s+([A-Z][a-z]+)",
                content, re.IGNORECASE
            )
            if name_match:
                name = name_match.group(1)
                continue

            # Fallback — "I'm X" only if X is not a common non-name word
            im_match = re.search(r"i'?m\s+([A-Za-z]+)", content, re.IGNORECASE)
            if im_match:
                candidate = im_match.group(1)
                if candidate.lower() not in NOT_NAMES:
                    name = candidate.capitalize()

    return name, contact


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

    claude_messages = [SystemMessage(content=INTAKE_SYSTEM_PROMPT)]
    claude_messages.extend(messages)

    response = llm.invoke(claude_messages)

    updated_messages = messages + [AIMessage(content=response.content)]

    name, contact = _extract_name_contact(messages)

    return {
        **state,
        "messages": updated_messages,
        "lead_name": name,
        "contact": contact,
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