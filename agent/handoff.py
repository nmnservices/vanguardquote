from langchain_core.messages import HumanMessage, AIMessage
from agent.intake import QuoteFlowState
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import json

load_dotenv()


# ── Handoff node ───────────────────────────────────────────────

def handoff_node(state: QuoteFlowState) -> QuoteFlowState:
    """Package the lead into a structured object and fire it to the business owner."""

    # Build the lead object from state
    price_range = state.get("price_range")
    lead_object = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lead_name": state.get("lead_name") or _extract_name(state),
        "contact": state.get("contact") or _extract_contact(state),
        "service_type": state.get("service_type", "unknown"),
        "in_service_area": state.get("in_service_area"),
        "needs_site_visit": state.get("needs_site_visit", False),
        "price_range": {
            "min": price_range[0] if price_range else None,
            "max": price_range[1] if price_range else None,
            "currency": "CAD"
        },
        "qualify_answers": state.get("qualify_answers", []),
        "decision": state.get("decision", "unknown"),
        "conversation_turns": len(state.get("messages", [])),
        "lead_quality": _score_lead(state),
    }

    # Fire the lead — currently prints to console.
    # In production: POST to webhook, send email via SendGrid,
    # or push to Slack. We wire this up in Stage 5 (deployment).
    _deliver_lead(lead_object)

    return {
        **state,
        "lead_object": lead_object,
    }


def _extract_name(state: QuoteFlowState) -> str:
    """Try to extract name from conversation if not in state."""
    messages = state.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content.lower()
            if "i'm" in content or "i am" in content or "my name is" in content:
                return msg.content  # raw — router will clean this later
    return "Unknown"


def _extract_contact(state: QuoteFlowState) -> str:
    """Try to extract contact from conversation if not in state."""
    messages = state.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            # Basic check for email or phone pattern
            if "@" in content or any(c.isdigit() for c in content):
                return content
    return "Unknown"


def _score_lead(state: QuoteFlowState) -> str:
    """Score lead quality based on available signals."""
    score = 0

    if state.get("lead_name"):
        score += 1
    if state.get("contact"):
        score += 1
    if state.get("in_service_area"):
        score += 2
    if state.get("decision") == "accepted":
        score += 3
    if state.get("price_range"):
        score += 1
    if not state.get("needs_site_visit"):
        score += 1

    if score >= 7:
        return "hot"
    elif score >= 4:
        return "warm"
    else:
        return "cold"


def _deliver_lead(lead_object: dict):
    """
    Deliver the lead to the business owner.
    Currently: prints a formatted summary to console.
    Next: POST to webhook URL stored in Supabase config.
    """
    print("\n" + "="*50)
    print("🔔  NEW LEAD — QuoteFlow")
    print("="*50)
    print(f"  Name:          {lead_object['lead_name']}")
    print(f"  Contact:       {lead_object['contact']}")
    print(f"  Service:       {lead_object['service_type'].replace('_', ' ').title()}")
    print(f"  In Area:       {lead_object['in_service_area']}")
    print(f"  Site Visit:    {lead_object['needs_site_visit']}")
    if lead_object['price_range']['min']:
        print(f"  Quote:         ${lead_object['price_range']['min']:,} – ${lead_object['price_range']['max']:,} CAD")
    else:
        print(f"  Quote:         Needs site visit")
    print(f"  Decision:      {lead_object['decision']}")
    print(f"  Lead Quality:  {lead_object['lead_quality'].upper()}")
    print(f"  Timestamp:     {lead_object['timestamp']}")
    print("="*50)


# ── Quick test ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: hot lead — accepted quote, in area, full info
    test1 = {
        "messages": [
            HumanMessage(content="I need my gutters cleaned in Mississauga"),
            AIMessage(content="Happy to help! Could I get your name and contact?"),
            HumanMessage(content="I'm Sarah, sarah@email.com"),
            AIMessage(content="How many storeys is your home?"),
            HumanMessage(content="2 storeys, about 120 linear feet"),
            AIMessage(content="Your quote is $276–$374. Would you like to book?"),
            HumanMessage(content="Yes let's do it!"),
        ],
        "lead_name": "Sarah",
        "contact": "sarah@email.com",
        "service_type": "gutter_cleaning",
        "in_service_area": True,
        "needs_site_visit": False,
        "qualify_answers": ["storeys_2"],
        "price_range": (276, 374),
        "decision": "accepted",
        "lead_object": None,
        "memory": {},
        "reclassified": False,
    }

    # Test 2: warm lead — needs site visit
    test2 = {
        **test1,
        "lead_name": "Mike",
        "contact": "416-555-0123",
        "service_type": "demolition",
        "needs_site_visit": True,
        "price_range": None,
        "decision": "more_info",
    }

    # Test 3: cold lead — out of area
    test3 = {
        **test1,
        "lead_name": None,
        "contact": None,
        "service_type": "lawn_care",
        "in_service_area": False,
        "price_range": None,
        "decision": "escalate",
    }

    for i, test in enumerate([test1, test2, test3], 1):
        print(f"\n\n── Test {i} ──")
        handoff_node(test)