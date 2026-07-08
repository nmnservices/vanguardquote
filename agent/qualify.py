from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.intake import VanguardQuoteState
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ── Pricing config ─────────────────────────────────────────────
# In production this is fetched from Supabase per business.
# Each service has qualifying questions that drive the quote calculation.

PRICING_CONFIG = {
    "gutter_cleaning": {
        "qualify_questions": [
            "How many storeys is the home?",
            "Roughly how many linear feet of guttering do you have? (if unsure, just say so)",
        ]
    },
    "lawn_care": {
        "qualify_questions": [
            "What is the approximate size of the lawn? (small/medium/large or sq ft)",
            "Is this a one-time cut or are you looking for recurring service?",
        ]
    },
    "junk_removal": {
        "qualify_questions": [
            "How much junk are we talking — a few items, half a truckload, or a full truckload?",
            "Are there any heavy or special items like appliances, mattresses, or construction debris?",
        ]
    },
    "house_cleaning": {
        "qualify_questions": [
            "How many bedrooms and bathrooms does the home have?",
            "Is this a standard clean, deep clean, or move-in/move-out clean?",
        ]
    },
    "window_cleaning": {
        "qualify_questions": [
            "How many windows approximately?",
            "How many storeys is the home?",
        ]
    },
    "pressure_washing": {
        "qualify_questions": [
            "What surface needs pressure washing? (driveway, deck, siding, etc.)",
            "Roughly what square footage are we looking at?",
        ]
    },
    "snow_removal": {
        "qualify_questions": [
            "Is this for a driveway, walkway, or both?",
            "Are you looking for a one-time visit or seasonal contract?",
        ]
    },
    "handyman": {
        "qualify_questions": [
            "What specific task needs to be done?",
            "Do you have the materials already or will the handyman need to supply them?",
        ]
    },
    "plumbing": {
        "qualify_questions": [
            "What is the plumbing issue? (leak, clog, installation, etc.)",
            "Is this urgent or can it be scheduled?",
        ]
    },
    "hvac": {
        "qualify_questions": [
            "Is this for maintenance, repair, or a new installation?",
            "What type of system do you have? (central air, heat pump, furnace, etc.)",
        ]
    },
    "demolition": {
        "qualify_questions": [
            "What needs to be demolished? (shed, deck, interior walls, full structure, etc.)",
            "Roughly what size is the structure?",
        ]
    },
    "painting": {
        "qualify_questions": [
            "Is this interior or exterior painting?",
            "How many rooms or what square footage needs to be painted?",
        ]
    },
    "unknown": {
        "qualify_questions": [
            "Could you describe what you need done in a bit more detail?",
        ]
    }
}


# ── Qualify system prompt ──────────────────────────────────────

QUALIFY_SYSTEM_PROMPT = """You are a friendly assistant for a home service business.

The customer has been identified as needing: {service_type}

Your job is to ask the following qualifying questions to gather enough information for a quote.
Ask them conversationally — do not list them all at once. Ask one or two at a time naturally.

Questions to cover:
{questions}

Once you have answers to all the questions, say something like:
"Great, I have everything I need to put together a quote for you!"

Keep responses brief and friendly. Do not give a price yet."""


def qualify_node(state: VanguardQuoteState) -> VanguardQuoteState:
    """Ask config-driven clarifying questions based on classified service type."""

    messages = state.get("messages", [])
    service_type = state.get("service_type", "unknown")

    # Get questions for this service type
    config = PRICING_CONFIG.get(service_type, PRICING_CONFIG["unknown"])
    questions = "\n".join(
        f"- {q}" for q in config["qualify_questions"]
    )

    system_prompt = QUALIFY_SYSTEM_PROMPT.format(
        service_type=service_type.replace("_", " ").title(),
        questions=questions
    )

    # Anthropic requires conversation to end with a human message.
    # Summarize conversation context and re-frame as a human turn.
    human_context = " ".join(
        m.content for m in messages if isinstance(m, HumanMessage)
    )

    claude_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the conversation so far: {human_context}\n\nPlease ask the next qualifying question.")
    ]

    response = llm.invoke(claude_messages)

    updated_messages = messages + [AIMessage(content=response.content)]

    # Store qualify answers in state (will be populated as conversation continues)
    return {
        **state,
        "messages": updated_messages,
        "qualify_answers": state.get("qualify_answers") or {},
    }


# ── Quick test ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: gutter cleaning
    test1 = {
        "messages": [
            HumanMessage(content="Hi, I need my gutters cleaned in Mississauga"),
            AIMessage(content="Happy to help! Could I get your name and contact info?"),
            HumanMessage(content="Sure, I'm Sarah, sarah@email.com"),
        ],
        "lead_name": "Sarah",
        "contact": "sarah@email.com",
        "service_type": "gutter_cleaning",
        "in_service_area": True,
        "needs_site_visit": False,
        "qualify_answers": {},
        "price_range": None,
        "decision": None,
        "lead_object": None,
        "memory": {},
        "reclassified": False,
    }

    # Test 2: junk removal
    test2 = {
        **test1,
        "messages": [
            HumanMessage(content="I need to get rid of some old furniture in Toronto"),
            AIMessage(content="Happy to help! Could I get your name and contact info?"),
            HumanMessage(content="I'm Mike, 416-555-0123"),
        ],
        "lead_name": "Mike",
        "contact": "416-555-0123",
        "service_type": "junk_removal",
    }

    for i, test in enumerate([test1, test2], 1):
        result = qualify_node(test)
        print(f"\n── Test {i} ({test['service_type']}) ──")
        print(result["messages"][-1].content)