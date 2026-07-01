from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.intake import QuoteFlowState
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ── Decision system prompt ─────────────────────────────────────

DECISION_SYSTEM_PROMPT = """You are a friendly assistant for a home service business.

You have just calculated a price range for the customer's job:
Service: {service_type}
Price range: ${price_min} – ${price_max}

Your job is to:
1. Present the price range to the customer clearly and confidently
2. Explain briefly what it includes
3. Ask if they would like to proceed and book

Keep it warm, brief, and professional. Do not apologize for the price.
If the price range is None, let the customer know this job requires an in-person
estimate and offer to book a free site visit instead.

After presenting the quote, listen for one of three responses:
- They want to proceed → decision: accepted
- They have more questions → decision: more_info
- They seem hesitant or want to think → decision: escalate (offer to connect with a human)"""

DECISION_CLASSIFIER_PROMPT = """Based on the customer's last message, classify their response.

Return ONLY one of these three words:
- accepted
- more_info  
- escalate

Customer's last message: {last_message}"""


def decision_node(state: QuoteFlowState) -> QuoteFlowState:
    """Present the quote to the customer and capture their decision."""

    messages = state.get("messages", [])
    service_type = state.get("service_type", "unknown")
    price_range = state.get("price_range")
    needs_site_visit = state.get("needs_site_visit", False)

    # Build price display
    if price_range and not needs_site_visit:
        price_min, price_max = price_range
    else:
        price_min, price_max = None, None

    system_prompt = DECISION_SYSTEM_PROMPT.format(
        service_type=service_type.replace("_", " ").title(),
        price_min=price_min,
        price_max=price_max,
    )

    human_context = " ".join(
        m.content for m in messages if isinstance(m, HumanMessage)
    )

    claude_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the conversation so far: {human_context}\n\nPlease present the quote.")
    ]

    response = llm.invoke(claude_messages)
    updated_messages = messages + [AIMessage(content=response.content)]

    # If there's a customer reply after the quote, classify their decision
    decision = "more_info"  # default until customer responds
    if len(messages) > 0:
        last_human = next(
            (m.content for m in reversed(messages)
             if isinstance(m, HumanMessage)), None
        )
        if last_human:
            classifier_response = llm.invoke([
                SystemMessage(content="You are a classifier. Return only one word: accepted, more_info, or escalate."),
                HumanMessage(content=DECISION_CLASSIFIER_PROMPT.format(
                    last_message=last_human
                ))
            ])
            raw = classifier_response.content.strip().lower()
            if raw in ["accepted", "more_info", "escalate"]:
                decision = raw

    return {
        **state,
        "messages": updated_messages,
        "decision": decision,
    }


# ── Quick test ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: customer accepts the quote
    test1 = {
        "messages": [
            HumanMessage(content="I need my gutters cleaned in Mississauga"),
            AIMessage(content="Happy to help! How many storeys is your home?"),
            HumanMessage(content="2 storeys, about 120 linear feet"),
            AIMessage(content="Great, let me put together a quote for you!"),
            HumanMessage(content="Sounds good, let's do it"),
        ],
        "lead_name": "Sarah",
        "contact": "sarah@email.com",
        "service_type": "gutter_cleaning",
        "in_service_area": True,
        "needs_site_visit": False,
        "qualify_answers": ["storeys_2"],
        "price_range": (276, 374),
        "decision": None,
        "lead_object": None,
        "memory": {},
        "reclassified": False,
    }

    # Test 2: customer wants more info
    test2 = {
        **test1,
        "messages": [
            *test1["messages"][:-1],
            HumanMessage(content="What exactly does that include?"),
        ],
    }

    # Test 3: needs site visit
    test3 = {
        **test1,
        "needs_site_visit": True,
        "price_range": None,
        "messages": [
            HumanMessage(content="I need a full house demolition in Toronto"),
            AIMessage(content="Happy to help! Can you tell me more about the structure?"),
            HumanMessage(content="It's a 3 storey detached home, about 2500 sq ft"),
        ],
        "service_type": "demolition",
    }

    for i, test in enumerate([test1, test2, test3], 1):
        result = decision_node(test)
        print(f"\n── Test {i} ──")
        print(f"  Decision: {result['decision']}")
        print(f"  Response: {result['messages'][-1].content}")