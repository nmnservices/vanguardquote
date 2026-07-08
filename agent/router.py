from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.intake import VanguardQuoteState
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ── Service taxonomy ───────────────────────────────────────────
# This is the master list of services the agent knows about.
# In production this comes from the business owner's Supabase config.
# For now we hardcode it so the router has something to classify against.

KNOWN_SERVICES = [
    "gutter_cleaning",
    "lawn_care",
    "junk_removal",
    "house_cleaning",
    "window_cleaning",
    "pressure_washing",
    "snow_removal",
    "handyman",
    "plumbing",
    "hvac",
    "demolition",
    "painting",
]

# ── Dummy service area ─────────────────────────────────────────
# In production this comes from Supabase config per business.
# Format: list of cities/regions the business serves.

SERVICE_AREA = [
    "toronto", "mississauga", "brampton", "vaughan",
    "markham", "richmond hill", "oakville", "burlington",
    "hamilton", "ajax", "pickering", "whitby", "oshawa"
]

# ── Router system prompt ───────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a classifier for a home service business AI agent.

Given the conversation history, extract and return a JSON object with these fields:

1. "service_type": the most likely service being requested. Must be one of: {services}
   If none match, use "unknown".

2. "in_service_area": true or false. The customer's location (if mentioned) must be in: {area}
   If location not mentioned yet, return null.

3. "needs_site_visit": true or false. Return true ONLY if:
   - The job is explicitly very large (full house demolition, commercial work)
   - The customer explicitly says they are unsure of measurements
   Default to false for standard residential jobs like gutter cleaning, lawn care,
   junk removal, window cleaning, house cleaning. These can always be quoted remotely.
   If unclear, return false.

4. "reclassified": true if this is a correction of a previous classification, false otherwise.

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks."""


def router_node(state: VanguardQuoteState) -> VanguardQuoteState:
    """Classify service type, check service area, flag if site visit needed."""

    messages = state.get("messages", [])

    # Build prompt with service list and area injected
    system_prompt = ROUTER_SYSTEM_PROMPT.format(
        services=", ".join(KNOWN_SERVICES),
        area=", ".join(SERVICE_AREA)
    )

    # Anthropic requires conversation to end with a human message.
    # Extract just the human messages for classification context.
    human_context = " ".join(
        m.content for m in messages if isinstance(m, HumanMessage)
    )

    claude_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Classify this conversation: {human_context}")
    ]

    response = llm.invoke(claude_messages)

    # Parse the JSON response
    try:
        raw = response.content.strip()
        classification = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if Claude doesn't return clean JSON
        classification = {
            "service_type": "unknown",
            "in_service_area": None,
            "needs_site_visit": False,
            "reclassified": False,
        }

    # Standard residential services never need a site visit by default
    ALWAYS_QUOTABLE = [
        "gutter_cleaning", "lawn_care", "junk_removal", "house_cleaning",
        "window_cleaning", "pressure_washing", "snow_removal", "painting"
    ]
    service = classification.get("service_type", "unknown")
    needs_visit = classification.get("needs_site_visit", False)
    if service in ALWAYS_QUOTABLE:
        needs_visit = False

    return {
        **state,
        "service_type": service,
        "in_service_area": classification.get("in_service_area", None),
        "needs_site_visit": needs_visit,
        "reclassified": classification.get("reclassified", False),
    }


# ── Quick test ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: clear service, in area
    test1 = {
        "messages": [
            HumanMessage(content="Hi I need someone to clean my gutters in Mississauga"),
            AIMessage(content="I'd love to help! Could I get your name and contact info?"),
            HumanMessage(content="Sure, I'm John, john@email.com"),
        ],
        "lead_name": "John",
        "contact": "john@email.com",
        "service_type": None,
        "in_service_area": None,
        "needs_site_visit": None,
        "qualify_answers": None,
        "price_range": None,
        "decision": None,
        "lead_object": None,
        "memory": {},
        "reclassified": False,
    }

    # Test 2: ambiguous service
    test2 = {
        **test1,
        "messages": [
            HumanMessage(content="I need to get rid of some stuff in my basement in Toronto"),
        ],
    }

    # Test 3: out of area
    test3 = {
        **test1,
        "messages": [
            HumanMessage(content="I need my lawn cut in Ottawa"),
        ],
    }

    for i, test in enumerate([test1, test2, test3], 1):
        result = router_node(test)
        print(f"\n── Test {i} ──")
        print(f"  service_type:     {result['service_type']}")
        print(f"  in_service_area:  {result['in_service_area']}")
        print(f"  needs_site_visit: {result['needs_site_visit']}")
        print(f"  reclassified:     {result['reclassified']}")