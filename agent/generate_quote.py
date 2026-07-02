from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.intake import QuoteFlowState
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ── Pricing rules ──────────────────────────────────────────────
# Base rates and modifiers per service.
# In production these are fetched from Supabase per business owner config.
# All prices in CAD.

PRICING_RULES = {
    "gutter_cleaning": {
        "base": 150,
        "modifiers": {
            "storeys_2": 75,
            "storeys_3plus": 150,
            "large_footage": 100,   # > 150 linear feet
        },
        "margin": 0.15              # ±15% range
    },
    "lawn_care": {
        "base": 60,
        "modifiers": {
            "medium_lawn": 40,
            "large_lawn": 100,
            "recurring_discount": -10,
        },
        "margin": 0.15
    },
    "junk_removal": {
        "base": 150,
        "modifiers": {
            "half_truckload": 100,
            "full_truckload": 250,
            "heavy_items": 75,
        },
        "margin": 0.15
    },
    "house_cleaning": {
        "base": 120,
        "modifiers": {
            "per_bedroom": 30,
            "per_bathroom": 25,
            "deep_clean": 80,
            "move_in_out": 120,
        },
        "margin": 0.15
    },
    "window_cleaning": {
        "base": 100,
        "modifiers": {
            "per_10_windows": 40,
            "storey_2": 50,
            "storey_3plus": 100,
        },
        "margin": 0.15
    },
    "pressure_washing": {
        "base": 150,
        "modifiers": {
            "per_500sqft": 75,
            "deck_add": 50,
            "siding_add": 100,
        },
        "margin": 0.15
    },
    "snow_removal": {
        "base": 80,
        "modifiers": {
            "driveway_and_walkway": 40,
            "seasonal_contract": 400,
        },
        "margin": 0.15
    },
    "handyman": {
        "base": 100,
        "modifiers": {
            "materials_supplied": 50,
            "per_hour_extra": 65,
        },
        "margin": 0.20
    },
    "plumbing": {
        "base": 150,
        "modifiers": {
            "urgent": 100,
            "installation": 200,
        },
        "margin": 0.20
    },
    "hvac": {
        "base": 200,
        "modifiers": {
            "repair": 150,
            "new_installation": 800,
        },
        "margin": 0.20
    },
    "demolition": {
        "base": 500,
        "modifiers": {
            "shed_deck": 300,
            "interior_walls": 800,
            "full_structure": 3000,
        },
        "margin": 0.20
    },
    "painting": {
        "base": 200,
        "modifiers": {
            "per_room": 150,
            "exterior_add": 500,
            "per_100sqft": 80,
        },
        "margin": 0.15
    },
}

# ── Quote extraction prompt ────────────────────────────────────

QUOTE_EXTRACTION_PROMPT = """You are a pricing assistant for a home service business.

Based on the conversation history, extract the customer's answers and calculate a quote.

Service type: {service_type}
Pricing rules: {pricing_rules}

Instructions:
1. Read the conversation and identify what the customer told us about their job
2. Map their answers to the relevant modifiers in the pricing rules
3. Calculate: base_rate + applicable modifiers = estimated_price
4. Apply ±{margin}% to get a price range (min and max)
5. Return ONLY a JSON object with these fields:
   - "estimated_price": the mid-point number
   - "price_min": estimated_price minus margin
   - "price_max": estimated_price plus margin
   - "modifiers_applied": list of modifier names used
   - "assumptions": any assumptions made due to missing info

Return ONLY valid JSON. No explanation, no markdown."""


def generate_quote_node(state: QuoteFlowState) -> QuoteFlowState:
    """Apply pricing config to qualify answers and produce a price range."""

    messages = state.get("messages", [])
    service_type = state.get("service_type", "unknown")

    # Get pricing rules for this service
    rules = PRICING_RULES.get(service_type)

    if not rules:
        return {
            **state,
            "needs_site_visit": True,
            "price_range": None,
        }

    # Don't quote without qualify answers — too inaccurate
    qualify_answers = state.get("qualify_answers")
    if not qualify_answers:
        return {
            **state,
            "needs_site_visit": True,
            "price_range": None,
            "decision": "more_info",
        }
        
    margin_pct = rules["margin"]
    margin_display = int(margin_pct * 100)

    system_prompt = QUOTE_EXTRACTION_PROMPT.format(
        service_type=service_type.replace("_", " ").title(),
        pricing_rules=json.dumps(rules, indent=2),
        margin=margin_display
    )

    human_context = " ".join(
        m.content for m in messages if isinstance(m, HumanMessage)
    )

    claude_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the conversation so far: {human_context}\n\nCalculate the quote.")
    ]

    response = llm.invoke(claude_messages)

    # Parse the quote
    try:
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        quote_data = json.loads(raw)
        price_range = (
            round(quote_data["price_min"]),
            round(quote_data["price_max"])
        )
    except (json.JSONDecodeError, KeyError) as e:
        return {
            **state,
            "needs_site_visit": True,
            "price_range": None,
        }

    return {
        **state,
        "price_range": price_range,
        "qualify_answers": quote_data.get("modifiers_applied", []),
    }


# ── Quick test ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: gutter cleaning - 2 storey home
    test1 = {
        "messages": [
            HumanMessage(content="I need my gutters cleaned in Mississauga"),
            AIMessage(content="Happy to help! How many storeys is your home?"),
            HumanMessage(content="It's a 2 storey home, maybe around 120 linear feet of gutters"),
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

    # Test 2: junk removal - full truckload with heavy items
    test2 = {
        **test1,
        "messages": [
            HumanMessage(content="I need junk removed in Toronto"),
            AIMessage(content="Sure! How much junk are we talking?"),
            HumanMessage(content="Probably a full truckload, includes an old fridge and washing machine"),
        ],
        "service_type": "junk_removal",
    }

    # Test 3: demolition
    test3 = {
        **test1,
        "messages": [
            HumanMessage(content="I need a shed demolished in Brampton"),
            AIMessage(content="Got it! What size is the shed roughly?"),
            HumanMessage(content="It's a medium sized shed, maybe 10x12 feet"),
        ],
        "service_type": "demolition",
    }

    for i, test in enumerate([test1, test2, test3], 1):
        result = generate_quote_node(test)
        price = result.get("price_range")
        if price:
            print(f"\n── Test {i} ({test['service_type']}) ──")
            print(f"  Price range:  ${price[0]:,} – ${price[1]:,}")
            print(f"  Modifiers:    {result.get('qualify_answers')}")
        else:
            print(f"\n── Test {i} ── needs site visit")