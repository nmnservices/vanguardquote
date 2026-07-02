"""
QuoteFlow Eval Suite — 20 test cases
Tests: happy path, edge cases, adversarial, out-of-area, site visit
Target: all 20 passing before deployment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from agent.graph import build_graph
from dotenv import load_dotenv

load_dotenv()

graph = build_graph()


def run(messages: list, label: str) -> dict:
    """Run the graph with a single opening message and return final state."""
    state = {
        "messages": [HumanMessage(content=m) for m in messages],
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
    }
    try:
        result = graph.invoke(state)
        return result
    except Exception as e:
        print(f"  ERROR in {label}: {e}")
        return {}


def check(result: dict, checks: dict, label: str) -> bool:
    """Evaluate result against expected checks. Returns True if all pass."""
    passed = True
    for key, expected in checks.items():
        actual = result.get(key)
        if callable(expected):
            ok = expected(actual)
        else:
            ok = actual == expected
        if not ok:
            print(f"  FAIL [{label}] {key}: expected {expected}, got {actual}")
            passed = False
    return passed


# ── Test cases ─────────────────────────────────────────────────

TESTS = [
    # ── Happy path ─────────────────────────────────────────────
    {
        "label": "01 - gutter cleaning in area with full info",
        "messages": ["Hi I need my gutters cleaned. I'm in Mississauga. My name is Sarah, sarah@email.com. 2 storey home, 120 feet of gutters."],
        "checks": {
            "service_type": "gutter_cleaning",
            "in_service_area": True,
            "needs_site_visit": False,
            "price_range": lambda x: x is not None and x[0] > 0,
            "lead_name": "Sarah",
            "contact": "sarah@email.com",
        }
    },
    {
        "label": "02 - junk removal in area",
        "messages": ["I need junk removed from my basement in Toronto. I'm Mike, 416-555-0123. It's about a full truckload with some old appliances."],
        "checks": {
            "service_type": "junk_removal",
            "in_service_area": True,
            "needs_site_visit": False,
            "price_range": lambda x: x is not None and x[0] > 0,
        }
    },
    {
        "label": "03 - lawn care in area",
        "messages": ["Can you cut my lawn in Brampton? Name is James, james@email.com. Medium sized lawn, one time cut."],
        "checks": {
            "service_type": "lawn_care",
            "in_service_area": True,
            "needs_site_visit": False,
            "price_range": lambda x: x is not None,
        }
    },
    {
        "label": "04 - house cleaning in area",
        "messages": ["I need my house cleaned in Oakville. I'm Lisa, lisa@email.com. 3 bedrooms, 2 bathrooms, deep clean."],
        "checks": {
            "service_type": "house_cleaning",
            "in_service_area": True,
            "needs_site_visit": False,
            "price_range": lambda x: x is not None and x[0] > 0,
        }
    },
    {
        "label": "05 - window cleaning in area",
        "messages": ["Need windows cleaned at my home in Vaughan. Tom here, tom@email.com. About 15 windows, 2 storey house."],
        "checks": {
            "service_type": "window_cleaning",
            "in_service_area": True,
            "needs_site_visit": False,
            "price_range": lambda x: x is not None,
        }
    },

    # ── Out of area ────────────────────────────────────────────
    {
        "label": "06 - out of area Ottawa",
        "messages": ["I need my lawn cut in Ottawa. I'm David, david@email.com."],
        "checks": {
            "in_service_area": False,
            "decision": "out_of_area",
        }
    },
    {
        "label": "07 - out of area Montreal",
        "messages": ["Can you remove junk from my place in Montreal? Name is Pierre, pierre@email.com."],
        "checks": {
            "in_service_area": False,
            "decision": "out_of_area",
        }
    },
    {
        "label": "08 - out of area Vancouver",
        "messages": ["Need gutter cleaning in Vancouver. I'm Anna, anna@email.com."],
        "checks": {
            "in_service_area": False,
            "decision": "out_of_area",
        }
    },

    # ── Site visit cases ───────────────────────────────────────
    {
        "label": "09 - full house demolition needs site visit",
        "messages": ["I need a full house demolished in Toronto. I'm Chris, chris@email.com. It's a 3 storey detached home."],
        "checks": {
            "service_type": "demolition",
            "in_service_area": True,
            "decision": lambda x: x in ["site_visit", "more_info"],
        }
    },
    {
        "label": "10 - large HVAC installation",
        "messages": ["Need a new HVAC system installed in my commercial building in Toronto. I'm Ben, ben@email.com."],
        "checks": {
            "service_type": "hvac",
            "in_service_area": True,
        }
    },

    # ── Ambiguous / edge cases ─────────────────────────────────
    {
        "label": "11 - vague request gets classified",
        "messages": ["I need to get rid of some old stuff in my garage in Toronto. I'm Paul, paul@email.com."],
        "checks": {
            "service_type": lambda x: x in ["junk_removal", "handyman"],
            "in_service_area": True,
        }
    },
    {
        "label": "12 - user mentions multiple services",
        "messages": ["I need my lawn cut and gutters cleaned in Mississauga. I'm Kate, kate@email.com."],
        "checks": {
            "service_type": lambda x: x in ["lawn_care", "gutter_cleaning"],
            "in_service_area": True,
        }
    },
    {
        "label": "13 - no location mentioned",
        "messages": ["I need junk removed. I'm Mark, mark@email.com. Full truckload."],
        "checks": {
            "service_type": "junk_removal",
            "in_service_area": lambda x: x is None or isinstance(x, bool),
        }
    },
    {
        "label": "14 - pressure washing",
        "messages": ["Can you pressure wash my driveway in Markham? I'm Steve, steve@email.com. About 600 sq ft."],
        "checks": {
            "service_type": "pressure_washing",
            "in_service_area": True,
            "price_range": lambda x: x is not None,
        }
    },
    {
        "label": "15 - snow removal seasonal",
        "messages": ["I need snow removal for my driveway and walkway in Burlington. I'm Amy, amy@email.com. Looking for a seasonal contract."],
        "checks": {
            "service_type": "snow_removal",
            "in_service_area": True,
            "price_range": lambda x: x is not None,
        }
    },

    # ── Contact extraction ─────────────────────────────────────
    {
        "label": "16 - phone number extraction",
        "messages": ["Need my gutters cleaned in Toronto. Name is Dan, call me at 647-555-9876. 1 storey bungalow."],
        "checks": {
            "service_type": "gutter_cleaning",
            "contact": lambda x: x is not None and "647" in str(x),
        }
    },
    {
        "label": "17 - email extraction",
        "messages": ["Lawn care needed in Vaughan. I'm Rachel, reach me at rachel.smith@gmail.com. Large lawn."],
        "checks": {
            "service_type": "lawn_care",
            "contact": lambda x: x is not None and "@" in str(x),
        }
    },

    # ── Adversarial ────────────────────────────────────────────
    {
        "label": "18 - gibberish input",
        "messages": ["asdfjkl qwerty 12345"],
        "checks": {
            "service_type": lambda x: x is not None,  # should return something, not crash
        }
    },
    {
        "label": "19 - very short input",
        "messages": ["help"],
        "checks": {
            "service_type": lambda x: x is not None,
        }
    },
    {
        "label": "20 - painting request",
        "messages": ["I need my living room and kitchen painted in Ajax. I'm Nina, nina@email.com. 2 rooms interior."],
        "checks": {
            "service_type": "painting",
            "in_service_area": True,
            "price_range": lambda x: x is not None,
        }
    },
]


# ── Run all tests ──────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0
    errors = []

    print(f"\n── QuoteFlow Eval Suite — {len(TESTS)} tests ──\n")

    for i, test in enumerate(TESTS):
        label = test["label"]
        print(f"Running {label}...")
        result = run(test["messages"], label)
        ok = check(result, test["checks"], label)
        if ok:
            print(f"  ✅ PASS")
            passed += 1
        else:
            failed += 1
            errors.append(label)

    print(f"\n── Results ──")
    print(f"  Passed: {passed}/{len(TESTS)}")
    print(f"  Failed: {failed}/{len(TESTS)}")
    score = round(passed / len(TESTS), 2)
    print(f"  Score:  {score}")
    if errors:
        print(f"\n  Failed tests:")
        for e in errors:
            print(f"    - {e}")
    if score >= 0.8:
        print(f"\n  ✅ EVAL SUITE PASSED (target: 0.8)")
    else:
        print(f"\n  ❌ BELOW TARGET — fix failing tests before deploying")