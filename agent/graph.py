from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from agent.intake import VanguardQuoteState, intake_node
from agent.router import router_node
from agent.qualify import qualify_node
from agent.generate_quote import generate_quote_node
from agent.decision import decision_node
from agent.handoff import handoff_node
from dotenv import load_dotenv

load_dotenv()


# ── Wrapped nodes with progress output ────────────────────────

def intake_with_log(state):
    print("  [1/6] intake...")
    return intake_node(state)

def router_with_log(state):
    print("  [2/6] router...")
    return router_node(state)

def qualify_with_log(state):
    print("  [3/6] qualify...")
    return qualify_node(state)

def generate_quote_with_log(state):
    print("  [4/6] generate_quote...")
    return generate_quote_node(state)

def decision_with_log(state):
    print("  [5/6] decision...")
    return decision_node(state)

def handoff_with_log(state):
    print("  [6/6] handoff...")
    return handoff_node(state)


# ── Conditional edges ──────────────────────────────────────────

def route_after_router(state: VanguardQuoteState) -> str:
    if state.get("in_service_area") is False:
        return "out_of_area"
    if state.get("needs_site_visit"):
        return "site_visit"
    return "qualify"


# ── Out of area handler ────────────────────────────────────────

def out_of_area_node(state: VanguardQuoteState) -> VanguardQuoteState:
    print("  [3/6] out_of_area...")
    msg = (
        "Thank you for reaching out! Unfortunately we don't currently "
        "service your area. We hope to expand soon!"
    )
    return {
        **state,
        "messages": state.get("messages", []) + [AIMessage(content=msg)],
        "decision": "out_of_area",
    }


# ── Site visit handler ─────────────────────────────────────────

def site_visit_node(state: VanguardQuoteState) -> VanguardQuoteState:
    print("  [3/6] site_visit...")
    msg = (
        "Based on what you've described, this job needs a free on-site "
        "estimate. Our site visits are completely free — would you like "
        "to book one? Share a couple of times that work for you! 🗓️"
    )
    return {
        **state,
        "messages": state.get("messages", []) + [AIMessage(content=msg)],
        "decision": "site_visit",
    }


# ── Build graph ────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(VanguardQuoteState)

    graph.add_node("intake", intake_with_log)
    graph.add_node("router", router_with_log)
    graph.add_node("out_of_area", out_of_area_node)
    graph.add_node("site_visit", site_visit_node)
    graph.add_node("qualify", qualify_with_log)
    graph.add_node("generate_quote", generate_quote_with_log)
    graph.add_node("decision", decision_with_log)
    graph.add_node("handoff", handoff_with_log)

    graph.set_entry_point("intake")

    graph.add_edge("intake", "router")
    graph.add_conditional_edges("router", route_after_router, {
        "out_of_area": "out_of_area",
        "site_visit": "site_visit",
        "qualify": "qualify",
    })
    graph.add_edge("out_of_area", "handoff")
    graph.add_edge("site_visit", "handoff")
    graph.add_edge("qualify", "generate_quote")
    graph.add_edge("generate_quote", "decision")
    graph.add_edge("decision", "handoff")  # always go to handoff, no loop
    graph.add_edge("handoff", END)

    return graph.compile()


# ── Full end-to-end test ───────────────────────────────────────

if __name__ == "__main__":
    graph = build_graph()

    initial_state = {
        "messages": [
            HumanMessage(content=(
                "Hi, I need my gutters cleaned. "
                "I'm in Mississauga. My name is Sarah and "
                "you can reach me at sarah@email.com. "
                "It's a 2 storey home with about 120 feet of gutters."
            ))
        ],
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

    print("\n── Running full VanguardQuote graph ──\n")
    result = graph.invoke(initial_state)

    print("\n── Final agent message ──")
    print(result["messages"][-1].content)
    print(f"\n── Final state ──")
    print(f"  Service:      {result['service_type']}")
    print(f"  In area:      {result['in_service_area']}")
    print(f"  Price range:  {result['price_range']}")
    print(f"  Decision:     {result['decision']}")
    quality = result['lead_object']['lead_quality'].upper() if result.get('lead_object') else 'N/A'
    print(f"  Lead quality: {quality}")