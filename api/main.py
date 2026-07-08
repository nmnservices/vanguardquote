from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.graph import build_graph
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

app = FastAPI(title="VanguardQuote API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

graph = build_graph()

# In-memory session store
# In production this moves to Supabase/Redis
sessions = {}

_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ── Conversational follow-up handler ───────────────────────────

def _conversational_reply(user_message: str, state: dict) -> dict:
    """Handle follow-up messages without re-running the full graph."""

    # Escalation signals — skip LLM, respond directly
    escalate_signals = [
        "speak with", "talk to", "call me", "phone", "human",
        "someone", "agent", "representative", "person"
    ]
    if any(s in user_message.lower() for s in escalate_signals):
        reply = "No problem — I'll have someone from the team reach out to you shortly. Is the best way to reach you at the contact info you provided?"
        state["messages"].append(AIMessage(content=reply))
        state["decision"] = "escalate"
        return state

    # Build context for Claude
    name = state.get("lead_name") or "there"
    service = (state.get("service_type") or "").replace("_", " ")
    price = state.get("price_range")
    price_str = f"${price[0]}–${price[1]} CAD" if price else "not yet calculated"

    system = f"""You are a concise assistant for a home service business.

Customer context:
- Name: {name}
- Service: {service}
- Quote: {price_str}
- Status: {state.get('decision')}

Rules — follow these strictly:
- Reply in 1-3 sentences maximum
- Never repeat the quote unless they specifically ask for it
- Never use their name more than once per conversation
- No bullet points, no headers, no markdown formatting
- Answer only what they asked — nothing more
- If they mention a new service, ask one clarifying question about it
- If they want to book, ask for their preferred date and time only"""

    human_context = " | ".join(
        m.content for m in state["messages"]
        if isinstance(m, HumanMessage)
    )

    response = _llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Conversation so far: {human_context}\n\nLatest message: {user_message}")
    ])

    state["messages"].append(AIMessage(content=response.content))
    return state


# ── Models ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    lead_captured: bool
    service_type: str | None
    price_range: dict | None
    decision: str | None


# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "VanguardQuote API running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    state = sessions.get(session_id)

    if state is None:
        # New conversation — run full graph
        state = {
            "messages": [],
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
        state["messages"].append(HumanMessage(content=req.message))
        result = graph.invoke(state)
    else:
        # Existing conversation — conversational reply only
        state["messages"].append(HumanMessage(content=req.message))
        result = _conversational_reply(req.message, state)

    sessions[session_id] = result

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_messages[-1].content if ai_messages else "Sorry, something went wrong."

    price_range = None
    if result.get("price_range") and not result.get("needs_site_visit"):
        price_range = {
            "min": result["price_range"][0],
            "max": result["price_range"][1],
            "currency": "CAD"
        }

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        lead_captured=result.get("lead_object") is not None,
        service_type=result.get("service_type"),
        price_range=price_range,
        decision=result.get("decision"),
    )