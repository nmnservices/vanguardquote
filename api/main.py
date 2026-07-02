from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph
from dotenv import load_dotenv
import uuid

load_dotenv()

app = FastAPI(title="QuoteFlow API")

# Allow widget to call from any domain
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


@app.get("/")
def root():
    return {"status": "QuoteFlow API running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Get or create session
    session_id = req.session_id or str(uuid.uuid4())
    state = sessions.get(session_id)

    if state is None:
        # New conversation
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

    # Add user message
    state["messages"].append(HumanMessage(content=req.message))

    # Run graph
    result = graph.invoke(state)

    # Save updated state
    sessions[session_id] = result

    # Get last AI message
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_messages[-1].content if ai_messages else "Sorry, something went wrong."

    # Build price range dict
    price_range = None
    if result.get("price_range"):
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