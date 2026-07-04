# QuoteFlow — AI Lead Qualification & Quoting Agent

An AI-powered conversational agent that qualifies inbound leads and generates instant price quotes for home service businesses. Built with LangGraph, FastAPI, and Claude (Anthropic).

**Live demo:** [quoteflow-agent-jet.vercel.app](https://quoteflow-agent-jet.vercel.app)
**API:** [web-production-0d92f.up.railway.app](https://web-production-0d92f.up.railway.app)

---

## What It Does

Home service businesses (junk removal, gutter cleaning, lawn care, demolition, HVAC, etc.) lose leads every day because they can't respond fast enough. QuoteFlow fixes that — it engages inbound leads instantly, qualifies them with service-specific questions, generates a price range from a configurable pricing model, and delivers a structured lead object to the business owner via webhook, email, or Slack.

**The full flow:**
1. Customer sends a message describing their job
2. Agent greets them and collects name + contact
3. Router classifies the service type and checks service area
4. Agent asks config-driven qualifying questions
5. Pricing engine applies business-owner-defined rules to generate a price range
6. Agent presents the quote and captures the customer's decision
7. Structured lead object is delivered to the business owner

---

## Architecture

Built as a LangGraph stateful agent with 6 nodes, each developed and committed as a separate PR:

```
agent/
├── intake.py          PR1 — greet lead, extract name + contact
├── router.py          PR2 — classify service type, check area, flag site visit
├── qualify.py         PR3 — ask config-driven qualifying questions
├── generate_quote.py  PR4 — apply pricing rules, produce price range
├── decision.py        PR5 — present quote, capture customer response
├── handoff.py         PR6 — package lead object, deliver to business owner
└── graph.py           — wire all nodes into compiled LangGraph graph
```

### State schema

```python
class QuoteFlowState(TypedDict):
    messages: list
    lead_name: Optional[str]
    contact: Optional[str]
    service_type: Optional[str]
    in_service_area: Optional[bool]
    needs_site_visit: Optional[bool]
    qualify_answers: Optional[dict]
    price_range: Optional[tuple]
    decision: Optional[str]        # accepted | more_info | escalate | out_of_area | site_visit
    lead_object: Optional[dict]
    memory: Optional[dict]
    reclassified: Optional[bool]
```

### Routing logic

```
intake → router → [out_of_area | site_visit | qualify] → generate_quote → decision → handoff
```

Conditional edges handle three router paths:
- **Out of area** — politely declines, logs lead as cold
- **Needs site visit** — offers free estimate, skips quote calculation
- **Can quote** — runs full qualify → generate_quote → decision flow

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 1.2 |
| LLM | Claude Sonnet (Anthropic) |
| LLM orchestration | LangChain Anthropic |
| API | FastAPI + Uvicorn |
| Database | Supabase (PostgreSQL) |
| Observability | LangSmith |
| Backend deployment | Railway |
| Frontend deployment | Vercel |
| Language | Python 3.11+ |

---

## Eval Suite

20 test cases covering happy path, out-of-area, site visit, ambiguous input, adversarial, and contact extraction scenarios.

```
── QuoteFlow Eval Suite — 20 tests ──

Passed: 20/20
Score:  1.0

✅ EVAL SUITE PASSED (target: 0.8)
```

Run evals locally:
```bash
python3 -m evals.test_graph
```

---

## Pricing Config

Each business owner configures their own pricing rules. The agent applies them dynamically — no hardcoded prices. Example config for gutter cleaning:

```python
"gutter_cleaning": {
    "base": 150,
    "modifiers": {
        "storeys_2": 75,
        "storeys_3plus": 150,
        "large_footage": 100,
    },
    "margin": 0.15   # ±15% range
}
```

The agent maps qualifying answers to modifiers, runs the calculation, and returns a price range (e.g. $276–$374) — never a single number, always a range to account for on-site variables.

---

## Supported Services

Gutter cleaning · Lawn care · Junk removal · House cleaning · Window cleaning · Pressure washing · Snow removal · Handyman · Plumbing · HVAC · Demolition · Painting

---

## Local Setup

```bash
# Clone
git clone https://github.com/nmnservices/quoteflow-agent.git
cd quoteflow-agent

# Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your API keys to .env

# Run API
uvicorn api.main:app --reload --port 8000

# Open widget
open widget/index.html
```

### Environment variables

```
ANTHROPIC_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=quoteflow-agent
SUPABASE_URL=
SUPABASE_KEY=
```

---

## API

**POST** `/chat`

```json
{
  "session_id": "optional-uuid",
  "message": "I need my gutters cleaned in Mississauga"
}
```

**Response:**
```json
{
  "session_id": "e4f19608-...",
  "reply": "Hi Sarah! Here's your quote for Gutter Cleaning...",
  "lead_captured": true,
  "service_type": "gutter_cleaning",
  "price_range": { "min": 191, "max": 259, "currency": "CAD" },
  "decision": "accepted"
}
```

---

## Roadmap

- [ ] Multi-turn qualify loop (LangGraph interrupt/resume)
- [ ] Supabase session persistence (replace in-memory store)
- [ ] Webhook delivery to Jobber / Housecall Pro
- [ ] Slack / email lead notifications
- [ ] Business owner config dashboard
- [ ] Multi-service quotes in a single conversation
- [ ] SMS and Instagram DM channels

---

## Author

Built by **Mark Nwulu** as a production AI engineering portfolio project demonstrating end-to-end agentic AI system design — from architecture and stateful graph construction to deployment, observability, and eval-driven quality assurance.

- GitHub: [github.com/nmnservices](https://github.com/nmnservices)
- Stack: LangGraph · FastAPI · Anthropic Claude · Supabase · Railway · Vercel
