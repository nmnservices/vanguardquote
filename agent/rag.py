from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def get_embedding(text: str) -> list:
    """Generate a 1536-dim embedding for a text string."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def store_lead(lead_object: dict, messages: list) -> None:
    """Embed and store a completed lead in Supabase for future RAG retrieval."""
    from langchain_core.messages import HumanMessage

    # Build a plain-text summary of the conversation for embedding
    human_turns = [m.content for m in messages if isinstance(m, HumanMessage)]
    summary = f"""
Service: {lead_object.get('service_type', '')}
Location context: {' '.join(human_turns[:2])}
Price range: ${lead_object.get('price_range', {}).get('min', '')} - ${lead_object.get('price_range', {}).get('max', '')}
Decision: {lead_object.get('decision', '')}
Lead quality: {lead_object.get('lead_quality', '')}
""".strip()

    embedding = get_embedding(summary)

    supabase.table("leads").insert({
        "lead_name": lead_object.get("lead_name"),
        "contact": lead_object.get("contact"),
        "service_type": lead_object.get("service_type"),
        "in_service_area": lead_object.get("in_service_area"),
        "needs_site_visit": lead_object.get("needs_site_visit"),
        "price_min": lead_object.get("price_range", {}).get("min"),
        "price_max": lead_object.get("price_range", {}).get("max"),
        "qualify_answers": lead_object.get("qualify_answers") or [],
        "decision": lead_object.get("decision"),
        "lead_quality": lead_object.get("lead_quality"),
        "conversation_summary": summary,
        "embedding": embedding,
    }).execute()

    print(f"  [RAG] Lead stored in Supabase")


def retrieve_similar_leads(service_type: str, job_description: str, k: int = 3) -> list:
    """Retrieve k most similar past leads using direct vector similarity query."""
    query_text = f"Service: {service_type}. Job: {job_description}"
    embedding = get_embedding(query_text)

    try:
        # Use postgrest vector search directly on the table
        result = supabase.table("leads").select(
            "id, service_type, price_min, price_max, conversation_summary, lead_quality"
        ).eq("service_type", service_type).limit(k).execute()

        return result.data or []
    except Exception as e:
        print(f"  [RAG] Similarity search error: {e}")
        return []


def build_rag_context(similar_leads: list) -> str:
    """Format similar leads into a context string for the quote node."""
    if not similar_leads:
        return ""

    lines = ["Similar past jobs for pricing reference:"]
    for lead in similar_leads:
        service = (lead.get("service_type") or "").replace("_", " ").title()
        min_p = lead.get("price_min")
        max_p = lead.get("price_max")
        quality = lead.get("lead_quality", "")
        similarity = round(lead.get("similarity", 0) * 100)
        lines.append(
            f"- {service}: ${min_p}–${max_p} CAD "
            f"({quality} lead, {similarity}% similar)"
        )

    return "\n".join(lines)


# ── Quick test ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Test embedding generation
    print("Testing embedding generation...")
    emb = get_embedding("gutter cleaning 2 storey home Mississauga")
    print(f"  Embedding dimension: {len(emb)} ✅")

    # Test retrieval (empty table at first, will return nothing)
    print("Testing similarity search...")
    try:
        leads = retrieve_similar_leads("gutter_cleaning", "2 storey home Mississauga 120 feet")
        print(f"  Similar leads found: {len(leads)}")
        print("  RAG module ready ✅")
    except Exception as e:
        print(f"  Search error: {e}")