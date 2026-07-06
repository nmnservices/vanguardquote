"""
QuoteFlow MCP Server
Exposes QuoteFlow as an MCP tool that AI assistants can call directly.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph

graph = build_graph()
sessions = {}

# ── MCP Protocol helpers ───────────────────────────────────────

def mcp_response(id, result):
    return json.dumps({"jsonrpc": "2.0", "id": id, "result": result})

def mcp_error(id, code, message):
    return json.dumps({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})


# ── Tool definitions ───────────────────────────────────────────

TOOLS = [
    {
        "name": "get_quote",
        "description": (
            "Get an instant AI-generated price quote for a home service job. "
            "Supports gutter cleaning, junk removal, lawn care, house cleaning, "
            "window cleaning, pressure washing, snow removal, handyman, plumbing, "
            "HVAC, demolition, and painting. "
            "Provide the job description and location to get a price range."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Describe the job, location, and any relevant details (e.g. 'I need my gutters cleaned in Mississauga, 2 storey home, 120 feet of guttering')"
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID for multi-turn conversations"
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "get_similar_jobs",
        "description": (
            "Retrieve similar past jobs and their price ranges from the QuoteFlow database. "
            "Useful for understanding typical pricing for a service type in a given area."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "service_type": {
                    "type": "string",
                    "description": "Type of service (e.g. gutter_cleaning, junk_removal, lawn_care)"
                },
                "job_description": {
                    "type": "string",
                    "description": "Brief description of the job for similarity matching"
                }
            },
            "required": ["service_type"]
        }
    }
]


# ── Tool handlers ──────────────────────────────────────────────

def handle_get_quote(params: dict) -> str:
    """Run QuoteFlow agent and return structured quote result."""
    import uuid
    message = params.get("message", "")
    session_id = params.get("session_id") or str(uuid.uuid4())

    state = sessions.get(session_id)

    if state is None:
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

    state["messages"].append(HumanMessage(content=message))

    try:
        result = graph.invoke(state)
        sessions[session_id] = result

        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        reply = ai_messages[-1].content if ai_messages else "Unable to generate quote."

        price_range = result.get("price_range")
        output = {
            "session_id": session_id,
            "reply": reply,
            "service_type": result.get("service_type"),
            "in_service_area": result.get("in_service_area"),
            "needs_site_visit": result.get("needs_site_visit"),
            "price_range": {
                "min": price_range[0],
                "max": price_range[1],
                "currency": "CAD"
            } if price_range else None,
            "decision": result.get("decision"),
            "lead_quality": result.get("lead_object", {}).get("lead_quality") if result.get("lead_object") else None,
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def handle_get_similar_jobs(params: dict) -> str:
    """Retrieve similar past jobs from Supabase via RAG."""
    try:
        from agent.rag import retrieve_similar_leads, build_rag_context
        service_type = params.get("service_type", "")
        job_description = params.get("job_description", service_type)

        similar = retrieve_similar_leads(service_type, job_description)

        if not similar:
            return json.dumps({"message": "No similar past jobs found yet.", "leads": []})

        return json.dumps({
            "message": f"Found {len(similar)} similar past jobs",
            "leads": similar
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# ── MCP request handler ────────────────────────────────────────

def handle_request(request: dict) -> str:
    method = request.get("method")
    id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return mcp_response(id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "quoteflow-mcp",
                "version": "1.0.0",
                "description": "QuoteFlow AI quoting agent for home service businesses"
            }
        })

    elif method == "tools/list":
        return mcp_response(id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name")
        tool_input = params.get("arguments", {})

        if tool_name == "get_quote":
            result = handle_get_quote(tool_input)
            return mcp_response(id, {
                "content": [{"type": "text", "text": result}]
            })

        elif tool_name == "get_similar_jobs":
            result = handle_get_similar_jobs(tool_input)
            return mcp_response(id, {
                "content": [{"type": "text", "text": result}]
            })

        else:
            return mcp_error(id, -32601, f"Tool not found: {tool_name}")

    elif method == "notifications/initialized":
        return None

    else:
        return mcp_error(id, -32601, f"Method not found: {method}")


# ── Stdio transport (standard MCP) ────────────────────────────

def run_server():
    """Run MCP server over stdio — standard for Claude Desktop integration."""
    print("QuoteFlow MCP Server started", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request)
            if response:
                print(response, flush=True)
        except json.JSONDecodeError as e:
            error = mcp_error(None, -32700, f"Parse error: {e}")
            print(error, flush=True)
        except Exception as e:
            error = mcp_error(None, -32603, f"Internal error: {e}")
            print(error, flush=True)


# ── Quick test ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing MCP server tool definitions...")

    # Test initialize
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    print("\n── Initialize ──")
    print(handle_request(init_req))

    # Test tools/list
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    print("\n── Tools List ──")
    response = json.loads(handle_request(list_req))
    for tool in response["result"]["tools"]:
        print(f"  ✅ {tool['name']}: {tool['description'][:60]}...")

    # Test get_quote tool
    print("\n── Test get_quote tool ──")
    quote_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_quote",
            "arguments": {
                "message": "I need my gutters cleaned in Mississauga, 2 storey home, 120 feet"
            }
        }
    }
    response = json.loads(handle_request(quote_req))
    content = json.loads(response["result"]["content"][0]["text"])
    print(f"  Service: {content.get('service_type')}")
    print(f"  Price range: {content.get('price_range')}")
    print(f"  Decision: {content.get('decision')}")
    print("\n  ✅ MCP server working")