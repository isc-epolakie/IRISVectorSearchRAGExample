"""Interactive Business Service RAG Chat using OpenAI Responses API.

This script:
- Uses the OpenAI Responses API with function-calling (tools)
- Exposes your existing search_embeddings(search: str) function from Search.py as a tool
- Lets the model decide when to call the RAG tool to answer questions about business services
- Returns structured outputs (answer + optional sources) for each turn

Prerequisites:
- `OPENAI_API_KEY` set in environment (or loaded by your existing .env setup)
- Your IRIS / vector DB configured and working via Search.search_embeddings

Run:
    python business_service_chat.py
"""

import os
import json
from typing import List, Dict, Any

from openai import OpenAI

from Search import search_embeddings


# ---------------------------- Configuration ----------------------------

MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5-nano")
SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions about InterSystems "
    "business services and related integration capabilities. You have access "
    "to a vector database of documentation chunks about business services. "
    "\n\n"
    "Use the `search_business_docs` tool whenever the user asks about specific "
    "settings, configuration options, or how to perform tasks with business "
    "services. Ground your answers in the retrieved context, quoting or "
    "summarizing relevant chunks. If nothing relevant is found, say so "
    "clearly and answer from your general knowledge with a disclaimer."
)


# ---------------------------- Tool Definition ----------------------------

TOOLS = [
    {
        "type": "function",
        "name": "search_business_docs",
        "description": (
            "Searches a vector database of documentation chunks related to "
            "business services and returns the most relevant snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language search query describing what you want "
                        "to know about business services."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        "Maximum number of results to retrieve from the vector DB."
                    ),
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def call_rag_tool(name: str, args: Dict[str, Any]) -> str:
    """Route function calls from the model to our local Python implementations.

    Currently only supports the `search_business_docs` tool, which wraps
    `Search.search_embeddings`.

    The return value must be a string. We will JSON-encode a small structure
    so the model can consume the results reliably.
    """

    if name == "search_business_docs":
        query = args.get("query", "")
        top_k = args.get("top_k", "")

        results = search_embeddings(query, top_k)

        # Expecting each row to be something like (ID, chunk_text)
        formatted: List[Dict[str, Any]] = []
        for row in results:
            if not row:
                continue
            # Be defensive in case row length/structure changes
            doc_id = row[0] if len(row) > 0 else None
            text = row[1] if len(row) > 1 else None
            formatted.append({"id": doc_id, "text": text})

        payload = {"query": query, "results": formatted}
        return json.dumps(payload, ensure_ascii=False)

    # Unknown tool; return an error-style payload
    return json.dumps({"error": f"Unknown tool name: {name}"})


# ---------------------------- Chat Loop Logic ----------------------------


def extract_answer_and_sources(response: Any) -> Dict[str, Any]:
    """Extract a structured answer and optional sources from a Responses API object.

    We don't enforce a global JSON response schema here. Instead, we:
    - Prefer the SDK's `output_text` convenience when present
    - Fall back to concatenating any `output_text` content parts
    - Also surface any tool-call-output payloads we got back this turn as
      `tool_context` for debugging/inspection.
    """

    answer_text = ""

    # Preferred: SDK convenience
    if hasattr(response, "output_text") and response.output_text:
        answer_text = response.output_text
    else:
        # Fallback: walk output items
        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text":
                        parts.append(getattr(c, "text", ""))
        answer_text = "".join(parts)

    # Collect any function_call_output items for visibility
    tool_context: List[Dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call_output":
            try:
                tool_context.append({
                    "call_id": getattr(item, "call_id", None),
                    "output": json.loads(getattr(item, "output", "")),
                })
            except Exception:
                tool_context.append({
                    "call_id": getattr(item, "call_id", None),
                    "output": getattr(item, "output", ""),
                })

    return {"answer": answer_text.strip(), "tool_context": tool_context}



def chat_loop() -> None:
    """Run an interactive CLI chat loop using the OpenAI Responses API.

    The loop supports multi-step tool-calling:
    - First call may return one or more `function_call` items
    - We execute those locally (e.g., call search_embeddings)
    - We send the tool outputs back in a second `responses.create` call
    - Then we print the model's final, grounded answer
    """

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=key)

    print("\nBusiness Service RAG Chat")
    print("Type 'exit' or 'quit' to stop.\n")

    # Running list of inputs (messages + tool calls + tool outputs) for context
    input_items: List[Dict[str, Any]] = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # Add user message
        input_items.append({"role": "user", "content": user_input})

        # 1) First call: let the model decide whether to call tools
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            tools=TOOLS,
            input=input_items,
        )

        # Save model output items to our running conversation
        input_items += response.output

        # 2) Execute any function calls
        # The Responses API returns `function_call` items in `response.output`.
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue

            name = getattr(item, "name", None)
            raw_args = getattr(item, "arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"query": user_input}

            result_str = call_rag_tool(name, args or {})

            # Append tool result back as function_call_output
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": getattr(item, "call_id", None),
                    "output": result_str,
                }
            )

        # 3) Second call: ask the model to answer using tool outputs
        followup = client.responses.create(
            model=MODEL,
            instructions=(
                SYSTEM_INSTRUCTIONS
                + "\n\nYou have just received outputs from your tools. "
                + "Use them to give a concise, well-structured answer."
            ),
            tools=TOOLS,
            input=input_items,
        )

        structured = extract_answer_and_sources(followup)

        print("Agent:\n" + structured["answer"] + "\n")


if __name__ == "__main__":
    chat_loop()
