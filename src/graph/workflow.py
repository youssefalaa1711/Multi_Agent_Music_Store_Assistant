"""
LangGraph workflow:
verify_info -> (maybe human_input) -> load_memory -> supervisor -> create_memory
"""

from __future__ import annotations
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.graph.state_schema import GraphState
from src.graph.nodes.verification_node import verify_info
from src.graph.nodes.memory_node import load_memory, create_memory
from src.agents.supervisor import build_supervisor_agent
from src.graph.nodes.human_input_node import human_input_node
from src.agents import supervisor
from pathlib import Path


MEMORY_FILE = Path("memory.json")
if MEMORY_FILE.exists():
    MEMORY_FILE.unlink()

# -----------------------------
# Wrap Supervisor
# -----------------------------
def _supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    profile = state.get("profile", {})  # comes from verify_info
    supervisor = build_supervisor_agent()

    user_text = state.get("input", "") or ""
    output = supervisor(user_text, profile=profile)  # <-- pass profile!

    if isinstance(output, str) and "please provide" in output.lower():
        return {**state, "need_human": True, "output": output}

    return {**state, "output": output, "need_human": False}



# -----------------------------
# Build Graph
# -----------------------------
def build_workflow():
    graph = StateGraph(GraphState)

    # Define nodes
    graph.add_node("verify_info", verify_info)
    graph.add_node("human_input", human_input_node)
    graph.add_node("load_memory", load_memory)
    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("create_memory", create_memory)

    # Entry point = verify_info
    graph.set_entry_point("verify_info")

    # Flow: if info is missing → human loop
    graph.add_conditional_edges(
        "verify_info",
        lambda state: "need_human" if state.get("need_human") else "ok",
        {
            "need_human": "human_input",
            "ok": "load_memory",
        },
    )

    # Human input loops back to verify_info
    graph.add_edge("human_input", "verify_info")

    # Normal pipeline
    graph.add_edge("load_memory", "supervisor")

    # Supervisor may require human input or finish directly
    graph.add_conditional_edges(
        "supervisor",
        lambda state: "human_input" if state.get("need_human") else "create_memory",
    )

    graph.add_edge("create_memory", END)

    return graph.compile()


# -----------------------------
# Quick Manual Run
# -----------------------------
if __name__ == "__main__":
    app = build_workflow()

    print("🎵 Welcome to the Music Store Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        result = app.invoke({"input": user_input})
        print("Bot:", result.get("output", ""))