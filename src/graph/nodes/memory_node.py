"""
memory_node:
- load_memory: loads conversation summary + profile into the state
- create_memory: persists summary + profile to disk (JSON-serializable only)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from src.memory.user_profile import user_profile
from src.agents.supervisor import conversation_memory  # reuse the same memory as your agents

MEMORY_FILE = Path("memory.json")


def load_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds chat summary + profile to the state (for downstream nodes/prompts).
    NOTE: conversation_memory.buffer is a plain string summary (JSON-safe).
    """
    summary_text = getattr(conversation_memory, "buffer", "") or ""
    return {
        **state,
        "chat_summary": summary_text,
        "profile": user_profile  # static profile dict
    }


def create_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persists the *serializable* memory snapshot after the run:
    - profile (dict)
    - chat_summary (string)
    - last output
    """
    snapshot = {
        "profile": state.get("profile", user_profile),
        "chat_summary": state.get("chat_summary", getattr(conversation_memory, "buffer", "")),
        "last_output": state.get("output", "")
    }
    try:
        MEMORY_FILE.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[Persist Error] {e}")

    return state
