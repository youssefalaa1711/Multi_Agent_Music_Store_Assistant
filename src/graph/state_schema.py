"""
Graph state schema shared by all nodes.
"""

from typing import TypedDict, Optional, Dict, Any, List


class GraphState(TypedDict, total=False):
    # raw user message
    input: str

    # final response string from the supervisor/agents
    output: Optional[str]

    # long-term profile (static dict for now)
    profile: Dict[str, Any]

    # short-term memory payloads
    chat_summary: Optional[str]          # human-readable summary
    chat_history: Optional[List[Any]]    # optional, raw messages if you ever want them

    # verification / control flags
    need_human_input: bool               # if True, upstream should prompt user
    intent: Optional[str]                # "music" | "invoice" | None
