"""
verify_info node:
- Verifies the account context (uses static user_profile for demo)
- Performs a *light* intent classification (music vs invoice) from the input
- Sets need_human_input flag if critical info is missing
"""

from __future__ import annotations
from typing import Dict, Any
from src.memory.user_profile import user_profile


def _classify_intent(text: str) -> str | None:
    t = text.lower()

    music_keywords = [
        "song", "songs", "track", "tracks", "album", "albums",
        "artist", "artists", "genre", "genres", "recommend", "music"
    ]
    invoice_keywords = [
        "invoice", "invoices", "purchase", "purchases", "order",
        "receipt", "support rep", "employee", "last purchase", "most recent"
    ]

    if any(k in t for k in invoice_keywords) and not any(k in t for k in music_keywords):
        return "invoice"
    if any(k in t for k in music_keywords) and not any(k in t for k in invoice_keywords):
        return "music"
    # mixed or unclear – let supervisor decide later
    return None


def verify_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
      state["input"]: user utterance
    Output (added/updated keys):
      - profile
      - intent (optional)
      - need_human_input (bool)
    """
    text = state.get("input", "") or ""
    intent = _classify_intent(text)

    # For demo: we treat user_profile as the verified account.
    # If you later require an explicit credential step, set need_human_input=True until provided.
    has_customer = bool(user_profile.get("customer_id"))

    return {
        **state,
        "profile": user_profile,
        "intent": intent,
        "need_human_input": not has_customer  # if no customer_id, pause for human input
    }
