"""
Human input node:
Pauses the workflow ONLY when key information is missing and asks the user.
- Prompts for phone if the user asks about invoices/purchases and no phone is available.
- Auto-extracts phone numbers from the query if present.
- Never prompts for generic/hello queries (prevents recursion loops).
"""

from typing import Dict, Any
import re


PHONE_RE = re.compile(r"\+?\d[\d\-\(\)\s]{6,}")  # slightly stricter: at least a few digits


def _extract_phone(text: str) -> str | None:
    m = PHONE_RE.search(text)
    return m.group().strip() if m else None


def human_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Always start with "no more human needed" unless we actually prompt.
    state["need_human"] = False

    raw_query = state.get("input", "") or ""
    query_lc = raw_query.lower()

    # --- If the user is asking about invoices/purchases, ensure we have a phone ---
    mentions_invoice = ("invoice" in query_lc) or ("purchase" in query_lc)

    if mentions_invoice:
        # 1) If phone already stored in state, we're good.
        if state.get("phone"):
            # Make sure we don't keep appending "phone X" over and over.
            if f"phone {state['phone']}" not in raw_query:
                state["input"] = f"{raw_query} phone {state['phone']}"
            return state

        # 2) Try to auto-extract from the current query.
        extracted = _extract_phone(raw_query)
        if extracted:
            state["phone"] = extracted
            if f"phone {extracted}" not in raw_query:
                state["input"] = f"{raw_query} phone {extracted}"
            return state

        # 3) No phone anywhere -> ask once.
        print("\n[HUMAN INPUT REQUIRED] The system needs a phone number for billing queries.\n")
        phone = input("👉 Please provide your phone number : ").strip()
        state["phone"] = phone
        state["input"] = f"{raw_query} phone {phone}"
        # state["need_human"] already False so the graph proceeds next tick.
        return state

    # --- For all other queries (music, greetings, etc.) we don't block here ---
    # Do NOT prompt for generic clarification here; let supervisor/agents handle it.
    return state
