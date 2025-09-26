"""
Human input node:
Pauses the workflow if key information is missing and asks the user.
This enables "human-in-the-loop" clarifications.
"""

from typing import Dict, Any
import re


def human_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = state.get("input", "").lower()
    print("\n[HUMAN INPUT REQUIRED] The system needs clarification.\n")

    # --- Helper: phone extractor ---
    def extract_phone(text: str):
        match = re.search(r"\+?\d[\d\-\(\)\s]+", text)
        return match.group().strip() if match else None

    # Case 1: Missing phone number for invoices / purchases
    if "invoice" in user_query or "purchase" in user_query:
        if "phone" not in state:
            # First, try to auto-extract from the query
            phone = extract_phone(user_query)
            if phone:
                state["phone"] = phone
            else:
                phone = input("👉 Please provide your phone number: ")
                state["phone"] = phone
        else:
            phone = state["phone"]  # ✅ reuse saved phone

        # Attach phone to the query so downstream agents see it
        state["input"] = f"{state['input']} phone {state['phone']}"
        state["need_human"] = False
        return state

    # Case 2: Missing artist for music-related query
    if any(word in user_query for word in ["album", "track", "song"]):
        if "by" not in user_query and "artist" not in user_query:
            artist = input("👉 Which artist are you interested in? ")
            state["input"] = f"{state['input']} by {artist}"
            state["need_human"] = False
            return state

    # Case 3: Missing genre
    if "genre" in user_query or "recommend" in user_query:
        if not any(g in user_query for g in ["rock", "pop", "jazz", "blues"]):
            genre = input("👉 What genre would you like? ")
            state["input"] = f"{state['input']} genre {genre}"
            state["need_human"] = False
            return state

    # Fallback: generic clarification
    clarification = input("👉 Could you clarify your request? ")
    state["input"] = clarification
    state["need_human"] = False
    return state
