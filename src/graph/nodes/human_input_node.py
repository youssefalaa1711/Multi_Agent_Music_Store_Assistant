"""
Human input node:
Pauses the workflow if key information is missing and asks the user.
This enables "human-in-the-loop" clarifications.
"""

from typing import Dict, Any


def human_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_query = state.get("input", "").lower()
    print("\n[HUMAN INPUT REQUIRED] The system needs clarification.\n")

    # Case 1: Missing customer_id for invoices
    if "invoice"  or "purchase"in user_query and "customer_id" not in user_query:
        customer_id = input("👉 Please provide your customer ID: ")
        state["input"] = f"{state['input']} customer_id {customer_id}"
        state["need_human"] = False
        return state

    # Case 2: Missing artist for music-related query
    if "album" in user_query or "track" in user_query or "song" in user_query:
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
