"""
Supervisor Agent: Routes user queries to the correct specialized agent,
handles multi-intents, and maintains conversation memory dynamically.
Also updates user profile dynamically (e.g. favorites, phone).
"""

import json
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from src.agents.music_agent import build_music_agent
from src.agents.invoice_agent import build_invoice_agent
from src.utils.config import OPENAI_API_KEY
from src.memory.user_profile import user_profile as static_profile  # static fallback

# -------------------------
# Global memory & persistence
# -------------------------
MEMORY_FILE = Path("memory.json")

conversation_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY),
    memory_key="chat_history",
    return_messages=True,
)
conversation_memory.chat_memory.clear()

# -------------------------
# Pending action memory
# -------------------------
pending_action = None


# -------------------------
# Supervisor Builder
# -------------------------
def build_supervisor_agent(profile: dict | None = None):
    """
    Build a supervisor agent that routes between music/invoice agents
    and dynamically enriches user profile with preferences + phone.
    """
    _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # ✅ Use dynamic profile if passed, otherwise fallback to static
    profile_data = profile or static_profile

    routing_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a supervisor agent. You always have access to the user profile and past conversation.\n\n"
            "User Profile:\n"
            "Name: {name}\n"
            "Email: {email}\n"
            "Customer ID: {customer_id}\n"
            "Phone: {phone}\n"
            "Favorites: {favorites}\n"
            "Preferences: {preferences}\n\n"
            "Conversation History:\n{history}\n\n"
            "Decide which specialist agent should handle the query:\n"
            "- Use 'music' for artist/track/genre queries.\n"
            "- Use 'invoice' for customer/invoice/employee queries.\n"
            "- If BOTH domains are present, return 'music, invoice'.\n"
            "- If the user asks about themselves (name, favorites, preferences, phone), "
            "answer directly using the profile or memory.\n"
            "⚠️ Respond ONLY with one of these:\n"
            "  • 'music'\n"
            "  • 'invoice'\n"
            "  • 'music, invoice'\n"
            "  • or a direct natural-language answer if memory/profile is enough."
        ),
        ("human", "{input}")
    ])

    music_agent = build_music_agent(memory=conversation_memory)
    invoice_agent = build_invoice_agent(memory=conversation_memory)

    def route(input_text: str, profile: dict | None = None):
        global pending_action
        p = profile or profile_data

        # --- Confirmation handler ---
        yes_words = {"yes", "yep", "yeah", "sure", "ok", "okay"}
        if input_text.strip().lower() in yes_words and pending_action:
            if pending_action["type"] == "songs_by_fav":
                artist = pending_action["artist"]
                input_text = f"List songs by {artist}"
                pending_action = None

        # Save user query into memory
        conversation_memory.chat_memory.add_user_message(input_text)

        # Build conversation history string
        history = "\n".join([
            f"{msg.type.upper()}: {msg.content}"
            for msg in conversation_memory.chat_memory.messages
        ])

        # Let LLM decide routing
        decision = _llm.invoke(
            routing_prompt.format_messages(
                input=input_text,
                history=history,
                name=p.get("name", "Unknown"),
                email=p.get("email", "Unknown"),
                customer_id=p.get("customer_id", "Unknown"),
                phone=p.get("phone", "Unknown"),
                favorites=json.dumps(p.get("favorites", {})),
                preferences=json.dumps(p.get("preferences", {})),
            )
        )
        choice = decision.content.strip().lower()

        # Routing logic
        if choice == "music":
            result = music_agent.invoke({"input": input_text})
            output = result.get("output", str(result))

        elif choice == "invoice":
            query = input_text
            if p.get("phone"):
                query = f"{query} phone {p['phone']}"
            result = invoice_agent.invoke({"input": query})
            output = result.get("output", str(result))

        elif "music" in choice and "invoice" in choice:
            splitter_prompt = (
                "The following user query contains both a music-related request "
                "and an invoice-related request.\n"
                f"Query: {input_text}\n\n"
                "Return JSON strictly in this format:\n"
                "{\n"
                "  \"music\": \"<music-only request>\",\n"
                "  \"invoice\": \"<invoice-only request>\"\n"
                "}"
            )

            split = _llm.invoke(splitter_prompt)
            try:
                parts = json.loads(split.content)
                music_query = parts.get("music", "").strip() or input_text
                invoice_query = parts.get("invoice", "").strip() or input_text
            except Exception:
                music_query = input_text
                invoice_query = input_text

            if p.get("phone"):
                invoice_query = f"{invoice_query} phone {p['phone']}"

            music_result = music_agent.invoke({"input": music_query})
            invoice_result = invoice_agent.invoke({"input": invoice_query})

            music_output = music_result.get("output", str(music_result))
            invoice_output = invoice_result.get("output", str(invoice_result))

            if any(word in invoice_output.lower() for word in ["album", "artist", "song", "track"]):
                invoice_output = "\n".join(
                    line for line in invoice_output.splitlines()
                    if not any(word in line.lower() for word in ["album", "artist", "song", "track"])
                ).strip()

            output = (
                "🎵 Music Result:\n"
                f"{music_output}\n\n"
                "📄 Invoice Result:\n"
                f"{invoice_output}"
            )
        else:
            # ✅ Only treat as direct answer if it's not one of the routing keywords
            output = decision.content.strip()

        # ✅ Update profile dynamically (favorites + phone)
        _update_profile_from_text(input_text, p)

        # ✅ If user asked about preferences and we have favorites → suggest + store pending action
        if "preference" in input_text.lower() and p.get("favorites", {}).get("artists"):
            fav_artist = p["favorites"]["artists"][0]
            output += f"\n\nWould you like to hear songs by {fav_artist}? (yes/no)"
            pending_action = {"type": "songs_by_fav", "artist": fav_artist}

        # Save to memory + persist
        conversation_memory.chat_memory.add_ai_message(output)
        _persist_memory(p)

        return output

    return route


# -------------------------
# Profile updater
# -------------------------
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _update_profile_from_text(user_text: str, profile: dict):
    """
    Dynamically enriches profile with phone numbers and music preferences.
    """
    import re
    t = user_text.lower()

    # --- Phone extraction ---
    phone_match = re.search(r"\+?\d[\d\-\(\)\s]+", user_text)
    if phone_match:
        profile["phone"] = phone_match.group().strip()

    # --- Artist/genre/song extraction via LLM ---
    extraction_prompt = f"""
    Extract the artist(s), genre(s), or song(s) mentioned in the text.
    Return JSON strictly in this format:
    {{
      "artists": [],
      "genres": [],
      "songs": []
    }}
    Text: "{user_text}"
    """

    try:
        result = _llm.invoke(extraction_prompt)
        data = {}
        import json
        data = json.loads(result.content)

        # Update profile safely
        if data.get("artists"):
            profile.setdefault("favorites", {}).setdefault("artists", [])
            for a in data["artists"]:
                if a not in profile["favorites"]["artists"]:
                    profile["favorites"]["artists"].append(a)

        if data.get("genres"):
            profile.setdefault("favorites", {}).setdefault("genres", [])
            for g in data["genres"]:
                if g not in profile["favorites"]["genres"]:
                    profile["favorites"]["genres"].append(g)

        if data.get("songs"):
            profile.setdefault("favorites", {}).setdefault("songs", [])
            for s in data["songs"]:
                if s not in profile["favorites"]["songs"]:
                    profile["favorites"]["songs"].append(s)

    except Exception as e:
        print(f"[Profile update error] {e}")


# -------------------------
# Persistence helper
# -------------------------
def _persist_memory(profile_data):
    try:
        convo = conversation_memory.chat_memory.messages
        convo_text = [
            {"role": msg.type, "content": msg.content}
            for msg in convo
        ]

        state = {
            "profile": profile_data,
            "conversation": convo_text
        }
        MEMORY_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[Persist Error] {e}")
