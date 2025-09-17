"""
Supervisor Agent: Routes user queries to the correct specialized agent,
maintains conversation + user profile memory, and persists state.
"""

import json
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from src.agents.music_agent import build_music_agent
from src.agents.invoice_agent import build_invoice_agent
from src.utils.config import OPENAI_API_KEY
from src.memory.user_profile import user_profile

# -------------------------
# Global memory & persistence
# -------------------------

# Path to save memory state
MEMORY_FILE = Path("memory.json")

# Shared short-term conversational memory
conversation_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0 , api_key=OPENAI_API_KEY),
    memory_key="chat_history",
    return_messages=True
)

# Long-term user profile
user_profile = {
    "name": None,
    "favorite_genres": [],
    "favorite_artists": [],
    "preferences": {}
}

# Load previous state if available
if MEMORY_FILE.exists():
    try:
        state = json.loads(MEMORY_FILE.read_text())
        user_profile.update(state.get("profile", {}))
        # conversation memory will rebuild dynamically during session
    except Exception:
        pass


# -------------------------
# Supervisor Builder
# -------------------------
def build_supervisor_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Router prompt
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a supervisor. Decide which specialist agent should handle the query.\n"
         "Options: 'music' for artist/track/genre queries, 'invoice' for customer/invoice/employee queries.\n"
         "Respond ONLY with one word: 'music' or 'invoice'."),
        ("human", "{input}")
    ])

    # Attach shared memory to sub-agents
    music_agent = build_music_agent(memory=conversation_memory)
    invoice_agent = build_invoice_agent(memory=conversation_memory)

    def route(input_text: str):
        """Route query, use memory, update profile, and return results."""

        # Save user query into memory
        conversation_memory.chat_memory.add_user_message(input_text)

        # Decide which agent should handle it
        decision = llm.invoke(routing_prompt.format_messages(input=input_text))
        choice = decision.content.strip().lower()

        if "music" in choice:
            result = music_agent.invoke({"input": input_text})
            output = result.get("output", str(result))
        elif "invoice" in choice:
            result = invoice_agent.invoke({"input": input_text})
            output = result.get("output", str(result))
        else:
            output = "Sorry, I couldn’t decide which agent should handle this."

        # Save assistant response into memory
        conversation_memory.chat_memory.add_ai_message(output)

        # Update user profile from text
        _update_user_profile(input_text)

        # Persist state
        _persist_memory()

        return output

    return route


# -------------------------
# Profile updater (LLM-powered)
# -------------------------
def _update_user_profile(user_text: str):
    """Extract profile info from text using LLM and save into persistent user_profile."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    extraction_prompt = ChatPromptTemplate.from_template("""
    Extract user profile info from this text:
    "{text}"

    Return valid JSON with fields:
    - name (string or null)
    - favorite_genres (list of strings)
    - favorite_artists (list of strings)
    - preferences (dict)   # e.g., {"mood": "happy", "activity": "workout"}
    """)

    try:
        response = llm.invoke(extraction_prompt.format_messages(text=user_text))
        data = json.loads(response.content)

        # --- Save extracted data ---
        if data.get("name"):
            user_profile.set_name(data["name"])

        for genre in data.get("favorite_genres", []):
            user_profile.add_favorite_genre(genre)

        for artist in data.get("favorite_artists", []):
            user_profile.add_favorite_artist(artist)

        for key, value in data.get("preferences", {}).items():
            user_profile.set_preference(key, value)

    except Exception as e:
        print(f"[Profile Update Error] {e}")


# -------------------------
# Persistence helper
# -------------------------
def _persist_memory():
    try:
        state = {
            "profile": user_profile,
            "conversation": conversation_memory.load_memory_variables({})
        }
        MEMORY_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


# -------------------------
# Demo
# -------------------------
if __name__ == "__main__":
    supervisor = build_supervisor_agent()

    print(supervisor("Hello my name is Youssef. I like sad songs."))
    #print(supervisor("List albums by U2"))
    #print(supervisor("Get the last 2 invoices for customer 1"))

    print("\n--- Conversation Memory ---")
    print(conversation_memory.load_memory_variables({}))

    print("\n--- User Profile ---")
    #from src.memory.user_profile import user_profile
#
    #print("Name:", user_profile.data.get("name"))
    #print("Favorite Genres:", user_profile.get_favorites()["genres"])
    #print("Favorite Artists:", user_profile.get_favorites()["artists"])
    #print("Favorite Songs:", user_profile.get_favorites()["songs"])
    #print("Invoices:", user_profile.get_invoices())

