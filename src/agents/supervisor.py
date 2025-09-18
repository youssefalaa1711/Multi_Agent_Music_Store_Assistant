"""
Supervisor Agent: Routes user queries to the correct specialized agent,
uses predefined user_profile, and maintains conversation memory.
"""

import json
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from src.agents.music_agent import build_music_agent
from src.agents.invoice_agent import build_invoice_agent
from src.utils.config import OPENAI_API_KEY
from src.memory.user_profile import user_profile  # ✅ static profile dict

# -------------------------
# Global memory & persistence
# -------------------------

MEMORY_FILE = Path("memory.json")

conversation_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY),
    memory_key="chat_history",
    return_messages=True,
)


# -------------------------
# Supervisor Builder
# -------------------------
def build_supervisor_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    routing_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a supervisor. Decide which specialist agent should handle the query.\n"
         "Options: 'music' for artist/track/genre queries, 'invoice' for customer/invoice/employee queries.\n"
         "Respond ONLY with one word: 'music' or 'invoice'."),
        ("human", "{input}")
    ])

    music_agent = build_music_agent(memory=conversation_memory)
    invoice_agent = build_invoice_agent(memory=conversation_memory)

    def route(input_text: str):
        """Route query, use memory, and return results."""

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

        # Persist memory state + static user profile
        _persist_memory()

        return output

    return route


# -------------------------
# Persistence helper
# -------------------------
def _persist_memory():
    try:
        # Get conversation as plain strings
        convo = conversation_memory.chat_memory.messages
        convo_text = [
            {"role": msg.type, "content": msg.content}
            for msg in convo
        ]

        state = {
            "profile": user_profile,
            "conversation": convo_text
        }
        MEMORY_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        print(f"[Persist Error] {e}")

# -------------------------
# Demo
# -------------------------
if __name__ == "__main__":
    supervisor = build_supervisor_agent()

    print("\n--- Test 1: Profile Debug ---")
    from src.memory.user_profile import user_profile
    print(f"Loaded profile for {user_profile['name']} ({user_profile['email']})")
    print(f"Favorite genres: {user_profile['favorites']['genres']}")
    print(f"Favorite artists: {user_profile['favorites']['artists']}")
    print(f"Preferences: {user_profile['preferences']}")
    print("-" * 50)

    print("\n--- Test 2: Music Query (should lean on profile genres) ---")
    print(supervisor("Recommend me some music for studying"))

    print("\n--- Test 3: Invoice Query (profile customer_id=1 should be relevant) ---")
    print(supervisor("Get the last 2 invoices for my account"))
