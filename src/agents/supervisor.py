"""
Supervisor Agent: Routes user queries to the correct specialized agent,
handles multi-intents, and maintains conversation memory dynamically.
"""

import json
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from src.agents.music_agent import build_music_agent
from src.agents.invoice_agent import build_invoice_agent
from src.utils.config import OPENAI_API_KEY
from src.memory.user_profile import user_profile  # dynamic dict

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
    _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    routing_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a supervisor agent. You always have access to the user profile and past conversation.\n"
         "User Profile: {profile}\n"
         "Conversation History: {history}\n\n"
         "Decide which specialist agent should handle the query.\n"
         "- Use 'music' for artist/track/genre queries.\n"
         "- Use 'invoice' for customer/invoice/employee queries.\n"
         "- If BOTH domains are present, return 'music, invoice'.\n"
         "- If the user asks about themselves (name, favorites, preferences), "
         "answer directly using the profile or memory.\n"
         "⚠️ Respond ONLY with one of these:\n"
         "  • 'music'\n"
         "  • 'invoice'\n"
         "  • 'music, invoice'\n"
         "  • or a direct natural-language answer if memory/profile is enough."),
        ("human", "{input}")
    ])

    music_agent = build_music_agent(memory=conversation_memory)
    invoice_agent = build_invoice_agent(memory=conversation_memory)

    def route(input_text: str):
        """Route query, use memory + profile, and return results."""

        # Save user query into memory
        conversation_memory.chat_memory.add_user_message(input_text)

        # Get conversation history
        history = "\n".join([
            f"{msg.type.upper()}: {msg.content}"
            for msg in conversation_memory.chat_memory.messages
        ])

        # Supervisor decides
        decision = _llm.invoke(
            routing_prompt.format_messages(
                input=input_text,
                history=history,
                profile=json.dumps(user_profile, indent=2)
            )
        )
        choice = decision.content.strip().lower()

        # -------------------------
        # Handle cases
        # -------------------------
        if choice == "music":
            result = music_agent.invoke({"input": input_text})
            output = result.get("output", str(result))

        elif choice == "invoice":
            result = invoice_agent.invoke({"input": input_text})
            output = result.get("output", str(result))

        elif "music" in choice and "invoice" in choice:
            # ✅ Multi-intent: split query into sub-queries
            music_query = f"Music-related part of user query: {input_text}"
            invoice_query = f"Invoice-related part of user query: {input_text}"

            music_result = music_agent.invoke({"input": music_query})
            invoice_result = invoice_agent.invoke({"input": invoice_query})

            output = (
                "🎵 Music Result:\n"
                f"{music_result.get('output', str(music_result))}\n\n"
                "📄 Invoice Result:\n"
                f"{invoice_result.get('output', str(invoice_result))}"
            )

        else:
            # Direct answer (from profile/memory)
            output = choice

        # -------------------------
        # Save & persist
        # -------------------------
        conversation_memory.chat_memory.add_ai_message(output)
        _persist_memory()

        return output

    return route


# -------------------------
# Persistence helper
# -------------------------
def _persist_memory():
    try:
        convo = conversation_memory.chat_memory.messages
        convo_text = [
            {"role": msg.type, "content": msg.content}
            for msg in convo
        ]

        state = {
            "profile": user_profile,
            "conversation": convo_text
        }
        MEMORY_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[Persist Error] {e}")


# -------------------------
# Demo
# -------------------------
if __name__ == "__main__":
    supervisor = build_supervisor_agent()

    print("\n--- Test 1: Profile Debug ---")
    print(f"Loaded profile for {user_profile.get('name')} ({user_profile.get('email')})")
    print(f"Favorite genres: {user_profile['favorites'].get('genres')}")
    print(f"Favorite artists: {user_profile['favorites'].get('artists')}")
    print(f"Preferences: {user_profile['preferences']}")
    print("-" * 50)

    print("\n--- Test 2: Music Query ---")
    print(supervisor("Recommend me some music for studying"))

    print("\n--- Test 3: Invoice Query ---")
    print(supervisor("Get the last 2 invoices for my account"))

    print("\n--- Test 4: Multi-Intent Query ---")
    print(supervisor("How much was my last purchase and give me 3 songs by Drake"))
