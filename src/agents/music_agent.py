from typing import Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from src.agents.base_agent import music_tools
from src.utils.config import OPENAI_API_KEY
from src.memory.user_profile import user_profile

def build_music_agent(memory: Optional[object] = None):
    """
    Build a music agent that uses tools to query the Chinook DB.
    Handles fuzzy genre mapping + user preferences via music_tools.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a music assistant for a digital music store.\n"
         "You MUST use the provided tools to answer questions.\n"
         "Genres that aren’t exact should still be handled (tools do the mapping).\n"
         "You may also recommend songs/artists from the user profile if relevant.\n"
         "Always return results in a clean, readable format."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(
        llm=llm,
        tools=music_tools,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=music_tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
