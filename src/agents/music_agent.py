"""
Music Agent: Handles artist, track, and genre queries using music_tools.
"""

from typing import Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from src.agents.base_agent import music_tools
from src.utils.config import OPENAI_API_KEY

def build_music_agent(memory: Optional[object] = None):
    """
    Build a music agent that uses tools to query the Chinook DB.
    Accepts optional memory (ConversationSummaryMemory) from supervisor.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # Strong system prompt → ensures tool use
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a music assistant for a digital music store. "
         "You MUST use the provided tools to answer questions. "
         "Do not guess. "
         "Return the tool’s raw JSON output directly."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(
        llm=llm,
        tools=music_tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=music_tools,
        verbose=True,
        memory=memory,   # attach shared memory if provided
        handle_parsing_errors=True,
    )

    return executor


if __name__ == "__main__":
    # Standalone test
    agent = build_music_agent()
    print(agent.invoke({"input": "List albums by U2"}))
    print(agent.invoke({"input": "List tracks by U2"}))
    print(agent.invoke({"input": "Songs in Rock genre"}))
    print(agent.invoke({"input": "Does song 'Let There Be Rock' exist?"}))
