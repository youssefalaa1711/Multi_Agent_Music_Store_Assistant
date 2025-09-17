"""
Music Agent: Handles artist, track, and genre queries using music_tools.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from src.agents.base_agent import music_tools
from src.utils.config import OPENAI_API_KEY


def build_music_agent():
    # ✅ Use GPT-4o-mini for reliable tool calling
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    # Strong system prompt to force tool usage
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a music assistant for a digital music store. "
         "You MUST use the provided tools to answer questions. "
         "Do not guess, do not format as markdown or tables. "
         "Always return the tool’s raw JSON output directly."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # Create the function-calling agent
    agent = create_openai_functions_agent(
        llm=llm,
        tools=music_tools,
        prompt=prompt,
    )

    # Wrap into executor
    return AgentExecutor(
        agent=agent,
        tools=music_tools,
        verbose=True,
        return_intermediate_steps=False,
    )


if __name__ == "__main__":
    agent = build_music_agent()

    # Demo queries
    print(agent.invoke({"input": "List albums by U2"}))
    print(agent.invoke({"input": "List tracks by U2"}))
    #print(agent.invoke({"input": "Songs in Rock genre"}))
    print(agent.invoke({"input": "Does song 'Let There Be Rock' exist?"}))
