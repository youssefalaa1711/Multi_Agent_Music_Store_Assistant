"""
Music Agent using Groq + LangChain tools.
"""

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from src.agents.base_agent import music_tools
from src.utils.config import GROQ_API_KEY


def build_music_agent():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0
    )
    
    agent = initialize_agent(
        tools=music_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent


if __name__ == "__main__":
    agent = build_music_agent()
    print(agent.run("List albums by U2"))
    print(agent.run("Give me some Rock songs"))
