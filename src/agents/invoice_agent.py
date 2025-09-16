"""
Invoice Agent using Groq + LangChain tools.
"""

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from src.agents.base_agent import invoice_tools
from src.utils.config import GROQ_API_KEY


def build_invoice_agent():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",  # Groq’s recommended model for reasoning
        temperature=0
    )

    agent = initialize_agent(
        tools=invoice_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent


if __name__ == "__main__":
    agent = build_invoice_agent()
    print(agent.run("Get the last 2 invoices for customer 1"))
    print(agent.run("Which employee handled invoice 1 for customer 1?"))
