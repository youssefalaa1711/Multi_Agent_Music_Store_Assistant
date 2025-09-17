"""
Invoice Agent: Handles invoice, customer, and employee queries using invoice_tools.
"""

from typing import Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from src.agents.base_agent import invoice_tools
from src.utils.config import OPENAI_API_KEY

def build_invoice_agent(memory: Optional[object] = None):
    """
    Build an invoice agent that uses tools to query the Chinook DB.
    Accepts optional memory (ConversationSummaryMemory) from supervisor.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # Strong system prompt → forces tool usage
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an invoice assistant for a digital music store. "
         "You MUST use the provided tools to answer questions. "
         "Do not guess or summarize. "
         "Return the tool’s raw JSON output directly."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(
        llm=llm,
        tools=invoice_tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=invoice_tools,
        verbose=True,
        memory=memory,   # attach shared memory if provided
        handle_parsing_errors=True,
    )

    return executor


if __name__ == "__main__":
    # Standalone test
    agent = build_invoice_agent()
    print(agent.invoke({"input": "Get the last 2 invoices for customer 1"}))
    print(agent.invoke({"input": "Get invoices sorted by unit price for customer 1"}))
    print(agent.invoke({"input": "Which employee handled invoice 1 for customer 1?"}))
