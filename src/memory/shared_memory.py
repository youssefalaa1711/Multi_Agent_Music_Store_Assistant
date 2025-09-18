"""
Shared memory module for all agents.
Provides a single ConversationSummaryBufferMemory instance
that music_agent and invoice_agent can both use.
"""

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

# Use the same LLM for summarizing memory
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Shared memory instance
shared_memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=1000,
)
