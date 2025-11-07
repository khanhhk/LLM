"""
This module defines a Celery task that utilizes a ReAct-style LLM agent (LlamaIndex + OpenAI)
to answer user questions by dynamically invoking tools for basic arithmetic, financial calculations,
and search capabilities. The agent chooses which tools to use in order to reason step-by-step.
"""

import logging

from brain import calculate_fixed_monthly_payment, calculate_future_value
from celery import shared_task
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from search import search_engine


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> float:
    """Divide two integers and returns the result float"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a * 1.0 / b

# ---------------------
# Convert Functions to Tools for ReActAgent
# ---------------------

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)
search_tool = FunctionTool.from_defaults(fn=search_engine)
calculate_fixed_monthly_payment_tool = FunctionTool.from_defaults(
    fn=calculate_fixed_monthly_payment
)
calculate_future_value_tool = FunctionTool.from_defaults(fn=calculate_future_value)

# ---------------------
# Initialize the Language Model Agent
# ---------------------

llm = OpenAI(model="gpt-4o-mini")

# Create the agent with all available tools
ai_agent = ReActAgent.from_tools(
    [
        multiply_tool,
        add_tool,
        subtract_tool,
        divide_tool,
        calculate_fixed_monthly_payment_tool,
        calculate_future_value_tool,
        search_tool,
    ],
    llm=llm,
    verbose=True,
)

# ---------------------
# Celery Task Definition
# ---------------------

@shared_task()
def ai_agent_handle(question: str) -> str:
    """Celery task to handle a question using the ReActAgent.

    Args:
        question (str): User's natural language question

    Returns:
        str: Final response from the agent after reasoning
    """
    response = ai_agent.chat(question)
    logging.info(f"Agent response: {response}")
    return response.response
