import logging

from celery import shared_task
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from search import search_engine
from brain import calculate_fixed_monthly_payment, calculate_future_value

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


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)
search_tool = FunctionTool.from_defaults(fn=search_engine)
calculate_fixed_monthly_payment_tool = FunctionTool.from_defaults(fn=calculate_fixed_monthly_payment)
calculate_future_value_tool = FunctionTool.from_defaults(fn=calculate_future_value)

llm = OpenAI(model="gpt-4o-mini")
ai_agent = ReActAgent.from_tools(
    [
        multiply_tool,
        add_tool,
        subtract_tool,
        divide_tool,
        calculate_fixed_monthly_payment_tool,
        calculate_future_value_tool,
        search_tool,
    ], llm=llm, verbose=True
)

@shared_task()
def ai_agent_handle(question):
    response = ai_agent.chat(question)
    logging.info(f"Agent response: {response}")
    return response.response
