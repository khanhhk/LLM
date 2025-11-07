"""
This module provides functionality to interact with OpenAI and DeepSeek LLMs,
handle function/tool calling (e.g. for financial calculations),
detect user intents, route queries, and generate embeddings.

It is designed to be integrated with a chatbot system that supports:
- Sync/async chat
- Function calling for financial tools
- Intent classification and question rephrasing
- Embedding generation for documents
"""
import json
import logging
import os

from functions import (calculate_fixed_monthly_payment, calculate_future_value,
                       get_tool_schema)
from openai import OpenAI
from redis import InvalidResponse

logger = logging.getLogger(__name__)

# Load API keys from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default=None)
DEEP_SEEK_API_KEY = os.environ.get("DEEP_SEEK_API_KEY")

# DeepSeek local deployment via Vast.ai
VAST_IP_ADDRESS = "64.139.209.4"
VAST_PORT = "29593"

# OpenAI / DeepSeek clients
deepseek_client = OpenAI(
    api_key="", base_url=f"http://{VAST_IP_ADDRESS}:{VAST_PORT}/v1"
)
deepseek_cloud_client = OpenAI(
    api_key=DEEP_SEEK_API_KEY, base_url="https://api.deepseek.com"
)


def get_openai_client():
    """Return OpenAI client using environment API key"""
    return OpenAI(api_key=OPENAI_API_KEY)

# Default OpenAI client
client = get_openai_client()

# -------------------------------
# Unified Chat Completion Handler
# -------------------------------

def chat_complete(messages=(), model="gpt-4o-mini", raw=False):
    """
    Main interface for multi-model chat completion (OpenAI / DeepSeek).
    
    Args:
        messages (list): List of chat messages
        model (str): Model name to use
        raw (bool): Whether to return full message dict or just content

    Returns:
        str | dict: Chat response
    """
    logger.info("Chat complete for {}".format(messages))
    if model == "deepseek-chat":
        called_client = deepseek_cloud_client
    elif model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
        called_client = deepseek_client
    else:
        called_client = client

    response = called_client.chat.completions.create(model=model, messages=messages)

    output = response.choices[0].message
    logger.info("Chat complete output: ".format(output))
    return output if raw else output.content


def openai_chat_complete(messages=(), model="gpt-4o-mini", raw=False):
    """Shortcut for OpenAI chat completion only"""
    response = client.chat.completions.create(model=model, messages=messages)
    output = response.choices[0].message
    logger.info("Chat complete output: ".format(output))
    return output if raw else output.content


def deepseek_chat_complete(messages=(), model="deepseek-chat", raw=False):
    """Shortcut for DeepSeek Cloud chat completion"""
    response = deepseek_cloud_client.chat.completions.create(
        model=model, messages=messages
    )
    output = response.choices[0].message
    logger.info("Chat complete output: ".format(output))
    return output if raw else output.content

# -------------------------------
# Embedding & Prompt Helpers
# -------------------------------

def get_embedding(text, model="text-embedding-3-small"):
    """Generate embedding for given text using OpenAI"""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def gen_doc_prompt(docs):
    """
    Format a list of document dicts into prompt string.

    Args:
        docs: List of documents with 'title' and 'content'

    Returns:
        Formatted string prompt
    """
    doc_prompt = ""
    for doc in docs:
        doc_prompt += f"Title: {doc['title']} \n Content: {doc['content']} \n"

    return "Document: \n + {}".format(doc_prompt)


def generate_conversation_text(conversations):
    conversation_text = ""
    for conversation in conversations:
        logger.info("Generate conversation: {}".format(conversation))
        role = conversation.get("role", "user")
        content = conversation.get("content", "")
        conversation_text += f"{role}: {content}\n"
    return conversation_text


def detect_user_intent(history, message):
    # Convert history to list messages
    history_messages = generate_conversation_text(history)
    logger.info(f"History messages: {history_messages}")
    # Update documents to prompt
    user_prompt = f"""
    Given following conversation and follow up question, rephrase the follow up question to a standalone question in the question's language.

    Chat History:
    {history_messages}

    Original Question: {message}

    Answer:
    """
    openai_messages = [
        {"role": "system", "content": "You are an amazing virtual assistant"},
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"Rephrase input messages: {openai_messages}")
    # call openai
    return openai_chat_complete(openai_messages)


def detect_route(history, message):
    logger.info(f"Detect route on history messages: {history}")
    # Update documents to prompt
    user_prompt = f"""
    Given the following chat history and the user's latest message, determine whether the user's intent is to ask for a faq, support ("faq") or finance, bank ("finance") or other ("other"). \n
    Provide only the classification label as your response.

    Chat History:
    {history}

    Latest User Message:
    {message}

    Classification (choose either "faq" or "finance" or "other"):
    """
    openai_messages = [
        {
            "role": "system",
            "content": "You are a highly intelligent assistant that helps classify customer queries",
        },
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"Route output: {openai_messages}")
    # call openai
    return openai_chat_complete(openai_messages)


available_tools = {
    "calculate_fixed_monthly_payment": calculate_fixed_monthly_payment,
    "calculate_future_value": calculate_future_value,
}


def get_financial_tools():
    tools = [
        get_tool_schema(calculate_fixed_monthly_payment),
        get_tool_schema(calculate_future_value),
    ]
    return tools


def get_financial_agent_answer(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()

    # Execute the chat completion request
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    # Attempt to extract response details
    if not resp.choices:
        logger.error("No choices available in the response.")
        return {
            "role": "assistant",
            "content": "An error occurred, please try again later.",
        }

    choice = resp.choices[0]
    return choice


def convert_tool_calls_to_json(tool_calls):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "arguments": json.dumps(call.function.arguments),
                    "name": call.function.name,
                },
            }
            for call in tool_calls
        ],
    }


def get_financial_agent_handle(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()
    choice = get_financial_agent_answer(messages, model, tools)

    resp_content = choice.message.content
    resp_tool_calls = choice.message.tool_calls
    # Prepare the assistant's message
    if resp_content:
        return resp_content

    elif resp_tool_calls:
        logger.info(f"Process the tools call: {resp_tool_calls}")
        # List to hold tool response messages
        tool_messages = []
        # Iterate through each tool call and execute the corresponding function
        for tool_call in resp_tool_calls:
            # Display the tool call details
            logger.info(
                f"Tool call: {tool_call.function.name}({tool_call.function.arguments})"
            )
            # Retrieve the tool function from available tools
            tool = available_tools[tool_call.function.name]
            # Parse the arguments for the tool function
            tool_args = json.loads(tool_call.function.arguments)
            # Execute the tool function and get the result
            result = tool(**tool_args)
            tool_args["result"] = result
            # Append the tool's response to the tool_messages list
            tool_messages.append(
                {
                    "role": "tool",  # Indicate this message is from a tool
                    "content": json.dumps(tool_args),  # The result of the tool function
                    "tool_call_id": tool_call.id,  # The ID of the tool call
                }
            )
        # Update the new message to get response from LLM
        # Append the tool messages to the existing messages
        # Check here: https://platform.openai.com/docs/guides/function-calling
        next_messages = (
            messages + [convert_tool_calls_to_json(resp_tool_calls)] + tool_messages
        )
        return get_financial_agent_handle(next_messages, model, tools)
    else:
        raise InvalidResponse(f"The response is invalid: {choice}")
