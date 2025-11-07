import json
import logging

from brain import chat_complete
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TextNode


def llm_split(text, metadata):
    messages = [
        {"role": "system", "content": "You are an amazing virtual assistant"},
        {
            "role": "user",
            "content": f"split the following text into 300 words each part, return only list of item: "
            f"## Input text ##: {text} \n"
            f'## Format ## : ["part 1", "part 2", ...] \n'
            f"## Output ##:",
        },
    ]
    llm_response = chat_complete(messages, model="deepseek-chat", raw=False)
    logging.info("LLM response: {}".format(llm_response))
    nodes = json.loads(llm_response)
    nodes = [TextNode(text=node, metadata=metadata) for node in nodes]
    return nodes


def split_document(text, metadata={"course": "LLM"}):
    meta_string = " ".join([f"{k}={v}" for k, v in metadata.items()])
    if len(text) < 200:
        return [TextNode(text=text, metadata=metadata)]
    elif len(text) < 1000:
        doc = Document(text=text, metadata=metadata)

        splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=10, separator=".")
        nodes = splitter.get_nodes_from_documents([doc])
    else:
        nodes = llm_split(text, metadata)
    return nodes
