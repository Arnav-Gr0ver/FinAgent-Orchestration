from langgraph_supervisor import create_supervisor
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

from data_agent_pool import *

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)

if __name__ == "__main__":

    supervisor = create_supervisor(
    agents=[],
    model=chat_model,
    prompt=(
        '''
        You are the supervisor of an end-to-end automated trading system. You manage a pool of Data Agents (Responsible for ingesting, validating,
        and transforming raw input data), a pool of Alpha Agents (Dedicated to predictive signal generation using statistical or model-based techniques), and a pool of Executation Agents (Handle order placement, routing, and market interface). 

        Assign work to them sequentially given user instructions.
        '''
    )
    ).compile()

    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "ADD TEST MESSAGE HERE"
                }
            ]
        }
    ):
        print(chunk)
        print("\n")