"""Main entry point for testing agent pools."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# This ensures that the 'test_client' directory is treated as the root,
# making absolute imports like 'from base import ...' possible.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Updated import path for the DataAgentPool
from test_client.agent_pools.data_agent_pool.agent_pool import DataAgentPool

# Load environment variables from a .env file
load_dotenv()


async def main():
    """Main function to run the agent pool demonstration."""
    print("Setting up the Data Agent Pool with a live OpenAI model...")

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please create a .env file in the root directory.")
        return

    # Initialize the actual language model from langchain-openai.
    model = ChatOpenAI(temperature=0)

    # Create the agent pool with the live model.
    agent_pool = DataAgentPool(model=model)

    # --- Example Queries ---
    queries = [
        "What is the current price of Ethereum (ETH)?",
        "What's the stock price for Google (GOOGL)?",
        "What's the latest news about semiconductor manufacturing?",
    ]

    for query in queries:
        print("\n" + "=" * 50)
        print(f"User Query: {query}")
        response = await agent_pool.process_query(query, session_id="session-live-123")
        print(f"\nAgent Response:\n{response}")
        print("=" * 50)


if __name__ == "__main__":
    # To run this, navigate to the directory containing 'test_client'
    # and run the following command:
    # python -m test_client.main
    asyncio.run(main())