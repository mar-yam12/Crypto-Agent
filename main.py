from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, RunConfig
import requests
from dotenv import load_dotenv, find_dotenv
import os
import chainlit as cl

# Load environment variables
load_dotenv(find_dotenv())

# Get API key and validate
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 2: Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# Step 3: Config define at run level
config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# # Step 4: Agent
# agent = Agent(
#     name="Assistant",
#     instructions="You are a helpful assistant"
# )

# Step 5: Function tool
@function_tool
def crypto_price(symbol: str) -> str:
    """
    Fetch the current price of a cryptocurrency.
    """
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    if symbol in data:
        return f"The current price of {symbol} is ${data[symbol]['usd']}."
    else:
        return "Cryptocurrency not found."

crypto_agent = Agent(
    name="CryptoPriceAssistant",
    instructions="You are a helpful assistant that provides cryptocurrency prices.",
    tools=[crypto_price],
)


@cl.on_chat_start
async def handle_chat_start():
    """Initialize chat session when a user connects."""
    # Initialize empty chat history
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Hello! I'm Maryam Shahid. How can I help you today?").send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("chat_history")
    history.append({"role": "user", "content": message.content})
    result = await Runner.run(
        crypto_agent, 
        input=history,
        run_config=config
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("chat_history", history)
    await cl.Message(content=result.final_output).send()
