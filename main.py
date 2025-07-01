from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI,RunConfig
import os 
from dotenv import load_dotenv


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client=provider
)

config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled= True
)

first_agent = Agent(
    name= "Fronted Expert",
    instructions= "You are a fronted expert agent that can help with any question about the  fronted of a website.",
)

result = Runner.run_sync(
    first_agent,
    input="What is the best way to improve the fronted of a website?",
    run_config=config
)

print(result.final_output)