import langchain
from dotenv import load_dotenv
import os

def check_env():
    load_dotenv()
    print(f"LangChain version: {langchain.__version__}")
    print(f"API Key found: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    print(f"Anthropic API Key found: {'Yes' if os.getenv('ANTHROPIC_API_KEY') else 'No'}")
    print(f"LangSmith API Key found: {'Yes' if os.getenv('LANGSMITH_API_KEY') else 'No'}")
    print(f"LangSmith Tracing enabled: {'Yes' if os.getenv('LANGSMITH_TRACING') == 'true' else 'No'}")

if __name__ == "__main__":
    check_env()