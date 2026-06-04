import os

from dotenv import load_dotenv
from langchain.messages import AIMessage
from .state_schema import ResearchState, SearchTaskState, llm, creative_llm
from .research_system import create_research_system
from .structlogger import logger

load_dotenv()

#def simple_llm_test():
#    response = llm.invoke("What is the capital of France?")
#    print("LLM Response:", response.content)

def run_app():
    print("Running Multi-Agent Research System...")
    logger.info("Multi-Agent Research System - started")
    #print(f"API Key found: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    #simple_llm_test()

    research_system = create_research_system()

    topic = ""
    while topic == "":
        topic = input("Please enter a valid research topic or 'END' to exit: ").strip()
        if topic.upper() == "END":
            print("Exiting application.")
            return

    #topic = "The impact of AI agents on software development in 2026"

    print(f"Topic: {topic}\n")
    logger.info("Executing research lifecycle", topic=topic)

    result = research_system.invoke(
        {
            "messages": [],
            "topic": topic,
            "search_queries": [],
            "findings": [],
            "analysis": "",
            "report": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "iteration": 0,
        }
    )

    # Print the conversation trace
    print("Agent Activity Log:")
    print("-" * 40)
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"  {msg.content[:120]}...")
    print()

    # Print final stats
    print(f"Total findings collected: {len(result['findings'])}")
    print(f"Quality score: {result['quality_score']:.1f}")
    print(f"Iterations: {result['iteration']}")
    print()

    # Print the final report
    print("=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(result["report"])    
    logger.info("Multi-Agent Research System - completed")

if __name__ == "__main__":
    run_app()
