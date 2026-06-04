from multiagentresearchsystem.state_schema import ResearchState, llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from pydantic import BaseModel, Field
import json
from multiagentresearchsystem.structlogger import logger

# ============================================================
# Supervisor Node — Plans the research
# ===============+============================================

class SupervisorPlan(BaseModel):
    queries: list[str] = Field(description="Exactly 3 specific, highly distinct search queries.", min_items=3, max_items=3)

def supervisor(state: ResearchState) -> dict:
    """Plans research by generating targeted search queries."""
    logger.info("Supervisor Agent- started")

    structured_llm = llm.with_structured_output(SupervisorPlan) 

    response = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research supervisor. Given a topic, generate exactly 3 "
                    "specific search queries that will cover different angles of the topic. "
                    #"Return ONLY a JSON array of strings. No markdown formatting."
                    )      
                ),
            HumanMessage(content=f"Research topic: {state['topic']}"),
        ]
    )

    #try:
    #    queries = response.queries
    #except json.JSONDecodeError:
    #    # Fallback: split by newlines if JSON parsing fails
    #    queries = [
    #        f"{state['topic']} overview",
    #        f"{state['topic']} latest developments",
    #        f"{state['topic']} practical applications",            
    #    ]
    logger.info(f"Supervisor Agent- completed with {len(response.queries)} queries: {response.queries}")
    
    return {
        "search_queries": response.queries[:3],
        "messages": [
            AIMessage(
                content=f"[SUPERVISOR]: Planned {len(response.queries)} research queries: {response.queries}",
                name="supervisor",
            )
        ],
    }

