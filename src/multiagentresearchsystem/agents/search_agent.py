from pydantic import BaseModel, Field

from multiagentresearchsystem.state_schema import ResearchState, SearchTaskState, llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import json
from langgraph.types import Send
from multiagentresearchsystem.structlogger import logger

class FindingItem(BaseModel):
    """An individual granular research finding."""
    title: str = Field(description="The short, punchy headline of the finding.")
    detail: str = Field(description="A 2-3 sentence technical extraction expanding on the title.")

class SearchAgentOutput(BaseModel):
    """The collection wrapper forced onto the LLM output layer."""
    findings: list[FindingItem] = Field(
        description="A list of 2-3 key technical findings extracted for the query.",
        min_items=2,
        max_items=3
    )

def search_agent(state: SearchTaskState) -> dict:
    """
    Executes one search query and returns findings.
    Each instance runs in parallel via the Send API.
    """
    logger.info("Search Agent- started")

    structured_llm = llm.with_structured_output(SearchAgentOutput)

    query = state["search_query"]

    response = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a web research agent. For the given search query, "
                    "provide 2-3 key findings. Each finding should have a 'title' "
                    "and 'detail' field."
                )
            ),
            HumanMessage(content=f"Search query: {query}"),         
        ]
    )

    #try:
    #    results = json.loads(response.content)
    #except json.JSONDecodeError:
        # Fallback: return a single finding if JSON parsing fails
        # results = [{"title": query, "detail": response.content}]

    # Convert the immutable Pydantic models to standard dictionaries 
    # and tag them with the 'source_query' metadata lineage tracking tag for later synthesis by the analyst agent.
    processed_findings = []

    for finding in response.findings:
        finding_dict = finding.model_dump() # Converts Pydantic object to native python dict
        finding_dict["source_query"] = query
        processed_findings.append(finding_dict)
    logger.info(f"Search Agent- completed with {len(processed_findings)} findings for query: '{query}'")
    
    return {"findings": processed_findings}

# ============================================================
# Edge: Searches using Send API
# ============================================================

def dispatch_searches(state: ResearchState) -> list[Send]:
    """Dispatches search queries to search agents via Send API."""
    return [
        Send("search_agent", {"search_query": query, "findings": []})
        for query in state["search_queries"]
    ]