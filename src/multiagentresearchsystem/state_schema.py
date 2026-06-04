
import operator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv

# ============================================================
# State Schema
# ============================================================

# State for the overall research process
class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    search_queries: list[str]
    findings: Annotated[list[dict], operator.add]
    analysis: str
    report: str
    quality_score: float
    quality_feedback: str
    iteration: int

# State for individual search tasks
class SearchTaskState(TypedDict):
    search_query: str
    findings: Annotated[list[dict], operator.add]

# ============================================================
# LLM models
# ============================================================
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)