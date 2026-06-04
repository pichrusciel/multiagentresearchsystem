from multiagentresearchsystem.agents.analyst import analyst
from multiagentresearchsystem.agents.quality_checker import quality_checker, quality_gate
from multiagentresearchsystem.agents.report_writer import report_writer
from multiagentresearchsystem.agents.search_agent import dispatch_searches, search_agent
from multiagentresearchsystem.agents.supervisor import supervisor
from multiagentresearchsystem.state_schema import ResearchState
from langgraph.graph import StateGraph, START, END


def create_research_system():
    """
    Builds the multi-agent research system graph.

    Flow:
    1. Supervisor plans search queries
    2. Search agents run in parallel (Send API)
    3. Analyst synthesizes findings
    4. Report writer produces report
    5. Quality checker reviews — loops back if needed
    """

    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("supervisor", supervisor) 
    graph.add_node("search_agent", search_agent)
    graph.add_node("analyst", analyst)
    graph.add_node("report_writer", report_writer)
    graph.add_node("quality_checker", quality_checker)

    # Define edges
    graph.add_edge(START, "supervisor")

    # Supervisor → parallel search agents (dynamic fan-out)
    graph.add_conditional_edges("supervisor", dispatch_searches, ["search_agent"])

    # All search agents → analyst (fan-in)
    graph.add_edge("search_agent", "analyst")

    # Analyst → report writer
    graph.add_edge("analyst", "report_writer")

    # Report writer → quality checker
    graph.add_edge("report_writer", "quality_checker")

    # Quality checker → either END or back to report writer for revisions
    graph.add_conditional_edges(
        "quality_checker", quality_gate, {"report_writer": "report_writer", "end": END}
    )

    return graph.compile()
