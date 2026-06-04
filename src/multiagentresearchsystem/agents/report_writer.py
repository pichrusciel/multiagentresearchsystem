import json

from langchain.messages import AIMessage
from multiagentresearchsystem.state_schema import ResearchState, SearchTaskState, llm, creative_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from multiagentresearchsystem.structlogger import logger

# ============================================================
# Node: Report Writer — Produces the final report
# ============================================================


def report_writer(state: ResearchState) -> dict:
    """Writes a structured research report from the analysis."""
    logger.info("Report Writer Agent- started")

    # Include quality feedback if this is a revision
    revision_note = ""
    if state["iteration"] > 0 and state.get("quality_feedback"):
        revision_note = (
            f"\n\nIMPORTANT — This is revision #{state['iteration']}. "
            f"Address this feedback: {state['quality_feedback']}"
        )

    response = creative_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a report writer. Produce a well-structured research report "
                    "with these sections:\n"
                    "1. Executive Summary (2-3 sentences)\n"
                    "2. Key Findings (bullet points)\n"
                    "3. Analysis (1-2 paragraphs)\n"
                    "4. Recommendations (3 actionable items)\n\n"
                    "Use markdown formatting. Be specific and actionable."
                    f"{revision_note}"
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n\n"
                    f"Analysis:\n{state['analysis']}\n\n"
                    f"Raw findings:\n{json.dumps(state['findings'][:6], indent=2)}"
                )
            ),
        ]
    )
    logger.info("Report Writer Agent- completed")
    
    return {
        "report": response.content,
        "messages": [
            AIMessage(
                content=f"[REPORT WRITER]: Report {'revised' if state['iteration'] > 0 else 'drafted'}.",
                name="report_writer",
            )
        ],
    }