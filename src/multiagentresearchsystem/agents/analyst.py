from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from multiagentresearchsystem.state_schema import ResearchState, SearchTaskState, llm
from multiagentresearchsystem.state_schema import ResearchState
from multiagentresearchsystem.structlogger import logger

# ============================================================
# Node: Analyst — Synthesizes all findings
# ============================================================

def analyst(state: ResearchState) -> dict:
    """Reads all findings from the blackboard and synthesizes."""
    logger.info("Analyst Agent- started")

    findings_list = state.get("findings", [])

    if not findings_list:
        return {
            "analysis": "No research findings were gathered to analyze.",
            "messages": [AIMessage(content="[ANALYST]: Execution skipped due to missing findings.", name="analyst")]
        }    
    
    # Clean data representation (Reduces token noise compared to raw json.dumps)
    formatted_findings_blocks = []
    for idx, f in enumerate(findings_list, 1):
        source_info = f.get("source_query", "Unknown Query")
        title = f.get("title", "No Title")
        detail = f.get("detail", "No Detail")

        block = f"Finding #{idx} [Originating Query: '{source_info}']\n- Title: {title}\n- Details: {detail}"
        formatted_findings_blocks.append(block)    

    findings_text = "\n\n".join(formatted_findings_blocks)

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a research analyst. Synthesize the collected findings into "
                    "a clear analysis. Identify:\n"
                    "1. Key themes across all findings\n"
                    "2. Any contradictions or gaps\n"
                    "3. The most important insights\n\n"
                    "Write 2-3 paragraphs."
                )
            ),
            HumanMessage(
                content=(
                    f"Research topic: {state['topic']}\n\n"
                    f"Collected findings:\n{findings_text}"
                )
            ),
        ]
    )
    logger.info("Analyst Agent- completed")

    return {
        "analysis": response.content,
        "messages": [
            AIMessage(content=f"[ANALYST]: {response.content}", name="analyst")
        ],
    }
