from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_protocol import Literal
from pydantic import BaseModel, Field
from multiagentresearchsystem.state_schema import ResearchState, SearchTaskState, llm, creative_llm
from multiagentresearchsystem.structlogger import logger

# ============================================================
# Node: Quality Checker — Reviews and scores the report
# ============================================================

class QualityReview(BaseModel):
    score: float = Field(description="Quality score from 0.0 to 1.0")
    feedback: str = Field(description="Specific feedback for improvement")
    approved: bool = Field(description="Whether the report meets quality standards")

def quality_checker(state: ResearchState) -> dict:
    """Reviews the report and either approves or sends back for revision."""
    logger.info("Quality Checker Agent- started")
    review_llm = llm.with_structured_output(QualityReview)

    review = review_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a quality reviewer. Score the report on:\n"
                    "- Completeness: Does it cover the topic well?\n"
                    "- Clarity: Is it well-written and easy to understand?\n"
                    "- Actionability: Are recommendations specific?\n\n"
                    "Score from 0.0 to 1.0. Approve if score >= 0.7.\n"
                    "If this is iteration 2 or higher, be more lenient."
                )
            ),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Iteration: {state['iteration']}\n\n"
                    f"Report:\n{state['report']}"
                )
            ),
        ]
    )

    # Force approve after 2 iterations to prevent infinite loops
    approved = review.approved or state["iteration"] >= 2
    logger.info(f"Quality Checker Agent- completed with score {review.score:.1f} and feedback: {review.feedback}")
    
    return {
        "quality_score": review.score,
        "quality_feedback": review.feedback,
        "iteration": state["iteration"] + 1,
        "messages": [
            AIMessage(
                content=(
                    f"[QUALITY CHECK]: Score {review.score:.1f} — "
                    f"{'APPROVED' if approved else 'REVISION NEEDED'}: {review.feedback}"
                ),
                name="quality_checker",
            )
        ],
    }

def quality_gate(state: ResearchState) -> Literal["report_writer", "end"]:
    """Route back to writer if quality is insufficient."""
    if state["quality_score"] >= 0.7 or state["iteration"] >= 2:
        return "end"
    return "report_writer"
