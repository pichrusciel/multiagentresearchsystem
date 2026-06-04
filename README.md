# MultiAgentResearchSystem

A structured, multi-agent research pipeline built with LangGraph and LangChain. This system coordinates specialized AI agents using a combination of architectural patterns (including Supervisor Planning and Blackboard-style workspaces) to plan research, execute searches, analyze data, write comprehensive reports, and perform quality checks.

## Project Architecture & Layout

The project follows the modern src/ layout managed by uv. The agents do not communicate directly with each other; instead, they operate over a structured central state (ResearchState class) acting as a shared workspace.

```text
src/
└── multiagentresearchsystem/
    ├── __init__.py           # Package initialization
    ├── main.py               # Application entry point / orchestration
    ├── research_system.py    # Main LangGraph graph compilation and wiring
    ├── state_schema.py       # Central Shared Fields TypedDict (ResearchState)
    ├── structlogger.py       # Custom formatted execution logger
    └── agents/
        ├── supervisor.py     # Plans research by generating Pydantic-validated search queries
        ├── search_agent.py   # Dispatches parallel external API searches
        ├── analyst.py        # Cleans, extracts, and summarizes raw data
        ├── report_writer.py  # Synthesizes analysis into the final document
        └── quality_checker.py# Acts as a strict gatekeeper (approves or routes back)
```

## State Management & Agent Roles

Rather than passing massive, token-heavy conversation transcripts back and forth, this system utilizes the Shared Fields (Blackboard) Pattern. Agents write to dedicated fields in a global ResearchState:

    1. Supervisor (supervisor.py): Uses a structured Pydantic schema (SupervisorPlan) via llm.with_structured_output() to generate exactly 3 highly targeted research queries based on the topic.

    2. Search Agent (search_agent.py): Executes parallel web searches for the generated queries and gathers raw data.

    3. Analyst (analyst.py): Sifts through raw web data, extracts critical insights, and evaluates information with an explicit confidence score.

    4. Report Writer (report_writer.py): Compiles the summarized insights and analysis into a polished markdown document.

    5. Quality Checker (quality_checker.py): A strict editorial node that reviews the final report. If it passes, the graph ends; if adjustments are needed, it injects structured feedback and loops execution back to the refinement nodes.