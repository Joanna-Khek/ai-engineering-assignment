from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from ai_engineering_assignment.part3.utils import merge_agent_dicts, merge_agent_lists


class ReportState(TypedDict):
    query: str
    doc_json_path: str
    plan: list[dict]
    agent_findings: Annotated[dict[str, list[str]], merge_agent_lists]
    agent_image_paths: Annotated[dict[str, list[str]], merge_agent_lists]
    agent_self_ref_image_paths: Annotated[dict[str, dict[str, str]], merge_agent_dicts]
    final_report: str


class AgentState(TypedDict):
    instructions: str
    document_path: str
    messages: Annotated[list, add_messages]
    agent_findings: Annotated[dict[str, list[str]], merge_agent_lists]
    agent_image_paths: Annotated[dict[str, list[str]], merge_agent_lists]
    agent_self_ref_image_paths: Annotated[dict[str, dict[str, str]], merge_agent_dicts]


# Supervisor's decision schema — which agents, and what to tell each one
class AgentTask(BaseModel):
    agent: Literal["revenue_agent", "expenditure_agent"]
    instructions: str = Field(
        description="Specific, self-contained task instructions for this agent"
    )


class SupervisorPlan(BaseModel):
    tasks: list[AgentTask] = Field(
        description="Only include agents actually needed for this query"
    )
