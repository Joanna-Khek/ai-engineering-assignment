from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from ai_engineering_assignment.settings import MainConfig
from ai_engineering_assignment.part3.prompts import SUPERVISOR_AGENT_PROMPT
from ai_engineering_assignment.part3.state import AgentState, ReportState
from ai_engineering_assignment.part3.visual_groundings import (
    extract_image_paths_from_tool_message,
    extract_self_ref_image_paths_from_tool_message,
)
from ai_engineering_assignment.part3.model import supervisor_model, model
from ai_engineering_assignment.part3.agents import build_agents

configs = MainConfig()


def build_subagent_graph(react_agent, agent_name: str):
    async def prepare(state):
        content = (
            f"{state['instructions']}\n\nDocument JSON path: {state['document_path']}"
        )
        return {"messages": [HumanMessage(content=content)]}

    async def collect(state):
        # Save the images
        image_paths = []
        self_ref_image_paths = {}
        for msg in state["messages"]:
            if type(msg).__name__ == "ToolMessage":
                image_paths.extend(extract_image_paths_from_tool_message(msg))
                self_ref_image_paths.update(
                    extract_self_ref_image_paths_from_tool_message(msg)
                )
        unique_image_paths = list(dict.fromkeys(image_paths))
        finding = f"[{agent_name}] {state['messages'][-1].content}"
        return {
            "agent_findings": {agent_name: [finding]},
            "agent_image_paths": {agent_name: unique_image_paths},
            "agent_self_ref_image_paths": {agent_name: self_ref_image_paths},
        }

    sub = StateGraph(AgentState)
    sub.add_node("prepare", prepare)
    sub.add_node(f"{agent_name}_llm", react_agent)
    sub.add_node("collect", collect)
    sub.add_edge(START, "prepare")
    sub.add_edge("prepare", f"{agent_name}_llm")
    sub.add_edge(f"{agent_name}_llm", "collect")
    sub.add_edge("collect", END)
    return sub.compile()


async def supervisor(state: ReportState):
    plan = await supervisor_model.ainvoke(
        [
            HumanMessage(
                content=SUPERVISOR_AGENT_PROMPT.format(user_query=state["query"])
            )
        ]
    )
    return {"plan": [t.model_dump() for t in plan.tasks]}


async def synthesize(state: ReportState):
    combined = "\n\n".join(
        finding for findings in state["agent_findings"].values() for finding in findings
    )
    result = await model.ainvoke(
        [
            HumanMessage(
                content=(
                    "Combine these specialist findings into one coherent report. "
                    "Preserve the relevant self-references, such as #/texts/224 or #/tables/7, "
                    "next to the claims they support.\n\n" + combined
                )
            )
        ]
    )
    return {"final_report": result.content}


# Send() now targets each agent's own node name
def route_to_agents(state: ReportState):
    return [
        Send(
            t["agent"],
            {  # "revenue_agent" or "expenditure_agent" directly
                "instructions": t["instructions"],
                "document_path": state["doc_json_path"],
            },
        )
        for t in state["plan"]
    ]


async def build_graph():
    agents = await build_agents()

    revenue_subgraph = build_subagent_graph(
        agents["revenue_agent"],
        "revenue_agent",
    )

    expenditure_subgraph = build_subagent_graph(
        agents["expenditure_agent"],
        "expenditure_agent",
    )

    graph = StateGraph(ReportState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("revenue_agent", revenue_subgraph)
    graph.add_node("expenditure_agent", expenditure_subgraph)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", route_to_agents, ["revenue_agent", "expenditure_agent"]
    )
    graph.add_edge("revenue_agent", "synthesize")
    graph.add_edge("expenditure_agent", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
