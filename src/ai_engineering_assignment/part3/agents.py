from langchain.agents import create_agent

from ai_engineering_assignment.part3.model import model, get_tools
from ai_engineering_assignment.part3.prompts import (
    REVENUE_AGENT_PROMPT,
    EXPENDITURE_AGENT_PROMPT,
)


async def build_agents():
    tools = await get_tools()

    return {
        "revenue_agent": create_agent(
            model=model,
            tools=tools,
            system_prompt=REVENUE_AGENT_PROMPT,
            name="revenue_agent",
        ),
        "expenditure_agent": create_agent(
            model=model,
            tools=tools,
            system_prompt=EXPENDITURE_AGENT_PROMPT,
            name="expenditure_agent",
        ),
    }
