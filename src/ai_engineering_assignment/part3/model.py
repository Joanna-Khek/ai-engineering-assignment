import os

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from ai_engineering_assignment.settings import MainConfig
from ai_engineering_assignment.part3.state import SupervisorPlan

configs = MainConfig()
_tools = None

mcp_client = MultiServerMCPClient(
    {
        "extractor": {
            "command": "uv",
            "args": [
                "run",
                "python",
                "src/ai_engineering_assignment/part2/extractor_server.py",
            ],
            "cwd": str(configs.ROOT),
            "transport": "stdio",
            "env": {
                **os.environ,
                "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            },
        }
    }
)

model = ChatAnthropic(
    model_name=configs.app.llm.model,
    api_key=configs.ANTHROPIC_API_KEY,
)  # type: ignore[call-arg]

supervisor_model = model.with_structured_output(SupervisorPlan)


async def get_tools():
    global _tools

    if _tools is None:
        _tools = await mcp_client.get_tools()

    return _tools
