import os

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

from ai_engineering_assignment.settings import MainConfig
from ai_engineering_assignment.part2.schema.query import DateResult

configs = MainConfig()

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

structured_model = model.with_structured_output(DateResult)
