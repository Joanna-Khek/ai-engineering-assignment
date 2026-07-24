from pprint import pformat
from textwrap import indent


def format_namespace(namespace: tuple) -> str:
    if not namespace:
        return "graph"

    return " > ".join(part.split(":")[0] for part in namespace)


def _print_content(content, prefix="      "):
    if not content:
        print(f"{prefix}<empty>")
        return

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    print(indent(block["text"], prefix))
                else:
                    print(indent(pformat(block), prefix))
            else:
                print(indent(str(block), prefix))
        return

    if not isinstance(content, str):
        content = str(content)

    print(indent(content.strip(), prefix))


def format_stream_update(namespace: tuple, chunk: dict) -> None:
    print(f"\n[{format_namespace(namespace)}]")

    for node_name, output in chunk.items():
        print(f"├── Node: {node_name}")

        if not isinstance(output, dict):
            print(f"│   {output}")
            continue

        if "plan" in output:
            print("│")
            print("├── Supervisor Plan")
            for task in output["plan"]:
                print(f"│   • {task['agent']}")
                print(f"│     {task['instructions']}")

        if "messages" in output:
            for msg in output["messages"]:
                msg_type = type(msg).__name__

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print("│")
                    print(f"├── 🤖 {msg_type}")

                    for tool_call in msg.tool_calls:
                        print(f"│   Tool: {tool_call['name']}")
                        print(indent(pformat(tool_call["args"]), "│     "))

                else:
                    print("│")
                    print(f"├── 💬 {msg_type}")
                    _print_content(getattr(msg, "content", str(msg)), "│     ")

        if "final_report" in output:
            print("│")
            print("└── ✅ Final report generated")
