import json
from itertools import chain
from langchain_core.messages import ToolMessage


def collect_images(result):
    payloads = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "extract_data":
            text = (
                msg.content[0]["text"] if isinstance(msg.content, list) else msg.content
            )
            try:
                payloads.append(json.loads(text))
            except (json.JSONDecodeError, TypeError, KeyError, IndexError):
                continue

    image_paths = chain.from_iterable(p.get("image_paths", []) for p in payloads)
    return list(dict.fromkeys(image_paths))
