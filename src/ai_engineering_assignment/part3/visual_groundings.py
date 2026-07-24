import json
from typing import Any


def _parse_possible_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _extract_image_paths_from_value(value: Any) -> list[str]:
    """Pull image_paths out of MCP/LangChain tool result shapes."""
    value = _parse_possible_json(value)

    if isinstance(value, dict):
        paths = value.get("image_paths", [])
        if isinstance(paths, str):
            paths = [paths]
        elif not isinstance(paths, list):
            paths = []

        nested_values = [
            value.get("structured_content"),
            value.get("structuredContent"),
            value.get("artifact"),
            value.get("content"),
            value.get("text"),
        ]
        return [
            *[path for path in paths if isinstance(path, str)],
            *[
                path
                for nested in nested_values
                for path in _extract_image_paths_from_value(nested)
            ],
        ]

    if isinstance(value, list):
        return [
            path for item in value for path in _extract_image_paths_from_value(item)
        ]

    return []


def _extract_self_ref_image_paths_from_value(value: Any) -> dict[str, str]:
    value = _parse_possible_json(value)

    if isinstance(value, dict):
        raw_mapping = value.get("self_ref_image_paths", {})
        mapping = (
            {
                ref: path
                for ref, path in raw_mapping.items()
                if isinstance(ref, str) and isinstance(path, str)
            }
            if isinstance(raw_mapping, dict)
            else {}
        )

        nested_values = [
            value.get("structured_content"),
            value.get("structuredContent"),
            value.get("artifact"),
            value.get("content"),
            value.get("text"),
        ]
        for nested in nested_values:
            mapping.update(_extract_self_ref_image_paths_from_value(nested))

        return mapping

    if isinstance(value, list):
        mapping = {}
        for item in value:
            mapping.update(_extract_self_ref_image_paths_from_value(item))
        return mapping

    return {}


def extract_image_paths_from_tool_message(msg: Any) -> list[str]:
    image_paths = []

    artifact = getattr(msg, "artifact", None)
    if artifact is not None:
        structured_content = getattr(artifact, "structured_content", None)
        image_paths.extend(_extract_image_paths_from_value(structured_content))
        image_paths.extend(_extract_image_paths_from_value(artifact))

    image_paths.extend(_extract_image_paths_from_value(getattr(msg, "content", None)))

    return list(dict.fromkeys(image_paths))


def extract_self_ref_image_paths_from_tool_message(msg: Any) -> dict[str, str]:
    self_ref_image_paths = {}

    artifact = getattr(msg, "artifact", None)
    if artifact is not None:
        structured_content = getattr(artifact, "structured_content", None)
        self_ref_image_paths.update(
            _extract_self_ref_image_paths_from_value(structured_content)
        )
        self_ref_image_paths.update(_extract_self_ref_image_paths_from_value(artifact))

    self_ref_image_paths.update(
        _extract_self_ref_image_paths_from_value(getattr(msg, "content", None))
    )

    return self_ref_image_paths
