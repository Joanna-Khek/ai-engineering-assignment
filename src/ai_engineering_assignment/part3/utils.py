def merge_agent_lists(
    left: dict[str, list[str]], right: dict[str, list[str]]
) -> dict[str, list[str]]:
    merged = {agent: values.copy() for agent, values in left.items()}

    for agent, values in right.items():
        merged.setdefault(agent, [])
        merged[agent].extend(values)
        merged[agent] = list(dict.fromkeys(merged[agent]))

    return merged


def merge_agent_dicts(
    left: dict[str, dict[str, str]], right: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    merged = {agent: values.copy() for agent, values in left.items()}

    for agent, values in right.items():
        merged.setdefault(agent, {})
        merged[agent].update(values)

    return merged
