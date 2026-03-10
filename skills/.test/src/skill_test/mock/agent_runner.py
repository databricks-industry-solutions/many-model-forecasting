"""Tool-use agent loop for Tier 1 skill tests.

Sends skill prompts + tool definitions to Claude Sonnet on Databricks,
parses tool_calls, executes mock handlers, and feeds results back until
the model produces a final text response or max_turns is reached.
"""

import json

from openai import OpenAI


def run_skill_agent(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    max_turns: int = 20,
    model: str = "databricks-claude-sonnet-4-6",
) -> dict:
    """Run a multi-turn agent loop with mock tools.

    Args:
        client: OpenAI-compatible client pointing at Databricks Foundation Model API.
        system_prompt: Skill instructions (from SKILL.md).
        user_prompt: The user request, e.g. "/explore-data test_catalog default".
        tools: OpenAI-format tool definitions.
        tool_handlers: Map of tool_name -> callable that returns a result string.
        max_turns: Maximum number of API round-trips before stopping.
        model: Databricks model endpoint name.

    Returns:
        dict with keys:
            messages: Full conversation history.
            tool_calls: All tool calls made [{"name": ..., "arguments": ...}].
            final_response: Last assistant text content.
            turns: Number of API round-trips taken.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    all_tool_calls = []
    turns = 0
    final_response = ""

    for _ in range(max_turns):
        turns += 1

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            temperature=0,
        )

        choice = response.choices[0]
        assistant_message = choice.message

        # Build the message dict to append to history.
        # Databricks rejects empty content strings when tool_calls are present,
        # so only include content when it's non-empty.
        msg_dict = {"role": "assistant"}
        if assistant_message.content:
            msg_dict["content"] = assistant_message.content
        if assistant_message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_message.tool_calls
            ]
        messages.append(msg_dict)

        # If no tool calls, the model has finished
        if not assistant_message.tool_calls:
            final_response = assistant_message.content or ""
            break

        # Execute each tool call and append results
        for tc in assistant_message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            # Record the tool call
            all_tool_calls.append(
                {
                    "name": tool_name,
                    "arguments": tool_args,
                }
            )

            # Execute the mock handler
            handler = tool_handlers.get(tool_name)
            if handler:
                result = handler(**tool_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )
    else:
        # max_turns exhausted — capture whatever the last assistant said
        if messages and messages[-1]["role"] == "assistant":
            final_response = messages[-1].get("content", "")

    return {
        "messages": messages,
        "tool_calls": all_tool_calls,
        "final_response": final_response,
        "turns": turns,
    }
