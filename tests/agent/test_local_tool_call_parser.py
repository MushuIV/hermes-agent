"""Unit tests for the multi-dialect local tool call rescue parser."""

import json
from agent.local_tool_call_parser import rescue_tool_calls_from_content


def test_rescue_hermes_qwen_xml():
    content = (
        "I'll list the files for you.\n"
        "<tool_call>\n"
        '{"name": "terminal", "arguments": {"command": "dir /b"}}\n'
        "</tool_call>"
    )
    valid_tools = {"terminal", "read_file"}
    cleaned, calls = rescue_tool_calls_from_content(content, valid_tool_names=valid_tools)

    assert len(calls) == 1
    assert calls[0].function.name == "terminal"
    assert json.loads(calls[0].function.arguments) == {"command": "dir /b"}
    assert cleaned == "I'll list the files for you."


def test_rescue_qwen3_coder_parameter_tags():
    content = (
        "<tool_call>\n"
        '<function=read_file>\n'
        '<parameter=path>src/main.py</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )
    valid_tools = {"read_file"}
    cleaned, calls = rescue_tool_calls_from_content(content, valid_tool_names=valid_tools)

    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert json.loads(calls[0].function.arguments) == {"path": "src/main.py"}
    assert cleaned is None


def test_rescue_llama3_python_tag():
    content = '<|python_tag|>terminal(command="python -V")<|eom_id|>'
    valid_tools = {"terminal"}
    cleaned, calls = rescue_tool_calls_from_content(content, valid_tool_names=valid_tools)

    assert len(calls) == 1
    assert calls[0].function.name == "terminal"
    assert json.loads(calls[0].function.arguments) == {"command": "python -V"}


def test_rescue_mistral_tool_calls():
    content = '[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "README.md"}}]'
    valid_tools = {"read_file"}
    cleaned, calls = rescue_tool_calls_from_content(content, valid_tool_names=valid_tools)

    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert json.loads(calls[0].function.arguments) == {"path": "README.md"}


def test_rescue_fenced_block():
    content = (
        "Here is the command:\n"
        "```tool_call\n"
        '{"name": "terminal", "arguments": {"command": "git status"}}\n'
        "```"
    )
    valid_tools = {"terminal"}
    cleaned, calls = rescue_tool_calls_from_content(content, valid_tool_names=valid_tools)

    assert len(calls) == 1
    assert calls[0].function.name == "terminal"
    assert json.loads(calls[0].function.arguments) == {"command": "git status"}
    assert cleaned == "Here is the command:"


def test_negative_regular_json_not_rescued():
    content = (
        "Here is an example config file:\n"
        "```json\n"
        '{"name": "my-app", "version": "1.0.0"}\n'
        "```"
    )
    valid_tools = {"terminal", "read_file"}
    cleaned, calls = rescue_tool_calls_from_content(content, valid_tool_names=valid_tools)

    assert len(calls) == 0
    assert cleaned == content
