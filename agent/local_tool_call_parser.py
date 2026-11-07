"""Multi-dialect content-to-ToolCall rescue parser for local models.

Rescues tool calls emitted directly into `assistant_message.content` when a local model
(or external local server such as Ollama, llama-server, LM Studio, or vLLM) emits
tool-call tags instead of populating structured `tool_calls`.

Supported Dialects:
1. Hermes / Qwen XML: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
2. Qwen3-Coder / Tag Parameters: `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
   or `<function=name>...</function>` or `<function name="...">...</function>`
3. Llama 3.1 / 3.3 / DeepSeek: `<|python_tag|>{"name": ..., "parameters": ...}` or `<|python_tag|>fn.call(...)`
4. Mistral: `[TOOL_CALLS] [{"name": ..., "arguments": ...}]`
5. Fenced JSON / tool_call blocks: ```tool_call ... ``` or validated ```json ... ```
"""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from types import SimpleNamespace
from typing import Any, Optional, Set, Sequence

logger = logging.getLogger(__name__)

# Patterns for XML / special tool tags
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_TAG_ATTR_RE = re.compile(
    r'<function(?:=|\s+name=)["\']?([a-zA-Z0-9_\-]+)["\']?>(.*?)</function>',
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_TAG_RE = re.compile(
    r'<parameter(?:=|\s+name=)["\']?([a-zA-Z0-9_\-]+)["\']?>(.*?)</parameter>',
    re.DOTALL | re.IGNORECASE,
)
_PYTHON_TAG_RE = re.compile(r"<\|python_tag\|>(.*?)(?:<\|eom_id\|>|<\|eot_id\|>|$)", re.DOTALL)
_MISTRAL_TAG_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)
_FENCED_BLOCK_RE = re.compile(r"```(?:tool_call|json)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def _repair_lenient_json(raw: str) -> Optional[Any]:
    """Parse JSON with lenient repairs: strip markdown fences, strip trailing commas, handle single quotes."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            s = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    # Trailing commas: e.g. {"a": 1,} or [1, 2,]
    repaired = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(repaired)
    except Exception:
        pass

    # Python literal fallback (e.g. single quotes, True/False/None)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (dict, list)):
            return val
    except Exception:
        pass

    return None


def _normalize_args_to_json(args: Any, fn_name: str, tool_defs: Optional[Sequence[dict[str, Any]]] = None) -> str:
    """Ensure tool arguments are serialized to a valid JSON string, applying coercion if possible."""
    if not isinstance(args, dict):
        if isinstance(args, str):
            parsed = _repair_lenient_json(args)
            args = parsed if isinstance(parsed, dict) else {}
        else:
            args = {}

    if tool_defs:
        try:
            from tools.arg_coercion import coerce_tool_args
            args = coerce_tool_args(tool_defs, fn_name, args)
        except Exception as e:
            logger.debug("Arg coercion for %s skipped: %s", fn_name, e)

    return json.dumps(args, ensure_ascii=False)


def _make_tool_call(name: str, args: Any, tool_defs: Optional[Sequence[dict[str, Any]]] = None) -> SimpleNamespace:
    call_id = f"call_local_{uuid.uuid4().hex[:12]}"
    args_json = _normalize_args_to_json(args, name, tool_defs)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=args_json),
    )


def rescue_tool_calls_from_content(
    content: str,
    valid_tool_names: Optional[Set[str] | frozenset[str]] = None,
    tool_defs: Optional[Sequence[dict[str, Any]]] = None,
) -> tuple[Optional[str], list[SimpleNamespace]]:
    """Inspect assistant content for embedded tool call syntax across dialects.

    Returns:
        (cleaned_content, rescued_tool_calls)
        If no valid tool calls are found, returns (content, []).
        If tool calls are found, the tag spans are removed from cleaned_content.
    """
    if not isinstance(content, str) or not content.strip():
        return content, []

    rescued: list[SimpleNamespace] = []
    consumed_spans: list[tuple[int, int]] = []

    def _is_valid(name: str) -> bool:
        if not name:
            return False
        if valid_tool_names is not None:
            return name in valid_tool_names
        return True

    # 1. Hermes / Qwen <tool_call> tags
    for match in _TOOL_CALL_TAG_RE.finditer(content):
        inner = match.group(1).strip()
        span = match.span()

        # Check for nested <function=name><parameter=k>v</parameter></function>
        fn_match = _FUNCTION_TAG_ATTR_RE.search(inner)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            fn_body = fn_match.group(2).strip()
            if _is_valid(fn_name):
                params: dict[str, Any] = {}
                for p_match in _PARAMETER_TAG_RE.finditer(fn_body):
                    p_name = p_match.group(1).strip()
                    p_val = p_match.group(2).strip()
                    parsed_val = _repair_lenient_json(p_val)
                    params[p_name] = parsed_val if parsed_val is not None else p_val
                rescued.append(_make_tool_call(fn_name, params, tool_defs))
                consumed_spans.append(span)
                continue

        # JSON inside <tool_call>
        parsed = _repair_lenient_json(inner)
        if isinstance(parsed, dict):
            name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
            arguments = parsed.get("arguments") or parsed.get("parameters") or parsed.get("args") or {}
            if isinstance(name, str) and _is_valid(name):
                rescued.append(_make_tool_call(name, arguments, tool_defs))
                consumed_spans.append(span)
        elif isinstance(parsed, list):
            valid_in_list = False
            for item in parsed:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("tool") or item.get("function")
                    arguments = item.get("arguments") or item.get("parameters") or item.get("args") or {}
                    if isinstance(name, str) and _is_valid(name):
                        rescued.append(_make_tool_call(name, arguments, tool_defs))
                        valid_in_list = True
            if valid_in_list:
                consumed_spans.append(span)

    # 2. Standalone <function=name> tags outside <tool_call>
    if not rescued:
        for match in _FUNCTION_TAG_ATTR_RE.finditer(content):
            fn_name = match.group(1).strip()
            fn_body = match.group(2).strip()
            if not _is_valid(fn_name):
                continue
            span = match.span()
            # Try parameter tags first
            param_matches = list(_PARAMETER_TAG_RE.finditer(fn_body))
            if param_matches:
                params = {}
                for p_match in param_matches:
                    p_name = p_match.group(1).strip()
                    p_val = p_match.group(2).strip()
                    parsed_val = _repair_lenient_json(p_val)
                    params[p_name] = parsed_val if parsed_val is not None else p_val
                rescued.append(_make_tool_call(fn_name, params, tool_defs))
                consumed_spans.append(span)
            else:
                parsed_body = _repair_lenient_json(fn_body)
                if isinstance(parsed_body, dict):
                    rescued.append(_make_tool_call(fn_name, parsed_body, tool_defs))
                    consumed_spans.append(span)

    # 3. Mistral [TOOL_CALLS]
    if not rescued:
        for match in _MISTRAL_TAG_RE.finditer(content):
            inner = match.group(1).strip()
            parsed = _repair_lenient_json(inner)
            if isinstance(parsed, list):
                valid_any = False
                for item in parsed:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("tool")
                        arguments = item.get("arguments") or item.get("parameters") or {}
                        if isinstance(name, str) and _is_valid(name):
                            rescued.append(_make_tool_call(name, arguments, tool_defs))
                            valid_any = True
                if valid_any:
                    consumed_spans.append(match.span())

    # 4. Llama 3.1/3.3 <|python_tag|>
    if not rescued:
        for match in _PYTHON_TAG_RE.finditer(content):
            inner = match.group(1).strip()
            span = match.span()
            # Try JSON first
            parsed = _repair_lenient_json(inner)
            if isinstance(parsed, dict):
                name = parsed.get("name") or parsed.get("tool")
                arguments = parsed.get("arguments") or parsed.get("parameters") or {}
                if isinstance(name, str) and _is_valid(name):
                    rescued.append(_make_tool_call(name, arguments, tool_defs))
                    consumed_spans.append(span)
                    continue
            # Try call syntax: fn_name(arg1=val1, ...)
            call_match = re.match(r"^([a-zA-Z0-9_\-]+)(?:\.call)?\((.*)\)$", inner, re.DOTALL)
            if call_match:
                fn_name = call_match.group(1).strip()
                if _is_valid(fn_name):
                    call_args_str = call_match.group(2).strip()
                    # Parse kwargs
                    try:
                        expr = ast.parse(f"dummy({call_args_str})").body[0].value  # type: ignore
                        kwargs = {}
                        for kw in expr.keywords:
                            kwargs[kw.arg] = ast.literal_eval(kw.value)
                        rescued.append(_make_tool_call(fn_name, kwargs, tool_defs))
                        consumed_spans.append(span)
                    except Exception:
                        pass

    # 5. Fenced code block fallback (```tool_call or ```json with valid tool name)
    if not rescued and valid_tool_names:
        for match in _FENCED_BLOCK_RE.finditer(content):
            inner = match.group(1).strip()
            parsed = _repair_lenient_json(inner)
            if isinstance(parsed, dict):
                name = parsed.get("name") or parsed.get("tool")
                arguments = parsed.get("arguments") or parsed.get("parameters") or {}
                if isinstance(name, str) and name in valid_tool_names and isinstance(arguments, dict):
                    rescued.append(_make_tool_call(name, arguments, tool_defs))
                    consumed_spans.append(match.span())

    if not rescued or not consumed_spans:
        return content, []

    # Remove consumed spans from content
    consumed_spans.sort(key=lambda s: s[0])
    parts: list[str] = []
    last_idx = 0
    for start, end in consumed_spans:
        if start > last_idx:
            parts.append(content[last_idx:start])
        last_idx = max(last_idx, end)
    if last_idx < len(content):
        parts.append(content[last_idx:])

    cleaned_content = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return (cleaned_content if cleaned_content else None), rescued
