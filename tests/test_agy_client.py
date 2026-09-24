from __future__ import annotations

import json
import queue
import threading

from agent.agy_client import AGYClient


def test_extract_model_ids():
    assert AGYClient._extract_model_ids(
        {"models": [{"id": "gemini-fast"}, {"slug": "gemini-pro"}, "plain"]}
    ) == ["gemini-fast", "gemini-pro", "plain"]


def test_format_messages():
    prompt = AGYClient._format_messages([
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Hello"},
    ])
    assert "[SYSTEM]" in prompt
    assert "[USER]" in prompt
    assert "Hello" in prompt


def test_stream_result_protocol():
    client = AGYClient(command="agy-test")
    client._events.put({
        "event": "init",
        "conversation_id": "conv-1",
        "init": {"tools": ["run_command"], "permission_mode": "request-review"},
    })
    client._events.put({
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-1",
            "step_type": "tool",
            "tool_name": "run_command",
            "tool_info": {"name": "run_command", "parameters": {"CommandLine": "echo hi"}},
        },
    })
    client._events.put({
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-1",
            "step_type": "agent_response",
            "text_delta": "hello",
        },
    })
    client._events.put({
        "event": "result",
        "result": {
            "conversation_id": "conv-1",
            "status": "SUCCESS",
            "response": "hello",
            "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
        },
    })

    client._proc = object()
    client._spawn = lambda: None
    client._send = lambda payload: None

    response, usage, conversation_id = client._run_turn(
        "hello", model=None, timeout_seconds=1
    )

    assert response == "hello"
    assert conversation_id == "conv-1"
    assert usage["total_tokens"] == 4
    assert client.conversation_id == "conv-1"
    assert client.init_info["permission_mode"] == "request-review"
    assert client.last_tool_event["tool_name"] == "run_command"


def test_error_result_protocol():
    client = AGYClient(command="agy-test")
    client._events.put({
        "event": "result",
        "result": {
            "conversation_id": "conv-2",
            "status": "ERROR",
            "response": "",
            "error": "authentication required",
        },
    })
    client._proc = object()
    client._spawn = lambda: None
    client._send = lambda payload: None

    try:
        client._run_turn("hello", model=None, timeout_seconds=1)
    except RuntimeError as exc:
        assert "authentication required" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
