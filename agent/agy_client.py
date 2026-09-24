"""Hermes transport client for the Antigravity (AGY) CLI.

The AGY CLI is an external agent process rather than an OpenAI-compatible HTTP
server. It accepts newline-delimited JSON on stdin and emits newline-delimited
JSON on stdout. This adapter presents a small OpenAI-client-compatible facade
to Hermes while preserving AGY's persistent conversation.

This file intentionally contains no code from AGY-TELPORT.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator


DEFAULT_TIMEOUT_SECONDS = 300.0


class AGYClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        agy_command: str | None = None,
        agy_args: list[str] | None = None,
        agy_cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "agy"
        self.base_url = base_url or "agy://local"
        self._default_headers = dict(default_headers or {})
        self._command = agy_command or command or "agy"
        self._args = list(agy_args or args or [
            "--input-format", "stream-json",
            "--output-format", "stream-json",
        ])
        self._cwd = str(Path(agy_cwd or os.getcwd()).resolve())

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_chat_completion)
        )

        self.is_closed = False
        self._proc: subprocess.Popen[str] | None = None
        self._proc_lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr: queue.Queue[str] = queue.Queue(maxsize=100)
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._conversation_id: str | None = None
        self._turns = 0

    def _spawn(self) -> None:
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                return

            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")

            try:
                proc = subprocess.Popen(
                    [self._command, *self._args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=self._cwd,
                    env=env,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Could not start AGY CLI '{self._command}'. "
                    "Install/authenticate the Antigravity CLI and ensure 'agy' is on PATH."
                ) from exc

            if proc.stdin is None or proc.stdout is None:
                with contextlib.suppress(Exception):
                    proc.kill()
                raise RuntimeError("AGY process did not expose stdin/stdout pipes.")

            self._proc = proc
            self.is_closed = False
            self._conversation_id = None
            self._turns = 0

            self._reader = threading.Thread(
                target=self._stdout_loop, args=(proc,), daemon=True, name="hermes-agy-stdout"
            )
            self._reader.start()
            self._stderr_reader = threading.Thread(
                target=self._stderr_loop, args=(proc,), daemon=True, name="hermes-agy-stderr"
            )
            self._stderr_reader.start()

    def _stdout_loop(self, proc: subprocess.Popen[str]) -> None:
        try:
            for line in proc.stdout or ():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self._events.put({
                        "_event": "protocol_error",
                        "error": f"AGY emitted invalid JSON: {line[:500]}",
                    })
                    continue
                if isinstance(event, dict):
                    self._events.put(event)
        finally:
            self._events.put({"_event": "process_exit", "returncode": proc.poll()})

    def _stderr_loop(self, proc: subprocess.Popen[str]) -> None:
        for line in proc.stderr or ():
            try:
                self._stderr.put_nowait(line.rstrip())
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._stderr.get_nowait()
                with contextlib.suppress(queue.Full):
                    self._stderr.put_nowait(line.rstrip())

    def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise RuntimeError("AGY process is not running.")
        proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role") or "user").strip().lower()
            content = msg.get("content", "")
            if isinstance(content, list):
                fragments = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            fragments.append(str(item.get("text") or ""))
                    elif isinstance(item, str):
                        fragments.append(item)
                content = "\n".join(fragments)
            if content is None:
                content = ""
            if role == "system":
                label = "SYSTEM"
            elif role == "assistant":
                label = "ASSISTANT"
            elif role == "tool":
                label = "TOOL"
            else:
                label = "USER"
            parts.append(f"[{label}]\n{content}")
        return "\n\n".join(parts).strip()

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        prompt = self._format_messages(messages or [])
        if not prompt:
            prompt = "(No prompt was supplied.)"

        response, usage, conversation_id = self._run_turn(
            prompt,
            model=model,
            timeout_seconds=float(timeout or DEFAULT_TIMEOUT_SECONDS),
        )

        message = SimpleNamespace(
            content=response,
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                completion_tokens=int(usage.get("output_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=int(usage.get("cache_read_tokens", 0) or 0)
                ),
            ),
            model=model or "agy",
            id=conversation_id or "",
        )
        return self._completion_to_stream(completion) if stream else completion

    def _run_turn(
        self,
        prompt: str,
        *,
        model: str | None,
        timeout_seconds: float,
    ) -> tuple[str, dict[str, Any], str | None]:
        with self._io_lock:
            self._spawn()

            if self._turns == 0 and model:
                self._restart_with_model(model)

            self._send({"event": "user", "message": {"content": prompt}})

            deadline = time.monotonic() + timeout_seconds
            response = ""
            usage: dict[str, Any] = {}
            current_conversation = self._conversation_id

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._restart_process()
                    raise TimeoutError(
                        f"AGY request timed out after {timeout_seconds:.1f}s."
                    )

                try:
                    event = self._events.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    proc = self._proc
                    if proc is not None and proc.poll() is not None:
                        raise RuntimeError(self._process_error())
                    continue

                kind = event.get("event") or event.get("_event")

                if kind == "init":
                    self._conversation_id = str(event.get("conversation_id") or "") or None
                    current_conversation = self._conversation_id
                    continue

                if kind == "step_update":
                    step = event.get("step_update") or {}
                    if isinstance(step, dict):
                        cid = step.get("conversation_id")
                        if cid:
                            self._conversation_id = str(cid)
                            current_conversation = self._conversation_id
                        delta = step.get("text_delta")
                        if isinstance(delta, str):
                            response += delta
                    continue

                if kind == "result":
                    result = event.get("result") or {}
                    if not isinstance(result, dict):
                        raise RuntimeError("AGY returned an invalid result event.")

                    cid = result.get("conversation_id")
                    if cid:
                        self._conversation_id = str(cid)
                        current_conversation = self._conversation_id

                    status = str(result.get("status") or "").upper()
                    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
                    final_response = result.get("response")
                    if isinstance(final_response, str):
                        response = final_response

                    if status != "SUCCESS":
                        detail = str(result.get("error") or status or "unknown AGY error")
                        raise RuntimeError(f"AGY request failed ({status}): {detail}")

                    self._turns += 1
                    return response, usage, current_conversation

                if kind == "process_exit":
                    raise RuntimeError(self._process_error())

                if kind == "protocol_error":
                    raise RuntimeError(str(event.get("error") or "AGY protocol error"))

    def _restart_with_model(self, model: str) -> None:
        self._restart_process()
        self._args = [
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--model", str(model),
        ]
        self._spawn()

    def _restart_process(self) -> None:
        with self._proc_lock:
            proc = self._proc
            self._proc = None
            self._conversation_id = None
            self._turns = 0
        if proc is not None:
            with contextlib.suppress(Exception):
                if proc.stdin:
                    proc.stdin.close()
            with contextlib.suppress(Exception):
                proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()

    def _process_error(self) -> str:
        lines: list[str] = []
        while True:
            try:
                lines.append(self._stderr.get_nowait())
            except queue.Empty:
                break
        detail = "\n".join(x for x in lines if x).strip()
        return detail or "AGY process exited unexpectedly."

    def list_models(self, *, timeout_seconds: float = 15.0) -> list[str]:
        try:
            proc = subprocess.run(
                [self._command, "models", "--output-format", "json"],
                cwd=self._cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        if proc.returncode != 0:
            return []

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return []

        return self._extract_model_ids(data)

    @staticmethod
    def _extract_model_ids(data: Any) -> list[str]:
        items = data
        if isinstance(data, dict):
            for key in ("models", "data", "items"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
        if not isinstance(items, list):
            return []

        result: list[str] = []
        for item in items:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                value = item.get("id") or item.get("name") or item.get("slug")
                if value:
                    result.append(str(value))
        return list(dict.fromkeys(result))

    def _completion_to_stream(self, completion: Any) -> Iterator[Any]:
        content = completion.choices[0].message.content or ""
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=content, role="assistant"),
                finish_reason=None,
            )],
            model=completion.model,
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None),
                finish_reason="stop",
            )],
            model=completion.model,
            usage=completion.usage,
        )

    def close(self) -> None:
        self._restart_process()
        self.is_closed = True
