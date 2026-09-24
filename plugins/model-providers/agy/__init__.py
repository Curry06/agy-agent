"""Antigravity (AGY) external-process provider.

AGY owns authentication and its agent/tool runtime. Hermes uses this profile as a
provider transport and talks to the local agy CLI over stream-json.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class AGYProfile(ProviderProfile):
    """Antigravity CLI provider."""

    def create_client(self, **client_kwargs: Any) -> Any:
        from agent.agy_client import AGYClient
        return AGYClient(**client_kwargs)

    def setup_status(self, **kwargs: Any) -> dict[str, Any] | None:
        """Probe the local AGY CLI without starting an interactive session."""
        import subprocess

        command = str(kwargs.get("command") or self.process_command or "agy")
        timeout = float(kwargs.get("timeout") or 10.0)
        try:
            probe = subprocess.run(
                [command, "models", "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return {
                "available": False,
                "logged_in": False,
                "detail": "AGY CLI was not found on PATH.",
                "login_command": "agy",
            }
        except subprocess.TimeoutExpired:
            return {
                "available": True,
                "logged_in": False,
                "detail": "AGY model discovery timed out.",
                "login_command": "agy",
            }

        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "").strip()
            return {
                "available": True,
                "logged_in": False,
                "detail": detail or "AGY model discovery failed.",
                "login_command": "agy",
            }

        return {
            "available": True,
            "logged_in": True,
            "detail": "AGY model discovery succeeded.",
            "login_command": "agy",
        }

    def discover_models(self, **kwargs: Any) -> list[dict[str, Any]] | None:
        models = self.fetch_models(timeout=float(kwargs.get("timeout") or 15.0))
        if not models:
            return None
        return [
            {"id": model, "label": model, "note": "Antigravity model"}
            for model in models
        ]

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        return ("low", "medium", "high")

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> list[str] | None:
        try:
            from hermes_cli.auth import resolve_external_process_provider_credentials

            creds = resolve_external_process_provider_credentials(self.name)
            client = self.create_client(
                api_key=creds.get("api_key"),
                base_url=creds.get("base_url"),
                command=creds.get("command"),
                args=creds.get("args"),
            )
            return client.list_models(timeout_seconds=timeout) or None
        except Exception:
            return None


agy = AGYProfile(
    name="agy",
    aliases=("antigravity", "antigravity-cli"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="agy://local",
    auth_type="external_process",
    supports_health_check=False,
    supports_model_listing=True,
    process_command="agy",
    process_args=("--input-format", "stream-json", "--output-format", "stream-json"),
    process_command_env_vars=("HERMES_AGY_COMMAND", "AGY_CLI_PATH"),
    process_args_env_var="HERMES_AGY_ARGS",
    fallback_models=(),
)

register_provider(agy)
