"""Runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8080
    backend_url: str = ""
    backend_api_key: str = ""
    api_key: str = ""
    required_input_tokens: int = 100_000_000
    max_upload_bytes: int = 64 * 1024**3
    request_timeout_seconds: int = 7200
    temp_dir: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            host=os.getenv("LONGCTX_HOST", "127.0.0.1"),
            port=_port_int("LONGCTX_PORT", 8080),
            backend_url=os.getenv("LONGCTX_BACKEND_URL", "").rstrip("/"),
            backend_api_key=os.getenv("LONGCTX_BACKEND_API_KEY", ""),
            api_key=os.getenv("LONGCTX_API_KEY", ""),
            required_input_tokens=_positive_int(
                "LONGCTX_REQUIRED_INPUT_TOKENS", 100_000_000
            ),
            max_upload_bytes=_positive_int("LONGCTX_MAX_UPLOAD_BYTES", 64 * 1024**3),
            request_timeout_seconds=_positive_int(
                "LONGCTX_REQUEST_TIMEOUT_SECONDS", 7200
            ),
            temp_dir=os.getenv("LONGCTX_TEMP_DIR") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("LONGCTX_PORT must be between 1 and 65535")
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if self.host not in local_hosts and not self.api_key:
            raise ValueError(
                "LONGCTX_API_KEY is required when binding beyond localhost"
            )
        if not self.backend_url:
            return
        parsed = urlsplit(self.backend_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LONGCTX_BACKEND_URL must be an absolute http(s) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("LONGCTX_BACKEND_URL must not contain a query or fragment")
        if parsed.scheme != "https" and parsed.hostname not in local_hosts:
            raise ValueError("Remote LONGCTX_BACKEND_URL must use HTTPS")


def _port_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
