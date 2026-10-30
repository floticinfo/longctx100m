"""HTTP client for the explicit 100M single-request backend protocol."""

from __future__ import annotations

import base64
import http.client
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .contract import BackendCapabilities, BackendContractError, InferenceResult
from .settings import Settings


class BackendError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class BackendClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base = urlsplit(settings.backend_url) if settings.backend_url else None

    @property
    def configured(self) -> bool:
        return self.base is not None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.backend_api_key:
            headers["Authorization"] = f"Bearer {self.settings.backend_api_key}"
        return headers

    def capabilities(self, *, output_tokens: int = 0) -> BackendCapabilities:
        if not self.base:
            raise BackendError("Inference backend is not configured", status_code=503)
        if self.base.scheme != "https" and self.base.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise BackendError(
                "Remote inference backends must use HTTPS", status_code=503
            )
        url = self._url("/v1/long-context/capabilities")
        request = Request(url, headers=self._headers(), method="GET")
        try:
            with urlopen(
                request, timeout=min(self.settings.request_timeout_seconds, 30)
            ) as response:
                payload = json.loads(response.read(1024 * 1024 + 1))
        except HTTPError as exc:
            raise BackendError(
                f"Backend capabilities request failed with HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"Could not read backend capabilities: {exc}") from exc
        try:
            capabilities = BackendCapabilities.from_payload(payload)
            capabilities.require_context(
                self.settings.required_input_tokens, output_tokens
            )
        except BackendContractError as exc:
            raise BackendError(str(exc), status_code=503) from exc
        return capabilities

    def count_tokens(
        self, *, source: Path, content_length: int, digest: str, question: str
    ) -> int:
        if not self.base:
            raise BackendError("Inference backend is not configured", status_code=503)
        if self.base.scheme != "https" and self.base.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise BackendError(
                "Remote inference backends must use HTTPS", status_code=503
            )
        connection_type = (
            http.client.HTTPSConnection
            if self.base.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            self.base.hostname,
            self.base.port,
            timeout=self.settings.request_timeout_seconds,
        )
        path = self._path("/v1/long-context/tokenize")
        headers = {
            **self._headers(),
            "Content-Type": "application/octet-stream",
            "Content-Length": str(content_length),
            "X-Question-B64": base64.urlsafe_b64encode(question.encode("utf-8")).decode(
                "ascii"
            ),
            "X-Input-SHA256": digest,
        }
        try:
            connection.putrequest("POST", path)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            sent = 0
            with source.open("rb") as body:
                while block := body.read(1024 * 1024):
                    connection.send(block)
                    sent += len(block)
            if sent != content_length:
                raise BackendError("Staged input size changed before tokenization")
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
            if response.status >= 400:
                raise BackendError(
                    f"Tokenizer backend returned HTTP {response.status}: {raw[:2000].decode(errors='replace')}"
                )
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise BackendError("Tokenizer response must be a JSON object")
            if payload.get("input_sha256", "").lower() != digest.lower():
                raise BackendError(
                    "Tokenizer response digest does not match uploaded input"
                )
            count = payload.get("input_tokens")
            if not isinstance(count, int) or count <= 0:
                raise BackendError("Tokenizer backend returned an invalid token count")
            return count
        except BackendError:
            raise
        except (
            OSError,
            http.client.HTTPException,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            raise BackendError(f"Tokenizer request failed: {exc}") from exc
        finally:
            connection.close()

    def infer(
        self,
        *,
        source: Path,
        content_length: int,
        question: str,
        digest: str,
        input_tokens: int,
        capabilities: BackendCapabilities,
        max_output_tokens: int,
    ) -> InferenceResult:
        if input_tokens < self.settings.required_input_tokens:
            raise BackendError(
                "Preflight input token count is below the required minimum",
                status_code=413,
            )
        capabilities.require_context(input_tokens, max_output_tokens)
        if not self.base:
            raise BackendError("Inference backend is not configured", status_code=503)
        connection_type = (
            http.client.HTTPSConnection
            if self.base.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            self.base.hostname,
            self.base.port,
            timeout=self.settings.request_timeout_seconds,
        )
        encoded_question = base64.urlsafe_b64encode(question.encode("utf-8")).decode(
            "ascii"
        )
        path = self._path("/v1/long-context/infer")
        headers = {
            **self._headers(),
            "Content-Type": "application/octet-stream",
            "Content-Length": str(content_length),
            "X-Question-B64": encoded_question,
            "X-Input-SHA256": digest,
            "X-Required-Input-Tokens": str(input_tokens),
            "X-Minimum-Input-Tokens": str(self.settings.required_input_tokens),
            "X-Max-Output-Tokens": str(max_output_tokens),
            "X-Required-Context-Tokens": str(input_tokens + max_output_tokens),
            "X-Capability-Model-ID": base64.urlsafe_b64encode(
                capabilities.model_id.encode()
            ).decode(),
        }
        try:
            connection.putrequest("POST", path)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            sent = 0
            with source.open("rb") as body:
                while block := body.read(1024 * 1024):
                    connection.send(block)
                    sent += len(block)
            if sent != content_length:
                raise BackendError(
                    "Staged input size changed before backend forwarding"
                )
            response = connection.getresponse()
            raw = response.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise BackendError("Backend response exceeded 16 MiB")
            if response.status >= 400:
                detail = raw.decode("utf-8", errors="replace")[:2000]
                raise BackendError(
                    f"Inference backend returned HTTP {response.status}: {detail}",
                    status_code=502,
                )
            try:
                result = InferenceResult.from_payload(json.loads(raw))
                result.verify(
                    digest=digest,
                    required_input_tokens=self.settings.required_input_tokens,
                    required_context_tokens=input_tokens + max_output_tokens,
                    output_tokens=max_output_tokens,
                    capabilities=capabilities,
                )
            except (json.JSONDecodeError, BackendContractError) as exc:
                raise BackendError(
                    f"Backend response failed verification: {exc}"
                ) from exc
            return result
        except BackendError:
            raise
        except (OSError, http.client.HTTPException, TimeoutError) as exc:
            raise BackendError(f"Inference backend request failed: {exc}") from exc
        finally:
            connection.close()

    def _url(self, suffix: str) -> str:
        assert self.base
        return f"{self.settings.backend_url}{suffix}"

    def _path(self, suffix: str) -> str:
        assert self.base
        base_path = self.base.path.rstrip("/")
        return f"{base_path}{suffix}" or "/"
