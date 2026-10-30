"""Capability and result contracts for a single-request long-context backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
SUPPORTED_ATTENTION_MODES = {
    "dense_full_attention",
    "sparse_attention",
    "recurrent_compressed",
}


class BackendContractError(ValueError):
    """The inference backend cannot safely serve the requested context."""


@dataclass(frozen=True)
class BackendCapabilities:
    model_id: str
    tokenizer_id: str
    context_limit_tokens: int
    supports_single_request: bool
    truncation_policy: str
    attention_mode: str
    protocol_version: int = PROTOCOL_VERSION

    @classmethod
    def from_payload(cls, value: Any) -> BackendCapabilities:
        if not isinstance(value, dict):
            raise BackendContractError("Backend capabilities must be a JSON object")
        try:
            result = cls(
                model_id=str(value["model_id"]),
                tokenizer_id=str(value["tokenizer_id"]),
                context_limit_tokens=int(value["context_limit_tokens"]),
                supports_single_request=value["supports_single_request"],
                truncation_policy=str(value["truncation_policy"]),
                attention_mode=str(value["attention_mode"]),
                protocol_version=int(value["protocol_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(
                f"Invalid backend capability manifest: {exc}"
            ) from exc
        if result.protocol_version != PROTOCOL_VERSION:
            raise BackendContractError(
                f"Unsupported backend protocol version {result.protocol_version}; expected {PROTOCOL_VERSION}"
            )
        if not result.model_id.strip() or not result.tokenizer_id.strip():
            raise BackendContractError(
                "Backend must declare non-empty model_id and tokenizer_id"
            )
        if result.context_limit_tokens <= 0:
            raise BackendContractError("context_limit_tokens must be positive")
        if (
            not isinstance(result.supports_single_request, bool)
            or not result.supports_single_request
        ):
            raise BackendContractError(
                "Backend must support one complete logical input request"
            )
        if result.truncation_policy != "reject":
            raise BackendContractError(
                "Backend truncation_policy must be 'reject'; silent truncation is unsafe"
            )
        if result.attention_mode not in SUPPORTED_ATTENTION_MODES:
            raise BackendContractError(
                "attention_mode must be one of: "
                + ", ".join(sorted(SUPPORTED_ATTENTION_MODES))
            )
        return result

    def require_context(self, required_tokens: int, output_tokens: int = 0) -> None:
        total = required_tokens + output_tokens
        if self.context_limit_tokens < total:
            raise BackendContractError(
                f"Backend advertises {self.context_limit_tokens:,} tokens, "
                f"below the required input plus output budget of {total:,}"
            )


@dataclass(frozen=True)
class InferenceResult:
    answer: str
    model_id: str
    tokenizer_id: str
    input_tokens: int
    processed_input_tokens: int
    input_sha256: str
    context_limit_tokens: int
    attention_mode: str
    truncated: bool

    @classmethod
    def from_payload(cls, value: Any) -> InferenceResult:
        if not isinstance(value, dict):
            raise BackendContractError("Inference result must be a JSON object")
        try:
            result = cls(
                answer=value["answer"],
                model_id=str(value["model_id"]),
                tokenizer_id=str(value["tokenizer_id"]),
                input_tokens=int(value["input_tokens"]),
                processed_input_tokens=int(value["processed_input_tokens"]),
                input_sha256=str(value["input_sha256"]),
                context_limit_tokens=int(value["context_limit_tokens"]),
                attention_mode=str(value["attention_mode"]),
                truncated=value["truncated"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(f"Invalid inference result: {exc}") from exc
        if not isinstance(result.answer, str):
            raise BackendContractError("Backend answer must be text")
        if not isinstance(result.truncated, bool):
            raise BackendContractError("Backend truncated field must be boolean")
        if result.input_tokens <= 0 or result.processed_input_tokens < 0:
            raise BackendContractError("Backend returned invalid token counts")
        if len(result.input_sha256) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in result.input_sha256
        ):
            raise BackendContractError("Backend returned an invalid SHA-256 digest")
        if result.attention_mode not in SUPPORTED_ATTENTION_MODES:
            raise BackendContractError("Backend returned an unknown attention_mode")
        return result

    def verify(
        self,
        *,
        digest: str,
        required_input_tokens: int,
        required_context_tokens: int,
        output_tokens: int,
        capabilities: BackendCapabilities,
    ) -> None:
        if self.truncated:
            raise BackendContractError("Backend reports that the input was truncated")
        if self.input_sha256.lower() != digest.lower():
            raise BackendContractError(
                "Backend input digest does not match the complete uploaded input"
            )
        if self.input_tokens < required_input_tokens:
            raise BackendContractError(
                f"Backend counted {self.input_tokens:,} input tokens, below the required {required_input_tokens:,}"
            )
        if self.input_tokens != self.processed_input_tokens:
            raise BackendContractError(
                "Backend did not process every input token it counted"
            )
        if self.input_tokens + output_tokens > self.context_limit_tokens:
            raise BackendContractError(
                "Backend input plus requested output exceeds its own context limit"
            )
        if self.context_limit_tokens < required_context_tokens:
            raise BackendContractError(
                "Backend result context limit is below input plus output budget"
            )
        if (
            self.model_id != capabilities.model_id
            or self.tokenizer_id != capabilities.tokenizer_id
        ):
            raise BackendContractError(
                "Backend model/tokenizer changed between preflight and inference"
            )
        if self.attention_mode != capabilities.attention_mode:
            raise BackendContractError(
                "Backend attention mode changed between preflight and inference"
            )
