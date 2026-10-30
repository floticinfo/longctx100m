"""Web gateway for a single logical long-context inference request."""

from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import hmac
import json
import logging
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .backend import BackendClient, BackendError
from .settings import Settings

log = logging.getLogger("longctx100m")
CHUNK = 1024 * 1024
ACTIVE_REQUESTS = threading.BoundedSemaphore(2)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, settings):
        self.settings, self.backend = settings, BackendClient(settings)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: Server
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.client_address[0], fmt % args)

    def do_GET(self):
        if self.path == "/healthz":
            self.json(200, {"status": "ok"})
            return
        if self.path == "/api/capabilities":
            if not self.authorized():
                self.json(401, {"error": "authentication required"})
                return
            try:
                c = self.server.backend.capabilities()
            except BackendError as e:
                self.json(e.status_code, {"error": str(e)})
                return
            self.json(
                200,
                {
                    "required_input_tokens": self.server.settings.required_input_tokens,
                    "backend": {
                        "model_id": c.model_id,
                        "tokenizer_id": c.tokenizer_id,
                        "context_limit_tokens": c.context_limit_tokens,
                        "attention_mode": c.attention_mode,
                        "supports_single_request": c.supports_single_request,
                        "truncation_policy": c.truncation_policy,
                    },
                },
            )
            return
        if self.path != "/":
            self.json(404, {"error": "not found"})
            return
        try:
            b = Path(__file__).with_name("web").joinpath("index.html").read_bytes()
        except OSError:
            self.json(500, {"error": "web UI unavailable"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(b)

    def authorized(self):
        key = self.server.settings.api_key
        header = self.headers.get("Authorization", "")
        return not key or (
            header.startswith("Bearer ") and hmac.compare_digest(header[7:], key)
        )

    def json(self, status, value):
        b = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        if self.path != "/api/infer":
            self.json(404, {"error": "not found"})
            return
        self.infer()

    def infer(self):
        if not self.authorized():
            self.json(401, {"error": "authentication required"})
            return
        if not self.server.backend.configured:
            self.json(503, {"error": "inference backend is not configured"})
            return
        if self.headers.get("Transfer-Encoding"):
            self.json(411, {"error": "Content-Length is required"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.json(411, {"error": "valid Content-Length is required"})
            return
        if length <= 0:
            self.json(400, {"error": "input file is empty"})
            return
        if length > self.server.settings.max_upload_bytes:
            self.json(413, {"error": "upload exceeds configured byte limit"})
            return
        if self.headers.get("Content-Encoding", "identity").lower() != "identity":
            self.json(415, {"error": "compressed upload unsupported"})
            return
        if not self.headers.get("Content-Type", "").startswith("text/"):
            self.json(415, {"error": "upload must be text"})
            return
        try:
            encoded = self.headers.get("X-Question-B64", "")
            if not encoded or len(encoded) > 180000:
                raise ValueError("question missing or too large")
            raw = base64.b64decode(
                encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
            )
            if len(raw) > 128 * 1024:
                raise ValueError("question too large")
            question = raw.decode("utf-8").strip()
            output_tokens = int(self.headers.get("X-Max-Output-Tokens", "2048"))
            if not question or not 1 <= output_tokens <= 32768:
                raise ValueError("invalid question or output limit")
        except (binascii.Error, UnicodeDecodeError, ValueError) as e:
            self.json(400, {"error": str(e)})
            return
        if not ACTIVE_REQUESTS.acquire(blocking=False):
            self.json(
                429, {"error": "long-context inference capacity is busy; retry later"}
            )
            return
        path = None
        try:
            digest, left, decoder = (
                hashlib.sha256(),
                length,
                codecs.getincrementaldecoder("utf-8")(),
            )
            with tempfile.NamedTemporaryFile(
                "wb",
                prefix="longctx-input-",
                suffix=".txt",
                dir=self.server.settings.temp_dir,
                delete=False,
            ) as f:
                path = Path(f.name)
                while left:
                    block = self.rfile.read(min(CHUNK, left))
                    if not block:
                        raise EOFError("upload ended before Content-Length")
                    f.write(block)
                    digest.update(block)
                    decoder.decode(block, final=False)
                    left -= len(block)
                decoder.decode(b"", final=True)
            caps = self.server.backend.capabilities(output_tokens=output_tokens)
            input_tokens = self.server.backend.count_tokens(
                source=path,
                content_length=length,
                digest=digest.hexdigest(),
                question=question,
            )
            total_tokens = input_tokens + output_tokens
            if input_tokens < self.server.settings.required_input_tokens:
                self.json(
                    413,
                    {
                        "error": f"input contains {input_tokens:,} tokens; at least {self.server.settings.required_input_tokens:,} are required"
                    },
                )
                return
            if total_tokens > caps.context_limit_tokens:
                self.json(
                    413,
                    {
                        "error": f"input plus output budget is {total_tokens:,} tokens; backend limit is {caps.context_limit_tokens:,}"
                    },
                )
                return
            result = self.server.backend.infer(
                source=path,
                content_length=length,
                question=question,
                digest=digest.hexdigest(),
                input_tokens=input_tokens,
                capabilities=caps,
                max_output_tokens=output_tokens,
            )
            self.json(
                200,
                {
                    "answer": result.answer,
                    "model_id": result.model_id,
                    "tokenizer_id": result.tokenizer_id,
                    "input_tokens": result.input_tokens,
                    "processed_input_tokens": result.processed_input_tokens,
                    "input_sha256": result.input_sha256,
                    "context_limit_tokens": result.context_limit_tokens,
                    "attention_mode": result.attention_mode,
                    "truncated": result.truncated,
                },
            )
        except BackendError as e:
            self.json(e.status_code, {"error": str(e)})
        except (OSError, EOFError, UnicodeDecodeError, ValueError, binascii.Error) as e:
            self.json(400, {"error": f"upload failed: {e}"})
        finally:
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    log.exception("failed to remove staged upload")
            ACTIVE_REQUESTS.release()


def serve(settings: Settings):
    server = Server((settings.host, settings.port), settings)
    log.info(
        "Long-context web gateway listening on %s:%s", settings.host, settings.port
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
