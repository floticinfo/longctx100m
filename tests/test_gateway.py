from __future__ import annotations

import hashlib
import http.client
import json
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from longctx100m.application import Server
from longctx100m.backend import BackendClient, BackendError
from longctx100m.contract import (
    BackendCapabilities,
    BackendContractError,
    InferenceResult,
)
from longctx100m.settings import Settings

MODEL, TOKENIZER = "research-100m", "research-tokenizer-v1"


class ContractTests(unittest.TestCase):
    def caps(self, **changes):
        v = {
            "protocol_version": 1,
            "model_id": MODEL,
            "tokenizer_id": TOKENIZER,
            "context_limit_tokens": 100002048,
            "supports_single_request": True,
            "truncation_policy": "reject",
            "attention_mode": "sparse_attention",
        }
        v.update(changes)
        return BackendCapabilities.from_payload(v)

    def result(self, digest, **changes):
        v = {
            "answer": "verified",
            "model_id": MODEL,
            "tokenizer_id": TOKENIZER,
            "input_tokens": 100000000,
            "processed_input_tokens": 100000000,
            "input_sha256": digest,
            "context_limit_tokens": 100002048,
            "attention_mode": "sparse_attention",
            "truncated": False,
        }
        v.update(changes)
        return InferenceResult.from_payload(v)

    def test_reject_unsafe_backend(self):
        with self.assertRaises(BackendContractError):
            self.caps(supports_single_request=False)
        with self.assertRaises(BackendContractError):
            self.caps(truncation_policy="silent")

    def test_reserve_output_context(self):
        with self.assertRaisesRegex(BackendContractError, "input plus output"):
            self.caps(context_limit_tokens=100001000).require_context(100000000, 2048)

    def test_result_requires_digest_all_tokens_and_no_truncation(self):
        c = self.caps()
        self.result("a" * 64).verify(
            digest="a" * 64,
            required_input_tokens=100000000,
            required_context_tokens=100002048,
            output_tokens=2048,
            capabilities=c,
        )
        for kwargs, pattern in [
            ({"input_sha256": "b" * 64}, "digest"),
            (
                {"input_tokens": 99999999, "processed_input_tokens": 99999999},
                "below the required",
            ),
            ({"truncated": True}, "truncated"),
        ]:
            with self.assertRaisesRegex(BackendContractError, pattern):
                self.result("a" * 64, **kwargs).verify(
                    digest="a" * 64,
                    required_input_tokens=100000000,
                    required_context_tokens=100002048,
                    output_tokens=2048,
                    capabilities=c,
                )

    def test_rejects_remote_plain_http_backend(self):
        with self.assertRaises(BackendError):
            BackendClient(Settings(backend_url="http://remote.example")).capabilities()


class FakeBackend(BaseHTTPRequestHandler):
    upload = b""
    tokenize_upload = b""
    question = ""
    truncated = False
    force_long_input = True
    token_count = 100000000
    infer_headers = None

    def log_message(self, *args):
        pass

    def reply(self, status, value):
        b = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.reply(
            200,
            {
                "protocol_version": 1,
                "model_id": MODEL,
                "tokenizer_id": TOKENIZER,
                "context_limit_tokens": 10001000000,
                "supports_single_request": True,
                "truncation_policy": "reject",
                "attention_mode": "dense_full_attention",
            },
        )

    def do_POST(self):
        import base64

        body = self.rfile.read(int(self.headers["Content-Length"]))
        d = hashlib.sha256(body).hexdigest()
        if self.path.endswith("/tokenize"):
            type(self).tokenize_upload = body
            count = type(self).token_count if type(self).force_long_input else 5
            self.reply(200, {"input_tokens": count, "input_sha256": d})
            return
        type(self).infer_headers = dict(self.headers)
        type(self).upload = body
        type(self).question = base64.urlsafe_b64decode(
            self.headers["X-Question-B64"] + "=="
        ).decode()
        n = 99999999 if type(self).truncated else 100000000
        self.reply(
            200,
            {
                "answer": "integration result",
                "model_id": MODEL,
                "tokenizer_id": TOKENIZER,
                "input_tokens": 100000000,
                "processed_input_tokens": n,
                "input_sha256": d,
                "context_limit_tokens": 10001000000,
                "attention_mode": "dense_full_attention",
                "truncated": type(self).truncated,
            },
        )


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackend)
        cls.bt = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.bt.start()
        cls.gateway = Server(
            ("127.0.0.1", 0),
            Settings(
                port=0,
                backend_url=f"http://127.0.0.1:{cls.backend.server_port}",
                required_input_tokens=100000000,
                max_upload_bytes=1024 * 1024,
                request_timeout_seconds=10,
            ),
        )
        cls.gt = threading.Thread(target=cls.gateway.serve_forever, daemon=True)
        cls.gt.start()

    @classmethod
    def tearDownClass(cls):
        cls.gateway.shutdown()
        cls.gateway.server_close()
        cls.backend.shutdown()
        cls.backend.server_close()
        cls.gt.join(2)
        cls.bt.join(2)

    def request(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection(
            "127.0.0.1", self.gateway.server_port, timeout=10
        )
        c.request(method, path, body, headers or {})
        r = c.getresponse()
        b = r.read()
        s = r.status
        c.close()
        return s, b

    def test_health_and_ui(self):
        s, b = self.request("GET", "/healthz")
        self.assertEqual(s, 200)
        self.assertIn(b"ok", b)
        s, b = self.request("GET", "/")
        self.assertEqual(s, 200)
        self.assertIn(b"Single-request long-context inference", b)
        self.assertNotIn(b"/api/preflight", b)

    def test_upload_forwards_entire_body_and_validates_attestation(self):
        FakeBackend.truncated = False
        FakeBackend.force_long_input = True
        text = b"whole-file-test"
        s, b = self.request(
            "POST",
            "/api/infer",
            text,
            {
                "Content-Type": "text/plain",
                "Content-Length": str(len(text)),
                "X-Question-B64": "V2hhdCBpcyB0aGUgZmlsZT8",
            },
        )
        p = json.loads(b)
        self.assertEqual(s, 200, p)
        self.assertEqual(FakeBackend.upload, text)
        self.assertEqual(FakeBackend.question, "What is the file?")
        self.assertEqual(p["input_sha256"], hashlib.sha256(text).hexdigest())
        self.assertEqual(FakeBackend.tokenize_upload, text)
        self.assertEqual(
            FakeBackend.infer_headers["X-Required-Input-Tokens"], "100000000"
        )
        self.assertEqual(
            FakeBackend.infer_headers["X-Minimum-Input-Tokens"], "100000000"
        )
        self.assertEqual(
            FakeBackend.infer_headers["X-Required-Context-Tokens"], "100002048"
        )

    def test_gateway_forwards_actual_preflight_count_not_only_minimum(self):
        FakeBackend.force_long_input = True
        FakeBackend.token_count = 100_123_456
        FakeBackend.truncated = False
        s, b = self.request(
            "POST",
            "/api/infer",
            b"whole-file",
            {
                "Content-Type": "text/plain",
                "Content-Length": "10",
                "X-Question-B64": "cQ",
            },
        )
        self.assertEqual(s, 200, json.loads(b))
        self.assertEqual(
            FakeBackend.infer_headers["X-Required-Input-Tokens"], "100123456"
        )
        self.assertEqual(
            FakeBackend.infer_headers["X-Required-Context-Tokens"], "100125504"
        )
        FakeBackend.token_count = 100000000

    def test_incomplete_backend_result_is_rejected(self):
        FakeBackend.truncated = True
        s, b = self.request(
            "POST",
            "/api/infer",
            b"x",
            {
                "Content-Type": "text/plain",
                "Content-Length": "1",
                "X-Question-B64": "cQ",
            },
        )
        self.assertEqual(s, 502)
        self.assertIn(b"truncated", b)

    def test_rejects_short_input_before_generation(self):
        FakeBackend.truncated = False
        FakeBackend.force_long_input = False
        s, b = self.request(
            "POST",
            "/api/infer",
            b"short",
            {
                "Content-Type": "text/plain",
                "Content-Length": "5",
                "X-Question-B64": "cQ",
            },
        )
        self.assertEqual(s, 413)
        self.assertIn(b"at least 100,000,000", b)

    def test_requires_gateway_key_when_exposed_offhost(self):
        with self.assertRaisesRegex(ValueError, "LONGCTX_API_KEY"):
            Settings(host="0.0.0.0").validate()
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            Settings(backend_url="http://backend.example").validate()

    def test_upload_limit(self):
        s, _ = self.request(
            "POST",
            "/api/infer",
            b"x" * 10,
            {
                "Content-Type": "text/plain",
                "Content-Length": "1048577",
                "X-Question-B64": "cQ",
            },
        )
        self.assertEqual(s, 413)


if __name__ == "__main__":
    unittest.main()
