from __future__ import annotations

import base64
import hashlib
import http.client
import json
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longctx100m.vllm_adapter import AdapterConfig, AdapterServer, compose_prompt


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.reported_count = None
        self.count_value = 100_000_000

    def count(self, document, question):
        return self.count_value

    def generate(self, prompt, max_output_tokens):
        self.calls.append((prompt, max_output_tokens))
        return "answer", self.reported_count or self.count_value


class VllmAdapterTests(unittest.TestCase):
    def test_prompt_contains_full_document_and_query(self):
        prompt = compose_prompt("whole document body", "analyze it")
        self.assertIn("whole document body", prompt)
        self.assertIn("analyze it", prompt)

    def test_config_requires_explicit_model_context_and_attention(self):
        config = AdapterConfig()
        config.model_id = "model@revision"
        config.tokenizer_id = "tokenizer@revision"
        config.attention_mode = "sparse_attention"
        config.context_limit = 100_000_001
        config.input_limit = 100_000_000
        config.validate()
        config.context_limit = 100_000_000
        with self.assertRaises(ValueError):
            config.validate()

    @classmethod
    def setUpClass(cls):
        cls.config = AdapterConfig()
        cls.config.host = "127.0.0.1"
        cls.config.port = 0
        cls.config.model_id = "test-model@rev"
        cls.config.tokenizer_id = "test-tokenizer@rev"
        cls.config.context_limit = 100_010_000
        cls.config.input_limit = 100_000_000
        cls.config.max_body_bytes = 1024 * 1024
        cls.config.max_output = 4096
        cls.config.attention_mode = "sparse_attention"
        cls.config.gateway_key = "secret"
        cls.engine = FakeEngine()
        cls.server = AdapterServer(("127.0.0.1", 0), cls.config, cls.engine)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(2)

    def call(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=10
        )
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        status = response.status
        conn.close()
        return status, json.loads(raw)

    def headers(self, body, question="What?"):
        return {
            "Authorization": "Bearer secret",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
            "X-Input-SHA256": hashlib.sha256(body).hexdigest(),
            "X-Question-B64": base64.urlsafe_b64encode(question.encode())
            .decode()
            .rstrip("="),
        }

    def test_tokenizer_preflight_checks_digest(self):
        body = b"entire document"
        status, response = self.call(
            "POST", "/v1/long-context/tokenize", body, self.headers(body)
        )
        self.assertEqual(status, 200, response)
        self.assertEqual(response["input_tokens"], self.engine.count_value)
        bad = self.headers(body)
        bad["X-Input-SHA256"] = "0" * 64
        status, _ = self.call("POST", "/v1/long-context/tokenize", body, bad)
        self.assertEqual(status, 400)

    def test_inference_validates_all_tokens_and_forwards_whole_prompt(self):
        self.engine.count_value = 100_000_000
        self.engine.reported_count = 100_000_000
        body = b"entire document"
        headers = self.headers(body)
        headers.update(
            {
                "X-Required-Input-Tokens": "100000000",
                "X-Required-Context-Tokens": "100000100",
                "X-Max-Output-Tokens": "100",
            }
        )
        status, response = self.call("POST", "/v1/long-context/infer", body, headers)
        self.assertEqual(status, 200, response)
        self.assertEqual(response["answer"], "answer")
        self.assertIn("entire document", self.engine.calls[-1][0])
        self.assertEqual(self.engine.calls[-1][1], 100)

    def test_inference_rejects_context_overflow_before_generation(self):
        body = b"doc"
        headers = self.headers(body)
        headers.update(
            {
                "X-Required-Input-Tokens": "100010000",
                "X-Minimum-Input-Tokens": "100000000",
                "X-Required-Context-Tokens": "100010100",
                "X-Max-Output-Tokens": "100",
            }
        )
        calls = len(self.engine.calls)
        self.engine.count_value = 100_010_000
        self.engine.reported_count = None
        headers["X-Required-Context-Tokens"] = "100010100"
        status, _ = self.call("POST", "/v1/long-context/infer", body, headers)
        self.assertEqual(status, 413)
        self.assertEqual(len(self.engine.calls), calls)

    def test_requires_backend_authentication(self):
        body = b"x"
        headers = self.headers(body)
        headers.pop("Authorization")
        status, _ = self.call("POST", "/v1/long-context/tokenize", body, headers)
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main()
