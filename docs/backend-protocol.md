# Long Context Backend Protocol v1

게이트웨이는 범용 chat-completions API를 100M 입력 가능하다고 추정하지 않습니다. 모델 자체 컨텍스트, 사용 tokenizer, 전체 단일 요청 지원 및 truncation 정책을 구현자가 선언하는 이 프로토콜을 요구합니다.

## 1. Capability preflight

`GET /v1/long-context/capabilities`는 JSON 반환:

```json
{
  "protocol_version": 1,
  "model_id": "model-name-and-revision",
  "tokenizer_id": "tokenizer-name-and-revision",
  "context_limit_tokens": 100002048,
  "supports_single_request": true,
  "truncation_policy": "reject",
  "attention_mode": "sparse_attention"
}
```

`attention_mode` 값은 `dense_full_attention`, `sparse_attention`, `recurrent_compressed` 중 하나입니다. `context_limit_tokens`는 모델이 지원하는 입력+출력 총 시퀀스 한도입니다. 게이트웨이는 목표 입력 토큰과 요청된 출력 예산을 합산해 확인합니다. 누락/잘못된 manifest, 버전 불일치, 1억 토큰+출력 예산 미달, 입력 분할/잘라내기 정책이면 요청 전에 거부합니다.

Capability는 backend가 주장한 값이며 독립적으로 입증된 보증은 아닙니다. 운영자는 같은 모델/tokenizer revision과 실제 설정(모델 config, `max_model_len`, tokenizer)을 점검해야 합니다. 이 프로토콜 응답만 거짓으로 만들 수 있는 악의적 backend까지 gateway는 암호학적으로 검증할 수 없습니다.

## 2. Exact tokenizer preflight

Before model inference, gateway calls `POST /v1/long-context/tokenize` with the complete UTF-8 body and `Content-Length`, plus `X-Question-B64` and `X-Input-SHA256`. The adapter must use the **same tokenizer, model revision, and system/chat wrapper** that the inference path will use, and return:

Reference adapter: `longctx100m-vllm-adapter` implements this protocol for a configured vLLM `/v1/completions` server. It uses the pinned Hugging Face tokenizer, then checks vLLM's `usage.prompt_tokens` against that count. It rejects any mismatch. vLLM still needs a model and runtime that actually support the declared sequence length.

```json
{"input_tokens": 100000000, "input_sha256": "<same 64 hex chars>"}
```

The count includes every token the model will receive for the uploaded document and user question, except the requested generation budget. If the count is below the configured minimum or count + output budget exceeds the preflight context limit, the gateway refuses to run model inference. This makes the actual tokenizer decision explicit instead of trusting a byte-size guess.

## 3. Single-request inference

`POST /v1/long-context/infer`

- Body: UTF-8 원본 입력 전체, `application/octet-stream`, `Content-Length` 필수. Gateway가 저장한 파일 바이트를 그대로 보냅니다.
- Headers:
  - `X-Question-B64`: 질문을 UTF-8 → Base64 URL-safe 인코딩한 값
  - `X-Input-SHA256`: body 전체의 64자리 SHA-256
  - `X-Required-Input-Tokens`: 최소 입력 token 수 (기본 100,000,000)
  - `X-Required-Context-Tokens`: 입력 목표 + max output token budget
  - `X-Max-Output-Tokens`: generation 상한
  - `X-Capability-Model-ID`: preflight에서 읽은 model ID (Base64 URL-safe)
- `Authorization: Bearer ...`는 `LONGCTX_BACKEND_API_KEY` 설정 시 전달됩니다.

입력 한 건 전체를 처리해야 합니다. 내부 chunked prefill/분산 실행은 가능하지만, 입력을 검색된 일부 청크/별도 에이전트 요약으로 바꾸거나 묵시적으로 자르면 안 됩니다. 모델에 따라 attention 구현은 명시되며, sparse/recurrent는 dense-full attention과 같은 의미가 아닙니다.

## 4. Required response

성공한 응답은 JSON이어야 합니다.

```json
{
  "answer": "generated response",
  "model_id": "model-name-and-revision",
  "tokenizer_id": "tokenizer-name-and-revision",
  "input_tokens": 100000000,
  "processed_input_tokens": 100000000,
  "input_sha256": "<64 hex chars, equal to request body digest>",
  "context_limit_tokens": 100002048,
  "attention_mode": "sparse_attention",
  "truncated": false
}
```

Gateway는 SHA-256, model/tokenizer/mode의 preflight 일치, `input_tokens == processed_input_tokens`, 목표 입력 길이 이상, context limit, `truncated == false`를 확인합니다. 위 검증이 실패하면 답변은 클라이언트에게 성공 결과로 노출되지 않습니다.

## 5. Model adapter duties

추론 엔진 어댑터는 다음을 책임집니다.

1. 전송된 body의 모든 바이트를 받은 뒤 SHA-256을 재계산하고 헤더와 비교.
2. 명시된 `tokenizer_id`에 해당하는 tokenizer로 입력 **및 실제 system/chat wrapper 토큰**을 계산.
3. token 길이가 input 요구량 미만이거나 input+output이 모델 한도를 넘으면 `truncation_policy=reject`로 요청 거부.
4. 입력 전체를 한 요청의 모델 execution graph에 포함. 내부 prefill를 여러 구간으로 스케줄해도 입력을 생략하지 않음.
5. 모든 입력 token이 처리됐을 때만 결과를 반환하며 양쪽 token count/digest를 응답에 기록.
6. 선택된 attention 방식, 모델 및 tokenizer revision을 응답에 명시.

단순히 `max_model_len`을 올리거나 기존 32K/128K 모델에 RoPE factor를 설정한 것은 100M 길이의 학습 적응, 입력 처리 가능성, 정확도 인증이 아닙니다.
