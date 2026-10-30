# 배포 가이드

> **운영 구성/시크릿**: 실제 배포 시크릿, 키 회전 정책, 프로비저닝/업그레이드/장애 runbook은
> **private 저장소** `floticinfo/longctx100m-deploy`에 보관합니다. 이 문서는 공개 토폴로지와
> 실행 명령만 담습니다.

## 구성 요소

- **브라우저 UI + gateway:** 업로드를 1 MiB 블록으로 받아 임시 파일에 기록하고 전체 SHA-256을 계산합니다. 본문 전체는 표준 입력 방식으로 추론 백엔드에 전달합니다.
- **자체 decoder engine:** `longctx100m.model` + sequence-sharded causal ring attention. `longctx100m-train`은 raw uint32 token corpus를 받아 context-parallel 학습 entry를 실행합니다. Gloo correctness test 외 multi-node CUDA/NCCL 성능은 target cluster에서 별도 검증이 필요합니다.
- **vLLM adapter:** protocol 호환 backend를 제공하는 별도 경로입니다. 자체 engine의 100M 지원을 대신하거나 임의 모델을 100M으로 확장하지 않습니다.

## Gateway 시작

```bash
python -m pip install -e .
LONGCTX_BACKEND_URL=https://inference.example.internal \
LONGCTX_API_KEY=replace-with-client-secret \
LONGCTX_BACKEND_API_KEY=replace-with-backend-secret \
LONGCTX_MAX_UPLOAD_BYTES=68719476736 \
LONGCTX_REQUIRED_INPUT_TOKENS=100000000 \
longctx100m-serve
```

환경 변수:

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `LONGCTX_HOST` | `127.0.0.1` | 수신 주소. 원격 노출 시 reverse proxy/TLS와 인증을 반드시 적용 |
| `LONGCTX_PORT` | `8080` | HTTP 포트 |
| `LONGCTX_BACKEND_URL` | 미설정 | 호환 backend의 base URL |
| `LONGCTX_BACKEND_API_KEY` | 빈 값 | Backend Authorization bearer key |
| `LONGCTX_API_KEY` | 빈 값 | 브라우저 client에서 요구할 gateway key |
| `LONGCTX_REQUIRED_INPUT_TOKENS` | `100000000` | 최소 input token 수 |
| `LONGCTX_MAX_UPLOAD_BYTES` | `68719476736` | 최대 파일 바이트. reverse proxy도 같은/더 큰 body limit 설정 필요 |
| `LONGCTX_REQUEST_TIMEOUT_SECONDS` | `7200` | backend 연결 및 inference timeout |
| `LONGCTX_TEMP_DIR` | OS 임시 디렉터리 | 업로드 staging 디렉터리. 충분한 임시 공간/암호화 저장소 권장 |

기본적으로 `127.0.0.1`에만 바인딩합니다. 네트워크 서비스로 노출 시 `LONGCTX_API_KEY`, TLS 종료 reverse proxy, 로그인/접근통제, rate limit, 동시 작업 제한을 운영자가 구성해야 합니다. 이 간결한 gateway 자체에는 다중 사용자 identity, quotas, job queue, HA, billing 기능이 없습니다.

## vLLM adapter 설정

프로토콜 어댑터를 설치합니다.

```bash
python -m pip install -e '.[vllm-adapter]'
```

별도 inference 환경에서 **실제 장문맥 모델**을 vLLM server로 준비한 다음 어댑터를 실행합니다.

```bash
LONGCTX_VLLM_URL=http://127.0.0.1:8000 \
LONGCTX_MODEL_ID=org/model@pinned-revision \
LONGCTX_TOKENIZER_ID=org/tokenizer@pinned-revision \
LONGCTX_MODEL_CONTEXT_LIMIT=100002048 \
LONGCTX_REQUIRED_INPUT_TOKENS=100000000 \
LONGCTX_ATTENTION_MODE=sparse_attention \
LONGCTX_BACKEND_API_KEY=replace-with-shared-secret \
LONGCTX_ADAPTER_HOST=0.0.0.0 \
LONGCTX_ADAPTER_PORT=8090 \
longctx100m-vllm-adapter
```

이 값들은 운영자가 실제 모델/서버 설정을 확인해 넣습니다. `LONGCTX_MODEL_CONTEXT_LIMIT`과 `LONGCTX_ATTENTION_MODE`는 **선언값이지 모델 능력 확장 스위치가 아닙니다.** vLLM은 `max_model_len` 및 context extension 옵션을 제공하지만, 실제 모델 context training과 runtime 경로가 해당 길이를 지원해야 합니다. 어댑터는 `transformers` tokenizer로 정확한 prompt를 계수하고 vLLM `/v1/completions`의 `usage.prompt_tokens`와 비교합니다. 두 tokenizer/wrapper 경로가 다르면 요청을 실패시킵니다. 현재 upstream vLLM의 모든 모델/API 조합이 100M context를 지원한다고 가정하지 마세요.

Gateway는 [`backend-protocol.md`](backend-protocol.md)의 v1 contract를 사용합니다. `LONGCTX_BACKEND_API_KEY`를 gateway 및 adapter에 동일하게 설정합니다. `LONGCTX_VLLM_API_KEY`는 vLLM server에 인증이 설정된 경우 추가합니다. 자체 decoder 모델 정의/학습/분산 사전 채움은 [`engine-design.md`](engine-design.md)를 참조하세요.

Gateway 관리 화면 `GET /api/capabilities`는 연결 모델, tokenizer, attention 및 context limit을 반환합니다.

UI에서 `.txt`, `.md`, `.json`, `.csv`, `.log` 같은 UTF-8 텍스트 파일 한 개와 질의를 전송합니다. Gateway는 파일 바이트를 메모리에 전부 올리지 않고 디스크로 스트리밍합니다. 처리 종료 후 임시 파일을 지우며, 로깅에는 파일/질의/답변 내용을 기록하지 않습니다. 파일을 처리하는 동안에는 임시 저장소에 평문으로 존재할 수 있습니다. 민감 자료에는 암호화된 임시 볼륨, 제한된 접근, 안전한 제거가 필요합니다.

## GPU/노드 병렬화 메모

추론 backend는 모델과 커널에서 지원하는 병렬 전략을 선택합니다. vLLM 문서의 tensor/pipeline parallelism은 모델 weights를 여러 GPU와 노드에 배치하는 방법을 설명합니다. 긴 입력의 attention 자체를 나누는 context/sequence parallelism은 모델 및 런타임 지원이 별도로 필요합니다. 여러 노드 간 GPU memory를 효과적으로 교환하려면 높은 대역폭/낮은 지연의 GPU-direct 네트워크가 중요합니다. 백엔드의 실제 지원 토큰 길이와 throughput은 target 모델을 사용한 prefill test로 확인해야 합니다.

참고: [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), [vLLM Context Extension](https://docs.vllm.ai/en/latest/features/context_extension/). 일반적인 RoPE scaling/context extension 문서는 특정 모델 설정을 넓힐 수 있다고 설명할 뿐, 임의의 100M 모델 입력이나 품질을 보증하지 않습니다.
