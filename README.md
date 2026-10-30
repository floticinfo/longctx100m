# Long Context 100M — Distributed Decoder and Gateway

**목표:** dense causal attention으로 1억 token 입력 전체를 실제 한 모델 요청에서 처리하는 학습/추론 stack을 구현한다. UI는 하나의 파일과 질문만 전송하고, engine은 sequence/context parallel ring attention으로 KV shard를 rank 사이에 전달한다. 검색·분할 에이전트 요약으로 입력을 대체하지 않는다.

## 저장소 구조

- **`floticinfo/longctx100m`** (public): 전체 소스 코드, 문서, 테스트, parity 검증 스크립트.
- **`floticinfo/longctx100m-deploy`** (private): 운영 배포 시크릿 템플릿, 키 회전 정책, 프로비저닝/업그레이드/장애 runbook. 배포 시 이 저장소를 참조하세요.

프로젝트에는 자체 decoder 모델, exact causal ring attention과 custom backward, optional Triton tiled kernel, 학습 진입점, CUDA 분산 prefill reference, 브라우저 gateway 및 vLLM adapter가 있다. 각 기능 범위와 아직 필요한 long-sequence 학습/성능 검증은 [`docs/engine-design.md`](docs/engine-design.md)에 적었다.

## 실행

Python 3.10+에서 gateway를 설치하고 실행합니다.

```bash
python -m pip install -e .
longctx100m-serve
```

Engine 의존성과 실행 경로는 다음과 같습니다.

```bash
python -m pip install -e '.[engine]'
longctx100m-train --help
longctx100m-infer --help
longctx100m-capacity --help
longctx100m-import-llama --help
```


Linux/CUDA/Triton kernel 개발은 `python -m pip install -e '.[kernel]'`로 선택 설치합니다. 모델/커널/학습 범위는 [engine design](docs/engine-design.md), gateway 배포는 [`docs/deployment.md`](docs/deployment.md)를 참조하세요. 브라우저 UI는 `http://127.0.0.1:8080`입니다.


## Rust 기반 전환 (병행 단계)

Python 경로를 rollback으로 보존한 채 Rust core/CLI·opt-in HTTP gateway·opt-in Llama importer·Rust model/training/inference runtime(candle)을 추가했습니다. Rust/Python importer는 cached pinned SmolLM2-135M에서 272 tensor의 key/shape/dtype/bytes와 manifest를 일치시켰고, Rust model은 같은 checkpoint를 로드해 forward logits(<1.5e-6), 학습 loss(<1e-3), resume 연속성, 2-rank ring attention forward(<1.7e-7)/backward(<3.5e-7), greedy 추론이 Python과 일치합니다(`scripts/verify_model_parity.py`). 하지만 실제 CUDA/NCCL transport, Triton, Linux CI는 아직 이행하지 않아 전체 엔진이 Rust로 완전 전환된 것은 아닙니다. migration gate와 다음 단계는 [`docs/rust-migration.md`](docs/rust-migration.md)를 확인하세요.

```powershell
cargo test --workspace --offline
cargo run --offline -p longctx100m-cli -- migration-status
cargo run --offline -p longctx100m-gateway
cargo run --offline -p longctx100m-importer --bin longctx100m-import-llama -- --help
cargo run --offline -p longctx100m-model --bin longctx100m-train -- --help
cargo run --offline -p longctx100m-model --bin longctx100m-infer -- --checkpoint <imported-dir> --prompt-tokens 1,2,3,4 --generate 8
python scripts/verify_model_parity.py
cargo run --offline -p longctx100m-cli -- capacity --tokens 100000000 --output-tokens 2048 --layers 24 --query-heads 16 --kv-heads 16 --head-dim 128 --world-size 128 --local-shard-tokens 1000000 --gpu-memory-gib 80
```

## Backend contract

게이트웨이는 다음 프로토콜을 구현하는 독립 추론 백엔드에 연결됩니다. 표준 OpenAI 호환 엔드포인트만 노출하는 엔진은 자체적으로 단일 요청 100M 처리 및 무절단을 보증하지 않으므로 프로토콜 호환 어댑터가 필요합니다.

- `GET /v1/long-context/capabilities`: `protocol_version`, `model_id`, `tokenizer_id`, `context_limit_tokens`, `supports_single_request: true`, `truncation_policy: "reject"`, `attention_mode`를 선언
- `POST /v1/long-context/tokenize`: 전체 입력을 inference 경로와 같은 tokenizer 및 wrapper로 세어 정확한 입력 토큰 수와 SHA-256을 반환
- `POST /v1/long-context/infer`: 원본 UTF-8 파일 바이트를 body로 받고, 질문·기대 SHA-256·요구 입력/총 컨텍스트 토큰·최대 출력 토큰을 헤더로 받음
- JSON 결과: `answer`, `model_id`, `tokenizer_id`, `input_tokens`, `processed_input_tokens`, `input_sha256`, `context_limit_tokens`, `attention_mode`, `truncated`
- 결과 검증: 업로드 SHA-256 동일, 입력 토큰 수가 목표 이상, 처리 토큰 수와 계산 토큰 수 일치, `truncated=false`, 출력 예산을 포함한 컨텍스트 여유 확인

프로토콜/배포 상세 및 capability 응답 예시는 [`docs/backend-protocol.md`](docs/backend-protocol.md)를 확인하세요.

## Llama checkpoint 가져오기

접근이 허용된 plain Llama `safetensors` checkpoint를 exact key/shape 검증 후 불러올 수 있습니다.

```bash
python -m pip install -e '.[engine,pretrained]'
longctx100m-import-llama --model org/model --revision <immutable-commit> \
  --max-position-embeddings 100002048 --out /shared/llama-base
```

이 작업은 **pretrained base weight/tokenizer를 포맷 변환**하는 것이지, 해당 weight가 100M context로 학습되어 있음을 의미하지 않습니다. 변환 manifest는 이를 `pretrained_for_100m_context: false`로 기록합니다. 이후 장문맥 continual pretraining, 품질평가와 GPU cluster prefill 검증이 별도로 필요합니다. Importer는 dense Llama 구조만 지원하며 sliding-window/MoE/비표준 head projection은 거부합니다.

## 자체 분산 attention 모델

- `src/longctx100m/model.py`: RMSNorm/RoPE/SwiGLU decoder
- `src/longctx100m/ring_attention.py`: exact causal ring forward와 gradient recomputation backward
- `src/longctx100m/triton_attention.py`: Linux/CUDA 환경에서 사용할 single-device tiled Triton attention kernel
- `src/longctx100m/train.py`: token-id corpus로 분산 sequence-parallel train entry point (`--initial-checkpoint`로 imported Llama 가중치 사용)
- `longctx100m-import-llama`: compatible Llama safetensors/토크나이저를 config 검증 후 변환
- `src/longctx100m/distributed_infer.py`: token-file 구간별 읽기와 greedy token broadcast를 사용하는 multi-rank generation reference (KV-cache 증분 decode는 미구현)
- 상세 알고리즘·설치·제약: [`docs/engine-design.md`](docs/engine-design.md)

분산 Gloo test 및 dense-reference 수치 비교:

```bash
python -m pip install -e '.[engine]'
python -m pytest tests/test_ring_attention.py -q
```

## 구조 선택 기준

1억 토큰 **dense full attention**과 1억 토큰 **모델 입력 길이**는 구분합니다. 희소/선형/압축 메모리 모델은 단일 요청 입력 길이를 늘릴 수 있지만 원문 모든 위치에 매 순간 동일한 dense attention을 제공하지는 않습니다. 사용자는 UI에서 한 파일로 한 번 제출하되, 추론 엔진은 하드웨어에 맞게 멀티 GPU/노드 분산·ring/context parallelism 등으로 구현해야 합니다. 여러 노드로 모델과 상태를 펼칠 때는 GPU 직접 통신용 고속 네트워크가 중요합니다.

- 현재 구현은 GPU/노드 간 exact causal K/V ring을 제공합니다. 모델 동작 길이 100M은 config에 설정할 수 있지만, 이를 학습한 체크포인트가 제공되지는 않습니다.
- Ring 통신은 메모리 footprint를 rank들에 나누고, dense attention O(N²) 연산은 그대로 유지합니다. 학습 비용/정확도/회수 품질은 target 장비와 실제 100M 학습 이후 별도로 확인해야 합니다.
- `longctx100m-capacity`는 단일 GPU/rank 가정에서 지정 topology가 KV 하한을 수용할 수 있는지 보고합니다. 현재 rank 내부 multi-GPU 배치는 지원하지 않으며, 산출값은 실측 배포 size가 아닙니다.
- vLLM은 별도 adapter로 계속 연결 가능하지만, 자체 모델 엔진과 동등한 100M 지원을 보장하는 것으로 취급하지 않습니다.

## 검증

```bash
python -m unittest discover -s tests -v
```

추가로 `python -m pytest tests/test_ring_attention.py -q`는 작은 tensor에서 dense-reference 출력/gradient, 2-rank Gloo ring, distributed decoder train step, autoregressive token-broadcast 결과를 확인합니다. Triton/CUDA kernel test는 해당 런타임이 없으면 skip됩니다. 이 검증은 100M scale, CUDA/NCCL 성능 또는 학습된 모델 품질을 입증하지 않습니다.
