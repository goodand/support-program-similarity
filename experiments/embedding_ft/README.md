# embedding_ft — Qwen3-Embedding-8B LoRA 파인튜닝 환경 (2026-11-02)

예산서 세부사업 문서 간 유사도 군집에 쓸 임베딩 모델을 파인튜닝하기 위한 Docker 실행 환경입니다. **들어 있는 것: 환경, 스모크 테스트, 임베딩 생성·비교 도구.** 학습 데이터 생성, 본 학습 코드는 아직 없습니다.

## 구성
| 파일 | 역할 |
|---|---|
| `run.sh` | 컨테이너 실행 (GPU 전체, 공유 메모리 16GB, 오프라인 모드) |
| `code/smoke_qwen_lora.py` | 스모크 테스트: 더미 쌍으로 LoRA 1 epoch 학습 → 어댑터 저장 → 다시 불러와 검증 |
| `code/embed_shards.py` | 글 → 임베딩 shard (st / kv / vllm) |
| `compare_shards.py` | 두 임베딩 저장소의 같은 해시끼리 코사인 비교 |
| `Dockerfile` | **로컬 전용** (공개 검사기가 허용하지 않는 형식이라 git에서 제외) |

## 이미지
- `embed-ft:20261102-v1`: 기존 학습용 이미지(torch 2.8.0+cu128, transformers 5.3.0, sentence-transformers 5.3.0, accelerate 1.10.1, datasets 5.0.1)에 `peft==0.21.2`만 추가했습니다.
- 기반 이미지는 수정하지 않았습니다.

## 실행
```bash
EMBED_FT_WORK=<작업 폴더> MODEL_DIR=<Qwen__Qwen3-Embedding-8B가 들어 있는 폴더> bash experiments/embedding_ft/run.sh
```
- `MODEL_DIR`은 읽기 전용으로 `/models`에 연결됩니다. 모델은 복사하지 않습니다.
- `EMBED_FT_WORK` 아래에 `data/`(읽기 전용), `output/`, `logs/`가 만들어집니다. 학습 결과물과 로그는 repo 밖에만 쌓입니다.
- 인자로 `code/` 아래 다른 스크립트 이름을 주면 그 스크립트를 실행합니다.

## 스모크 결과 (2026-11-02 실측, 성공)
설정: Qwen3-Embedding-8B(bf16), LoRA(r=16, alpha=32, q/k/v/o_proj), 의미 없는 영문 더미 쌍 20개, MultipleNegativesRankingLoss, 1 epoch, batch 4, gradient checkpointing.

| 항목 | 값 |
|---|---|
| 학습 가능한 값 | 15,335,424개 (전체의 약 0.2%) |
| 최대 GPU 메모리 | 15.3GB (96GB GPU) |
| train_loss | 약 0.043 |
| 임베딩 차원, 장치 | 4096, cuda:0 |
| 재로드 vs 학습 직후 최대 차이 | 0.0023 (bf16 오차 범위) |
| 학습으로 바뀐 방향 재현 (delta cosine) | 0.94 |

8B 전체 학습은 가중치, 기울기, 옵티마이저 상태를 합쳐 약 100~130GB가 필요해서 96GB를 넘습니다. 그래서 LoRA를 씁니다.

## 알려진 사항
- transformers 5.3의 `SentenceTransformer.load_adapter`는 어댑터 가중치 이름을 잘못 대응시켜, 학습한 값이 새로 초기화됩니다(MISSING 경고). 그래서 재로드에는 `peft.PeftModel.from_pretrained`를 씁니다.
- 메모리 여유가 커서 본 학습에서는 배치 크기를 늘릴 수 있습니다. MNRL은 배치가 클수록 음성 예시가 많아집니다.

## 임베딩 생성 (embed_shards.py, 2026-11-04)
Windows 기준 경로(sentence-transformers, bf16, sdpa, left padding, max_seq 2048, last-token pooling, L2 정규화, 문서 instruction 없음)와 같은 벡터를 Docker에서 만듭니다.

### 실행
```bash
EMBED_FT_WORK=<작업 폴더> MODEL_DIR=<모델 폴더> INPUT_DIR=<입력 jsonl 폴더> \
  bash experiments/embedding_ft/run.sh embed_shards.py /input/<입력>.jsonl /output/<결과 폴더> \
  --backend st --gpu-mem 0.35 --report /output/<결과 폴더>.report.json
```
- 이미지: `embed-ft:20261102-v1` (st, kv). vLLM은 `IMAGE=vllm/vllm-openai:v0.11.0`을 앞에 붙이고 `--backend vllm`.
- `INPUT_DIR`은 읽기 전용으로 `/input`에 연결됩니다(선택).
- 입력: jsonl 한 줄에 `{"h": sha256 hex, "t": 글}`. `--hashes 파일`로 일부만 고를 수 있습니다.
- 출력: `<결과 폴더>/s#####.npz` (키 `h` 해시 배열, `v` float16 N×4096, 정규화됨). 4,096개씩, 임시 파일 저장 후 이름 변경.
- 재개: 결과 폴더에 이미 있는 해시는 건너뜁니다. 중간에 끊겨도 같은 명령을 다시 실행하면 됩니다.
- `--gpu-mem`: GPU 메모리 상한(전체 대비 비율, 기본 0.85). st/kv는 이 프로세스의 상한, vLLM은 시작 시 미리 잡는 양. 다른 작업과 함께 돌릴 땐 0.35 정도. 모델 가중치만 약 16GB라 0.3 미만은 비현실적입니다.
- 비교: `python compare_shards.py <a> <b> <최소 코사인> [--json 결과.json]` (shard 폴더 또는 npz 하나, 같은 해시끼리 코사인).

### 검증 결과 (2026-11-04 실측)
| 항목 | 결과 |
|---|---|
| Docker st vs Windows (64개) | 코사인 min 0.99980 |
| Docker st vs Windows (43,936개 전체) | min 0.99939 / mean 0.99991, 검색 평가 MAP·MRR 차이 0.002 이하 → **채택 가능** |
| 속도 (st, 2,000개씩) | ≤300자 13,858 / 300–1500자 14,647 / 1500–2826자 14,143 chars/s |
| vLLM vs st (64개) | min 0.99974 |

### KV 앞부분 재사용 결론: 쓰지 않음
- `--backend kv`: 앞부분이 같은 글은 그 부분을 한 번만 계산하고 KV 캐시를 복사해 뒷부분만 계산합니다.
- 동일성: st 대비 200개 min 0.99854 (기준 0.999 미달, mean 0.99984). bf16 오차로 추정, 미검증.
- 실제 데이터에서 공통 앞부분이 짧아 토큰 절감 2–4%(길이 구간별), 사업 묶음에서도 17%. 짧은·중간 글에서는 st보다 느립니다(3,562 / 11,685 chars/s), 긴 글만 약 10% 빠름.
- vLLM `enable_prefix_caching`은 embed에서 작동합니다(200개 적중 51,984/130,706 토큰). st보다 빠른 구간은 300–1500자(18,189 chars/s)뿐이고, 2,000개 이상 긴 입력에서 마지막 요청 몇 개가 멈추는 현상이 2회 있었습니다(원인 미확인). 기본 경로로 쓰지 않습니다.
