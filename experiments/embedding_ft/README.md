# embedding_ft — Qwen3-Embedding-8B LoRA 파인튜닝 환경 (2026-11-02)

예산서 세부사업 문서 간 유사도 군집에 쓸 임베딩 모델을 파인튜닝하기 위한 Docker 실행 환경입니다. **지금 들어 있는 것은 환경과 스모크 테스트뿐입니다.** 학습 데이터 생성, 본 학습, 평가 코드는 아직 없습니다.

## 구성
| 파일 | 역할 |
|---|---|
| `run.sh` | 컨테이너 실행 (GPU 전체, 공유 메모리 16GB, 오프라인 모드) |
| `code/smoke_qwen_lora.py` | 스모크 테스트: 더미 쌍으로 LoRA 1 epoch 학습 → 어댑터 저장 → 다시 불러와 검증 |
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
