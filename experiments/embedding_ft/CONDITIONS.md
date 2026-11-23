# 파인튜닝 실험 조건 (기준 실행: 2026-11-06, seed 42)

668개 공고 검색 평가에서 "임베딩 학습 후(5스텝 seed 42)"로 보고한 실행의 조건입니다. 값은 실행이 남긴 `manifest.json`에서 옮겼습니다.

## 모델과 방법
| 항목 | 값 |
|---|---|
| 기반 모델 | Qwen/Qwen3-Embedding-8B (bf16, attention 구현 sdpa) |
| 방법 | LoRA 어댑터 학습. 어댑터는 따로 저장하고 기반 모델에 병합하지 않음 |
| LoRA | r 16, alpha 32, dropout 0.1, 대상 q_proj·k_proj·v_proj·o_proj |
| attention dropout | 0.1 (학습 때만 작동) |
| 손실 | MultipleNegativesRankingLoss, scale 20.0 |
| 짝 만들기 | 한 문서에서 두 구간을 잘라 anchor·positive로 사용. 구간 길이는 문서의 10~50%(최소 150자), 400자 미만 문서는 제외 |
| 묶음 구성 | 같은 글이 한 묶음에 둘 들어가지 않게 함(no_duplicates), 마지막 남는 묶음은 버림 |

## 학습 설정
| 항목 | 값 | 환경 변수 |
|---|---|---|
| batch size | 128 | `BATCH=128` |
| 학습 스텝 | 5 (128 × 5 = 640짝) | `MAX_STEPS=5` |
| 학습률 | 2e-4, warmup 비율 0.1 | `LR=2e-4` |
| 짝 수 상한 | 9,321짝(문서에서 만들 수 있는 짝 46,950개 중) | `LIMIT=9321` |
| 학습 최대 토큰 길이 | 512 | `TRAIN_LEN=512` |
| seed | 42 | `SEED=42` |
| 짝 방식 | 구간 자르기 | `PAIRS_FILE=crop` |
| 말뭉치 | 예산서 파싱 원문 문서 53,686건(2020~2026) | `CORPUS=train_docs_v2.jsonl` |
| 기타 | bf16, gradient checkpointing, gradient accumulation 없음 | |

- batch 128은 96GB GPU에서 최대 70.1GB를 썼습니다.
- 같은 조건(학습률 2e-4, 1 epoch, seed 42)에서 batch 64와 128을 비교한 결과는 판단불가였습니다. 128을 고른 근거는 "묶음이 클수록 묶음 안 오답이 많다"는 가설과 GPU 메모리 여유입니다.
- 이 기준 실행은 평가셋 점수를 본 뒤에 골랐습니다.

## 학습 말뭉치
- 예산서 DB의 서술형 섹션 10종(목적·내용, 개요, 추진경위, 법적근거, 추진체계, 집행절차, 기대효과, 추진방향, 외부지적, 결산사항)만 사용했습니다. 30자 미만 섹션은 제외했습니다.
- 사업 1건이 문서 1건입니다. 문서 길이는 10% 지점 355자, 중앙 1,417자, 90% 지점 4,012자입니다.
- 평가에 쓰는 사업의 원문도 말뭉치에 들어 있습니다(짝 정보는 쓰지 않음).

## 인코딩(검색) 설정
| 항목 | 값 |
|---|---|
| 정밀도 | bf16 |
| 최대 토큰 길이 | 2048 |
| 프롬프트 | 없음(질의와 문서를 같은 방식으로 인코딩) |
| 정규화 | L2 정규화 후 float32로 저장 |
| batch size | 학습 직후 인코딩은 32(길이순 정렬), 668개 평가용 재인코딩은 16 |

## 실행 결과
| 항목 | 값 |
|---|---|
| train_loss | 0.699 |
| 학습 시간 | 190초 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q (96GB) |
| 이미지 | embed-ft:20261102-v1 (torch 2.8.0+cu128, transformers 5.3.0, sentence-transformers 5.3.0, peft 0.21.2) |

## 파일
| 파일 | 역할 |
|---|---|
| `code/prep_corpus_v2.py` | 예산서 DB에서 학습 말뭉치 만들기 |
| `code/qwen_simcse.py` | LoRA 학습과 학습 직후 인코딩 |
| `code/encode_adapter.py` | 저장한 어댑터로 텍스트 인코딩 |
| `code/score_ft.py` | 임베딩만 바꿔 평가 스크립트 실행 |
| `seed_repeat_b3b128_mem5.sh` | 위 조건 그대로 seed만 바꿔 반복하는 docker 명령 |

어댑터를 다시 불러올 때는 `peft.PeftModel.from_pretrained`를 씁니다.
