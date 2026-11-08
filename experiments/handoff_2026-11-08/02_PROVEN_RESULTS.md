# 증명된 결과만 (2026-11-08 14:30 기준)

"증명됨"의 기준: (a) 같은 채점기·같은 풀(run 011)·같은 질의로 재채점된 숫자이고, (b) seed 2개 이상에서 같은 방향이거나 짝 비교 95% 구간이 0을 벗어나며, (c) 11-08에 sonnet·haiku가 독립 대조해 일치한 것. 이 기준에 이르기 전인 것은 §B "판정 대기"에 둔다.
풀 = run 011 37,670원소. 사람판정 평가셋(test) = 51쌍·53질의·21그룹(경영, 2026 중앙부처). 규칙 선별셋(dev) = 87그룹·433질의(선별 전용, 이름 열 유리 편향). 지표: 마지막 정답 순위 logmean(낮을수록 좋음), 100위 안 비율, MAP, 100% 찾는 지점(최악 순위, Qwen 단독). 짝 비교 = 같은 seed run끼리 질의별 Δ의 그룹 bootstrap.
모든 숫자의 원장: `runs/037/runs_compare.xlsx`(run당 1행, 71 run 재채점). 짝 판정: `runs/036/VERDICT.md`, `runs/038/VERDICT_night4.md`.

## A. 증명된 것
| # | 결과 | 숫자 | 근거 파일 |
|---|---|---|---|
| P1 | **신규검토(②) 최고 = crop 대조학습 1 epoch, 학습률 2e-4, LoRA r16 (B3)**. 학습 전(base) 대비 test logmean −0.40 안팎 | seed 42/43/45/46: −0.410/−0.404/−0.402/−0.398. 100% 찾는 지점 291/449/694/273(학습 전 1,496) | runs_compare.xlsx; VERDICT.md R-A 표 |
| P2 | B3가 기존 crop(lr 5e-5)보다 낫다 — **test에서만**. dev(공고 질의 80%)는 0 | 짝 비교 seed 4: test −0.070 [−0.131, −0.009], dev +0.002 [−0.024, +0.027] | VERDICT.md R-C |
| P3 | **학습은 1 epoch가 최고(2·3 epoch는 같거나 나쁨)** | 2ep−1ep test +0.063 [+0.006, +0.122], dev +0.022 [−0.004, +0.048] (seed 2); 3ep test +0.055 [−0.068, +0.173], dev +0.061 [+0.018, +0.106] (seed 1) | VERDICT.md |
| P4 | **공고 질의는 학습 전후가 같음**(학습 말뭉치가 예산 DB로만 구성) | P2의 dev Δ 0; 질의별 진단: 공고 정답 질의 Δ 중앙값 0 | VERDICT.md, runs/021/diag_queries.json |
| P5 | **효율화(①) 최고 = 학습 전 임베딩, 그래프 k=5, 군집 1,575개**. 학습 모델은 모두 그보다 낮음 | 학습 전 45/51 비용 32.35; crop 43~44; B3 41; 2ep 42~43; 3ep 43; batch128 41. 해상도는 plateau 폭만으로 고른 뒤(`chosen`) 사후에 51쌍으로 평가함(`post_hoc_gold_chosen`) | runs/011·020/leiden/pareto_*.json |
| P6 | 해상도를 올리면 비용↓ 쌍 소폭↓ | 학습 전 1,575→2,066 군집: 45/32.35 → 44/27.38 | runs/011/leiden/pareto_chunk_vn+qwen_k5.json |
| P7 | **후처리·결합은 학습 전후와 같은 수준**(화이트닝·PageRank·관점 결합·앙상블·순위 융합·학습 전 보간·어댑터 평균) | 75후보 Holm 보정 유의 0건; 어댑터 평균 test 2.158 vs B3 2.140, 최악 488 vs 291 | runs/014~021·023·033, POOL_DIFF.md |
| P8 | **키워드(어휘) 검색 → 임베딩 전환이 가장 큰 단일 이득** | 사업당 읽을 후보 평균(사람판정 평가셋, Qwen 단독 마지막 정답 순위) 키워드 593.5 → 학습 전 106.5 → 학습 후 B3 seed별 33.5/36.6/39.6/28.5(4 seed 평균 34.5); 100개 안 비율 74% → 81% → 91%(seed 42) | runs/036/scores/011__keyword_VN.npz, REPORT_LABOR_SAVINGS |
| P9 | **풀을 L5 확정 정의(37,461)로 바꾸면 전 run이 0.08~0.13 좋아지고 모델 그룹은 유지**(인접 순위는 바뀜) | 16 run test −0.082~−0.111, dev −0.12~−0.13 | runs/036/POOL_DIFF.md |
| P10 | **seed 간 흔들림은 작다**(질의별 Δ 기준 sd test 0.010, dev 0.016) → seed 반복해도 결론은 같음 | 3쌍 실측: (B3 s42, crop s42), (B3 s43, crop s43), (B3 s45, crop s45) | runs/036/scores/011__qwen_crop_lr2e-4_1ep_s4{2,3,5}_*.npz 대 011__qwen_crop_len512_s4{2,3,5}_*.npz; CONCLUSIONS §2 |
| P11 | **단기 학습 시험(15스텝)과 1 epoch의 순위는 서로 다르다 → 선별은 1 epoch 짝 비교로 한다** | B5 r32+MLP: 단기 학습 시험 통과 → 1ep는 B3보다 낮음(−0.335 vs B3 −0.410); H3 lr5e-4: 통과 → 확증 미재현 | runs/030/redo_decide1.json, runs_compare.xlsx |
| P12 | **100% 찾는 지점은 seed에 크게 흔들림**(291~694) | B3 4 seed | runs_compare.xlsx 최악순위 열 |
| P13 | **학습 후 임베딩은 상위 700개(풀의 1.9%)만 읽으면 test 53·dev 433질의 모두, seed 4개 모두에서 유사 사업을 100% 찾음.** 키워드는 8,842개, 학습 전은 1,496(test)/3,671(dev)개 필요 | 100% 찾는 지점: B3 273~694(test)·528~686(dev); 학습 전 1,496/3,671; 키워드 8,842/5,712 | runs/036/scores/011__*.npz, REPORT_LABOR_SAVINGS §100% |

## B. 판정 대기 (근거 보강 필요 — 인용할 때는 '판정 대기'로 표기)
| 항목 | 상태 | 숫자 |
|---|---|---|
| 2026년 문서만 학습(D9c) | 판단불가: 구간 0 포함(seed 1) | B3 대비 test −0.010 [−0.055, +0.032], dev +0.012 [−0.007, +0.033] |
| 학습률 1e-4·3e-4, 짧은 crop, batch 128, LoRA r32+MLP, 쌍 2배·100% | B3 이하 또는 판단불가: 구간 0 포함(각 seed 1~2) | runs_compare.xlsx |
| 열 조합(절차·성과 추가, z-점수) | 효과 ≤0.02, seed 42 불통과·43 통과로 엇갈림 | runs/024/fields_combo_B3_s4*_z.json |
| 청크 형식 24 view × B3 | 11-08 인코딩 중, 결과 미도착 | runs/038/VERDICT_night4.md(갱신 예정) |
| 공고↔소속 세부 짝 학습 | 미실행. 짝 데이터만 v1(416)·v2(233) 존재, 사용자 판정 대기 | runs/040, runs/041 |
| 다른 임베딩 모델 3종 zero-shot(EmbeddingGemma 2·bge-m3·KURE-v1; KoE5는 10-21 결과와 중복이라 제외) | 11-08 17:00 채점 완료(zero-shot, seed 1회). 같은 view `all` 텍스트의 학습 전 Qwen 대비 3종 모두 나쁨: 규칙 선별셋 dev는 악화 확정(+0.15~+0.26), 사람판정 평가셋 test는 구간이 0을 걸쳐 판단불가: 구간 0 포함(+0.11~+0.12). 100% 찾는 지점 6,544~11,490 대 Qwen 2,802 | runs/038/VERDICT_night4.md(P1_zeroshot 행), runs/036/scores/011__{google__embeddinggemma-2,BAAI__bge-m3,nlpai-lab__KURE-v1}__view_all.npz |
| 멀티모달(표·이미지) | 미실험(로컬 표 이미지 7장) | — |
| 노동시간 절감 액수 | t(후보 1건 검토 시간) 미측정 → 범위로만 | REPORT_LABOR_SAVINGS |

## C. 데이터 고정값 (해시 앞 16자)
- 사람판정 평가셋 runs/009/eval_set_v2.json 05f13817…; 제외 목록 runs/009/eval_exclusion_v1.json 92888f78…
- 규칙 선별셋 runs/017/dev_v0.json; 풀 run 011 chunks(37,670); L5 확정 풀 runs/035(37,461, chunks_meta.json에 전 입력 sha)
- 학습 말뭉치 train_docs_v2.jsonl 73caf89b…(budget DB(sqlite) 98b26d72…, 2020~2026 섹션 10종, 53,686건)
- 최고 run 폴더: embedding_ft_docker/output/qwen_crop_lr2e-4_1ep_s42_20261105_night2 (manifest에 전 설정)
