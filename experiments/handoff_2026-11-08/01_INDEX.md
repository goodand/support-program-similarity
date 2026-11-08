# 찾기 안내 — 어떤 질문이든 3번 안에 (모두 `<workspace>\program_inventory\` 기준)

## 1단계: 질문 유형 → 파일 하나
| 알고 싶은 것 | 열 파일 | 그 안에서 |
|---|---|---|
| 어떤 run이 몇 점인가(모든 run, 같은 기준) | `runs/037/runs_compare.xlsx` (csv 동일) | run_id 행. 열: 평가셋 logmean·Δ(학습 전 base)·100위안·MAP·100% 찾는 지점(최악순위 열), 선별셋 같은 열, 조건(seed·lr·epochs·batch·r·짝·학습 s), 서술(가설·사전 규칙·기록된 판정·문제) |
| A가 B보다 나은가(통계 판정) | `runs/036/VERDICT.md`(11-06 밤), `runs/038/VERDICT_night4.md`(11-07~08) | 비교 행: test Δ [95%], dev Δ [95%], 판정(개선/악화/…만 확정/엇갈림/판단불가: 원인 — 구간 0 포함·자료 부족·자료 없음·seed 엇갈림) |
| 증명된 결론만 | `HANDOFF_CONTEXT_2026-11-08/02_PROVEN_RESULTS.md` | P1~P12, §B 증명 안 됨 |
| 전체 결론·한계·다음 할 일 | `CONCLUSIONS_2026-11-08.md` | §1 결론, §2 한계, §3 미해결, §5 검증 기록 |
| 보고용 문장·노동시간 계산 | `REPORT_LABOR_SAVINGS_2026-11-08.md` | 핵심 문장, 후보 수 표, 계산식(t 변수) |
| 왜 그렇게 결정했나(3세션 합의) | `DECISIONS.md` | D-1~D-9 행(제안자·찬반·상태·근거) |
| 사용자가 무엇을 정했/시켰/물었나 | `user_log/<날짜>/decisions.md·directives.md·questions.md`(11-06~), `QUESTIONS_2026-11-05.md`, `HUMAN_MESSAGES_2026-11-05.md`(11-05 이전) | msg_id 행 |
| 사용자 의도·원칙 요약 | `HANDOFF_CONTEXT_2026-11-08/03_USER_INTENT.md` | 의도 파악 세션 작성 |
| 용어 뜻 | `GLOSSARY.md` | 규칙 선별셋=dev, 사람판정 평가셋=test, 단기 학습 시험=15스텝 등 |
| 풀(L5) 정의·개수 | `L5_DEFINITION_2026-11-06.md`, `runs/035/chunks_meta.json` | 정의 문장, 37,461, 입력 sha |
| 풀 변경 영향 | `runs/036/POOL_DIFF.md` | run별 011→035 |
| 군집화(①) 결과 | `runs/011/leiden/pareto_chunk_vn+qwen_k{3,5,10}.json`, `runs/020/leiden/pareto_<run>_vn+qwen_k5.json` | `post_hoc_gold_chosen`(51쌍 together·cost), `pareto`(해상도별) |
| 사고·교훈 | `runs/030/INCIDENT_2026-11-05_decide1.md`, `runs/040/INCIDENT_2026-11-07_night4.md` | 원인·조치·재발 방지 |
| 비교표 바깥에 있는 run(어댑터 평균 soup, 학습 전 재인코딩 034 — manifest가 빠진 run) | `runs/036/POOL_DIFF.md`(soup 점수), `DECISIONS.md` D-7(soup 사전 규칙·불통과), `output/qwen_crop_soup3lr_1ep_s42_20261106_day4/adapter/MERGE_MANIFEST.json`(병합 입력 sha) | — |
| 원천 DB 두 판(budget_sme 08-10 vs 08-11)·공고 마스터 파일 결정 | `L5_DEFINITION_2026-11-06.md`(확정 정의·출처), `runs/035/chunks_meta.json`(실제 읽은 파일 sha: budget_sme a4410d77…=08-11판, 마스터 (1).xlsx 690a4ac6…, master_fills 0f28133d…), `HUMAN_MESSAGES_2026-11-05.md`·`runs/layers_2026/LAYERS_FINAL.md`(11-03~05 경위) | 사용자 11-06 결정: 정의 (1)안, 공고 = 마스터 (1).xlsx + fills |
| LLM 사용 범위 지시(규칙 기반 재현 + LLM은 검증·규칙 수정 근거만) | `user_log/2026-11-08/directives.md`(13:5x 원문), `HANDOFF_CONTEXT_2026-11-08/03_USER_INTENT.md` 2절, 메모리 `imitate-human-patterns-as-rules`(평가에 LLM judge 안 씀) | |
| seed 간 변동 실측(3쌍) | `runs/036/scores/011__{qwen_crop_lr2e-4_1ep_s42_20261105_night2, _s43_20261106_day3, _s45_20261106_day4}.npz` 대 `011__qwen_crop_len512_{s42,s43}_20261104_night, s45_20261105_confirm}.npz`; 계산 `runs/036/verdict.py` 독스트링 | sd test 0.010·dev 0.016 |
| 학습 run의 정확한 설정 | `<workspace>\embedding_ft_docker\output\<run_id>\manifest.json` | hyperparams, pairs_rule, 말뭉치 sha, 시간 |
| 야간/주간 실행 기록(파인튜닝 세션) | `embedding_ft_docker/NIGHT_RUN_*.md`, `DAY_RUN_*.md` | 시각순 줄 |

## 2단계: run_id 읽는 법 (파일 열기 전에 뜻을 알 수 있음)
`qwen_<레시피>_<단계>_s<seed>_<날짜>_<대기열>` — 예 `qwen_crop_lr2e-4_1ep_s42_20261105_night2`
- 레시피: `crop`=조각 대조학습(기본 lr 5e-5), `crop_lr2e-4`=B3, `cropB5r32mlp`, `cropD5short`, `D9c_y2026`, `soup3lr`, `placebo`(위약)
- 단계: `st15`=단기 학습 시험(15스텝), `1ep/2ep/3ep`, `len512`(11-04 1ep 기본)
- 대기열: `scr`(11-05 선별), `probe`(단기 학습 시험), `night`(11-04), `night2`(11-05), `confirm`(확증), `day3~6`(11-06), `night3`(11-06 밤), `night4`(11-07~08)
- 학습 전(base): `base` = Qwen3-Embedding-8B 학습 전(runs/011/emb_cache/qwen_chunk.npz)

## 3단계: 숫자 재계산이 필요할 때
- 질의별 원점수: `runs/036/scores/011__<run>.npz`(키 `test__VN+Qwen__last` 등, 질의 키 `test__qkeys`) → `runs/036/score_pool.py`로 생성
- 짝 판정 재실행: `runs/038/verdict_pairs.py`(짝 목록 `embedding_ft_docker/logs/night4_pairs.json`), `runs/036/verdict.py`
- 통합 표 재생성: `runs/037/build_merged.py` → `merge_narrative.py`
- 채점기 본체: `eval_v3.py`(풀·VN·별칭·순위), 군집화: `knn_build.py`→`select_n_pareto.py`
