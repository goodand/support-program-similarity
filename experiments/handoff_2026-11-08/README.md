# Handoff context (2026-11-08)

새 세션에 넘기는 인계 문서 묶음. 읽는 순서는 `NEW_SESSION_PROMPT_v2.md`에 있다 (00 → 03 → 02 → 01).
워크스테이션 경로는 `<workspace>`로, 내부 저장소 이름은 `program_inventory`로 바꿔 올렸다. 수치·run_id·파일 상대경로는 원본 그대로다.

## materials/ — 표·그래프 재료 (markdown만)
- `runs_compare.md`: run 71개 원장(설정·지표 열). 비고·사고 열은 비공개.
- `runs__044__CLUSTER_GROUPS.md`: 효율화 그룹 단위 채점(사업 53개·그룹 21개). 임베딩별 군집 수, 가장 큰 군집, 한 군집에 완전히 든 그룹 수, 학습 전 대비 차이 구간.
- `CONCLUSIONS_2026-11-08.md`, `REPORT_LABOR_SAVINGS_2026-11-08.md`, `GLOSSARY.md`, `DECISIONS.md`: 결론·노동시간 절감 계산·용어·결정 기록.
- `runs__036__VERDICT.md`, `runs__038__VERDICT_night4.md`, `runs__038__VERDICT_night5.md`, `runs__036__POOL_DIFF.md`: 짝 비교 통계 판정표.
- `baselines.md`: 키워드 검색·학습 전(base)·다른 모델 zero-shot 3종·학습 후 B3 seed 4개의 후보 수 요약(평균·중앙값·100위 안 비율·100% 찾는 지점), 두 평가 자료 × 두 시스템.
- `recall_at_k.md`: 같은 대상의 recall@k 곡선 점(k=1~10,000). 도식용 곡선은 이 표로 그린다.
