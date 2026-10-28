# Harness: 남윤형 경영 54개 사업 ↔ 김미정 자료 대응 검증 (eval_name_match_001)

## 질문
남윤형 보고서 경영 분야 54개 사업이 김미정 자료(중앙부처 계열)에 **실제로 없는가**, 아니면 **내 추출(사업명 열 규칙·연도 필터)이 놓쳤는가**.

## 입력 (경로 + sha256 고정)
| 역할 | 경로 | sha256 |
|---|---|---|
| 평가 사업 54개 | `<DEV_EVAL_ROOT>/jh_dev/kim_program_inventory_v1/runs/002/eval_match.json` (`rows`) | `58f0259015edaeed4b5fc1cf4fd9e7fe063ced5a9d3da218579fe94f1a842d56` |
| 병합 DB (run 002) | `<DEV_EVAL_ROOT>/jh_dev/kim_program_inventory_v1/runs/002/kim_programs.duckdb` | `3424ff69cefca208a361e3b7662dccfa98ce117dbcf6e64e376e83f5e36980da` |
| 평가 원본 엑셀 | `<DEV_EVAL_ROOT>/jh_dev/govparse-claude-lab/outputs/goldenset_similar_groups_xlsx/유사사업그룹_표추출.xlsx` | `f185d861455f2e9fee9fe35dd517452cdaf342d653ec7d7ffa7432172f32aee2` |
| 중앙 예산 DB (대표) | `<DEV_EVAL_ROOT>/전부처사업정보DB/build/budget_all.sqlite` | `98b26d72633d038ce412fea6bde92e641b7b6fe3620b367a55193521212be38d` |
| 결정적 전수 검색 결과 | `<DEV_EVAL_ROOT>/jh_dev/kim_program_inventory_v1/runs/002/eval_fullscan.json` (모든 텍스트 칸 포함 검색) | 실행 후 기록 |

검색 범위: `<DEV_EVAL_ROOT>` 아래 `jh_dev`, `_codex_outputs`, `_embedding_v3_env` 제외. 지자체 계열(`businfo`, `지자체사업정보DB`, `세출예산`)은 제외(평가 대상이 중앙부처 사업).

## 판정 범주 (사업마다 하나)
- `exact`: 정규화(괄호·특수문자·띄어쓰기 제거) 후 같은 이름이 있음
- `renamed_same`: 이름은 다르지만 같은 사업 (같은 소관 + 사업목적·내용 근거)
- `inside_parent`: 김미정 자료의 상위 사업(세부/내역) 안에 이 사업이 하위 구성으로 들어 있음 (근거: 상위 사업의 내용·산출근거 문구)
- `notice_only`: 예산 자료에는 없고 공고 자료에만 있음
- `not_found`: 찾지 못함 (검색한 위치 목록 필수)

각 판정에 근거(파일, 테이블/시트, 열, 값 일부, 연도)를 붙인다. 연도는 기록만 하고 판정에 쓰지 않는다.

## 단계
1. **결정적 전수 검색** (스크립트, 완료 후 입력): 모든 텍스트 칸에서 정규화 포함 검색.
2. **Find** (agent, 그룹 묶음 5개): 묶음 안의 사업마다 1의 결과와 DB를 직접 조회해 판정. 읽기 전용.
3. **Verify** (agent, 묶음 5개, 적대적): Find가 `not_found`가 아닌 판정을 낸 사업마다 인용 근거를 직접 다시 조회해 값이 실제로 있는지, 같은 사업이 맞는지 반박을 시도. 불확실하면 반박으로 본다. `not_found`는 다른 검색어(핵심어·소관)로 한 번 더 찾는다.
4. **집계** (스크립트 코드): 최종 판정 = Verify를 통과한 판정, 반박된 것은 `disputed`.

## 카나리 (harness 자체 검증; 하나라도 틀리면 결과를 쓰지 않는다)
- 원문 일치 5개(방송영상콘텐츠 제작지원, XaaS 선도 프로젝트, AI·디지털 기반 방송프로그램 제작지원, 안전동행지원사업, 일터혁신 상생컨설팅 지원) → `exact`
- 안경산업 3개 → `exact`가 아니어야 하고 근거에 `한국안경산업 고도화육성` 등장
- 데이터바우처 지원사업 → 근거 연도에 2026 등장

## 제약
- 원본 파일·DB 수정 금지, sqlite는 `mode=ro`, duckdb는 `read_only=True`.
- PDF/HWP는 Read 금지(보안 훅). 필요하면 로컬 파서로 필요한 부분만.
- 근거 없는 판정 금지. 도구로 확인하지 못한 것은 `not_found` + 사유.
