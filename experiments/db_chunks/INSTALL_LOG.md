# 설치 기록

- 2026-10-28: `.venv` (uv 0.11.28, Python 3.12) 새로 만듦. duckdb 2.0.0, openpyxl 3.1.5 설치.
  - 이유: 사용자 결정 "DuckDB로 병합". 시스템 Python에는 duckdb가 없음.
  - numpy 2.5.4 추가: duckdb의 Python UDF(l1, l2 등록)가 numpy를 요구함 (run 002 적재 단계 실패 원인).
  - 1차 추출(run 001)은 시스템 Python 3.12 + openpyxl로 시작했다. 정규화 함수는 같은 소스를 쓴다(load_duckdb.py가 AST로 가져옴).
  - pymupdf 추가: eval_name_match_001 전수 검색에서 공고 PDF 텍스트 추출(적대적 검토 지적 1).
  - olefile 추가: 확장자는 .hwpx지만 실제로는 HWP5(OLE) 바이너리인 파일의 본문 텍스트 추출.
  - 2026-10-28: leidenalg 0.12.0, python-igraph 1.0.0, infomap 2.16.0, scikit-learn 1.9.1 추가 (사용자 선택 '추천 조합': Infomap 기록 길이 + Significance 파레토로 군집 수 결정, select_n_pareto.py). 다른 프로젝트 venv는 건드리지 않으려고 이 venv에 설치.
