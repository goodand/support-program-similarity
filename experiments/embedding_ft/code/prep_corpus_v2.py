"""학습 말뭉치 v2: 예산서 파싱 원문(<BUDGET_DB> 섹션원문, 2020~2026) 기반. 메인 세션 chunks.jsonl은 쓰지 않음(채점 인코딩 전용).
서술형 섹션만 사용(표·숫자 위주 섹션 제외). 사업 1건 = 문서 1건, 섹션 목록도 함께 보관(Contriever 구간 자르기·섹션 간 짝 용).
DB는 읽기 전용. 출력: data/train_docs_v2.jsonl, data/corpus_meta_v2.json
usage (호스트): <EMBED_ENV_PYTHON> -I -X utf8 code/prep_corpus_v2.py
"""
import hashlib, json, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
DB = Path("<DEV_EVAL_ROOT>/<BUDGET_DB>")
OUT = Path(__file__).resolve().parents[1] / "data"
NARRATIVE = ["1)목적내용", "2)개요", "2)②추진경위", "2)①법적근거", "2)②추진체계", "7)집행절차", "4)③기대효과", "10)추진방향", "9)외부지적", "다.2)결산사항"]
MIN_SEC = 30  # 이보다 짧은 섹션은 버림(빈 표 머리 등)

c = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
meta = {r[0]: r[1:] for r in c.execute("select 사업_id, 회계연도, 소관부처_정규화, 세부사업명, 프로그램코드, 단위사업코드, 세부사업코드 from 사업")}
secs = defaultdict(list)
q = f"select 사업_id, 섹션키, 원문 from 섹션원문 where 섹션키 in ({','.join('?' * len(NARRATIVE))}) order by 사업_id, id"
for sid, key, txt in c.execute(q, NARRATIVE):
    t = " ".join((txt or "").split())
    if len(t) >= MIN_SEC: secs[sid].append({"key": key, "text": t})

n = 0; lens = []
with open(OUT / "train_docs_v2.jsonl", "w", encoding="utf-8") as f:
    for sid, ss in secs.items():
        y, dept, name, p, u, s = meta[sid]
        doc = " ".join(x["text"] for x in ss)
        f.write(json.dumps({"사업_id": sid, "year": y, "dept": dept, "name": name, "code": [p, u, s], "sections": ss,
                            "text": doc, "sha256": hashlib.sha256(doc.encode("utf-8")).hexdigest()}, ensure_ascii=False) + "\n")
        n += 1; lens.append(len(doc))
lens = np.array(lens)
info = {"db": str(DB), "db_sha256": hashlib.sha256(DB.read_bytes()).hexdigest(), "sections_used": NARRATIVE, "min_section_chars": MIN_SEC,
        "docs": n, "chars_pct": {p: int(np.percentile(lens, p)) for p in (10, 50, 90)},
        "by_year": {int(y): int(k) for y, k in zip(*np.unique([meta[s][0] for s in secs], return_counts=True))},
        "out_sha256": hashlib.sha256((OUT / "train_docs_v2.jsonl").read_bytes()).hexdigest(),
        "note": "평가 원소 텍스트 포함(transductive, 메인 세션 동의 11-04). 쌍 정보를 쓰는 학습에서는 eval_exclusion_v1 적용할 것."}
(OUT / "corpus_meta_v2.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(info, ensure_ascii=False, indent=1))
