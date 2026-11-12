"""11-12: LLM 응답에서 같은 번호가 반복되면 첫 등장만 순위로 센다. 모든 질의별 파일의 mate_rank를 top_raw로 다시 계산.
이전 값은 mate_rank_before_dedupe로 남긴다(한 번만). usage: .venv/Scripts/python.exe -X utf8 runs/044/llm_rank_dedupe_rescore_harness.py"""
import json, glob
ch = 0
for f in sorted(glob.glob('runs/044/llm_rank_harness/claude-*/q*.json')):
    d = json.load(open(f, encoding='utf-8'))
    if d['top_raw'] is None: continue
    q = d['query']; clean = list(dict.fromkeys(x for x in d['top_raw'] if x != q and 0 <= x < 668))
    new = {str(m): (clean.index(m) + 1 if m in clean else None) for m in d['mates']}
    if new != d['mate_rank']:
        d.setdefault('mate_rank_before_dedupe', d['mate_rank']); d['mate_rank'] = new; d['dedupe_rule'] = '반복 번호는 첫 등장만 (11-12)'; ch += 1
        json.dump(d, open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f.split('llm_rank_harness')[1], d['mate_rank_before_dedupe'], '->', new)
print('바뀐 파일', ch)
