"""11-12 사용자 승인 규칙: 응답 안의 'A if False else B'는 B로 읽는다. 모든 조건의 질의별 파일에 똑같이 적용.
top_raw가 None이고 result 원문이 있는 파일만 대상. 원문(result)은 그대로 두고 top_raw·mate_rank·parse_rule을 채운다.
usage: .venv/Scripts/python.exe -X utf8 runs/044/llm_rank_lenient_parse_harness.py"""
import json, re, glob
for f in sorted(glob.glob('runs/044/llm_rank_harness/claude-*/q*.json')):
    d = json.load(open(f, encoding='utf-8'))
    if d['top_raw'] is not None or not d.get('result'): continue
    t = re.sub(r'(\d+)\s+if\s+False\s+else\s+(\d+)', r'\2', d['result'])
    try:
        top = [int(x) for x in json.JSONDecoder().raw_decode(t[t.index('{'):])[0]['top']]
    except Exception:
        print('여전히 실패', f); continue
    q = d['query']; clean = list(dict.fromkeys(x for x in top if x != q and 0 <= x < 668))
    d.update({'top_raw': top, 'parse_rule': "'A if False else B' -> B (11-12 승인)", 'self_in_top': q in top,
              'mate_rank': {str(m): (clean.index(m) + 1 if m in clean else None) for m in d['mates']}})
    json.dump(d, open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('적용', f, d['mate_rank'])
