"""LLM 순위 결과 검증 + 기록. 질의별 파일에서 순위를 독립적으로 다시 계산해 summary.json과 대조한다.
usage: .venv/Scripts/python.exe -X utf8 runs/044/llm_rank_verify_harness.py
출력: runs/044/llm_rank_harness/VERIFY_LOG_harness.md (실행할 때마다 맨 아래에 추가), verify_<시각>_harness.json"""
import json, glob, os, hashlib, datetime
from pathlib import Path
from collections import defaultdict
B = Path('runs/044/pool668_newreview_harness'); O = Path('runs/044/llm_rank_harness')
R = [json.loads(l) for l in open(B / 'chunks.jsonl', encoding='utf-8')]
keys = json.load(open(B / 'texts.json', encoding='utf-8'))['keys']; kix = {k: i for i, k in enumerate(keys)}
ev = json.load(open(B / 'eval_set_pool668.json', encoding='utf-8'))
T = defaultdict(set)
for g, a, b in ev['pairs']: T[g] |= {kix[a], kix[b]}
QS = sorted((g, q) for g, m in T.items() for q in m)
PRICE = {'claude-haiku-5-5': (0.10, 0.50), 'claude-sonnet-5-5': (2.0, 10.0), 'claude-opus-5-5': (4.0, 20.0), 'claude-fable-5-2': (8.0, 40.0)}  # $/1M 입력·출력, claude-api 스킬 표(2026-11-03 캐시)
now = datetime.datetime.now().astimezone().isoformat(timespec='seconds'); rows = []
for d in sorted(p for p in O.iterdir() if p.is_dir() and p.name.startswith('claude-')):
    model, eff, inp = d.name.rsplit('_', 2)
    fs = sorted(d.glob('q*.json')); chk = {'condition': d.name, 'files': len(fs)}
    if len(fs) != 35: chk['status'] = '미완료'; rows.append(chk); continue
    last, prob = [], []
    for i, f in enumerate(fs):
        r = json.load(open(f, encoding='utf-8')); g, q = QS[i]
        if (r['group'], r['query']) != (g, q): prob.append(f'{f.name}: 질의 불일치')
        if r['top_raw'] is None: prob.append(f'{f.name}: 응답 해석 실패'); last.append(None); continue
        clean = list(dict.fromkeys(x for x in r['top_raw'] if x != q and 0 <= x < len(R)))  # 반복 번호는 첫 등장만
        rk = [clean.index(m) + 1 if m in clean else None for m in sorted(T[g] - {q})]
        last.append(None if None in rk else max(rk))
        if r.get('modelUsage') and model not in r['modelUsage']: prob.append(f'{f.name}: 모델 불일치 {list(r["modelUsage"])}')
    S = json.load(open(d / 'summary.json', encoding='utf-8')) if (d / 'summary.json').exists() else {}
    u = [json.load(open(f, encoding='utf-8')) for f in fs]
    tin = sum((x['usage'] or {}).get('input_tokens', 0) for x in u); tcw = sum((x['usage'] or {}).get('cache_creation_input_tokens', 0) for x in u)
    tcr = sum((x['usage'] or {}).get('cache_read_input_tokens', 0) for x in u); tout = sum((x['usage'] or {}).get('output_tokens', 0) for x in u)
    pi, po = PRICE.get(model, (None, None))
    chk.update({'status': '통과' if not prob and S.get('last') == last else '확인 필요', 'problems': prob,
                'summary_matches_recomputed': S.get('last') == last, 'worst': None if None in last else max(last),
                'missed_30': sum(x is None for x in last), 'self_in_top': sum(bool(x['self_in_top']) for x in u),
                'model_recorded': sum(bool(x.get('modelUsage')) for x in u),
                'tokens': {'input': tin, 'cache_write': tcw, 'cache_read': tcr, 'output': tout},
                'cost_cli_usd': round(sum(x['total_cost_usd'] or 0 for x in u), 2),
                'cost_list_price_nocache_usd': round(((tin + tcw + tcr) * pi + tout * po) / 1e6, 2) if pi else None})
    rows.append(chk)
out = {'at': now, 'script_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], 'rows': rows}
json.dump(out, open(O / f'verify_{now[:19].replace(":", "")}_harness.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
L = [f'\n## {now} 검증', '', '| 조건 | 상태 | 가장 나쁜 순위 | 30위 밖 | 자기 번호 포함 | 모델 기록 | 재계산=요약 | CLI 비용 | 정가(캐시 없음) |', '|---|---|---|---|---|---|---|---|---|']
for c in rows:
    if c.get('status') == '미완료': L.append(f"| {c['condition']} | 미완료 ({c['files']}/35) |  |  |  |  |  |  |  |"); continue
    L.append(f"| {c['condition']} | {c['status']} | {c['worst'] if c['worst'] else '30위 밖 있음'} | {c['missed_30']} | {c['self_in_top']} | {c['model_recorded']}/35 | {c['summary_matches_recomputed']} | ${c['cost_cli_usd']} | ${c['cost_list_price_nocache_usd']} |")
    for p in c['problems']: L.append(f'  - {p}')
new = not (O / 'VERIFY_LOG_harness.md').exists()
with open(O / 'VERIFY_LOG_harness.md', 'a', encoding='utf-8') as fh:
    if new: fh.write('# LLM 순위 실험 검증 기록\n\n질의별 파일의 top_raw에서 순위를 다시 계산해 summary.json과 대조한다. 정가는 claude-api 스킬 단가표(2026-11-03) x 토큰, 캐시 할인 없이 계산.\n')
    fh.write('\n'.join(L) + '\n')
print('\n'.join(L))
