"""격리 claude -p 시험(질의 1개). workflow-runner claude_adapter와 같은 격리 옵션 + --effort.
usage: .venv/Scripts/python.exe -X utf8 runs/044/llm_rank_probe_harness.py <model> <effort> <name|chunk> [qi]
출력: runs/044/llm_rank_harness/probe_<model>_<effort>_<inp>_q<qi>.json"""
import json, sys, shutil, subprocess, tempfile, time, os
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.environ['WORKFLOW_RUNNER_DIR'])  # adapters._safe_child_env 제공
from adapters import _safe_child_env  # 원본 그대로 사용 (API 키류 환경변수 제거)
MODEL, EFF, INP = sys.argv[1], sys.argv[2], sys.argv[3]; QI = int(sys.argv[4]) if len(sys.argv) > 4 else 0
B = Path('runs/044/pool668_newreview_harness'); O = Path('runs/044/llm_rank_harness'); O.mkdir(exist_ok=True)
R = [json.loads(l) for l in open(B / 'chunks.jsonl', encoding='utf-8')]
keys = json.load(open(B / 'texts.json', encoding='utf-8'))['keys']
assert keys == [r['type'] + '|' + json.dumps(r['id'], ensure_ascii=False) for r in R], 'keys 순서 불일치'
kix = {k: i for i, k in enumerate(keys)}
ev = json.load(open(B / 'eval_set_pool668.json', encoding='utf-8'))
T = defaultdict(set)
for g, a, b in ev['pairs']: T[g] |= {kix[a], kix[b]}
QS = sorted((g, q) for g, m in T.items() for q in m)
g, q = QS[QI]; mates = sorted(T[g] - {q})
txt = (lambda r: r['name']) if INP == 'name' else (lambda r: r['text'])
cands = [i for i in range(len(R)) if i != q]
SYS = '주어진 요청에만 답하세요.'
inst = ('아래 질의 사업과 가장 비슷한 사업을 후보 목록에서 골라, 비슷한 순서로 상위 30개의 후보 번호를 JSON으로만 답하라. '
        '형식: {"top": [번호, ...]}. 설명은 쓰지 마라.\n\n[질의]\n' + txt(R[q]) + '\n\n[후보]\n' +
        '\n'.join(f'{i}\t{txt(R[i])}' for i in cands))
args = [shutil.which('claude') or 'claude', '-p', '--setting-sources', '', '--tools', '', '--strict-mcp-config',
        '--mcp-config', '{"mcpServers":{}}', '--system-prompt', SYS, '--exclude-dynamic-system-prompt-sections',
        '--disable-slash-commands', '--no-session-persistence', '--model', MODEL, '--effort', EFF, '--output-format', 'json']
t0 = time.time()
with tempfile.TemporaryDirectory(prefix='iso_') as w:
    cp = subprocess.run(args, input=inst, cwd=w, env=_safe_child_env(Path(w)), capture_output=True, text=True,
                        encoding='utf-8', errors='replace', timeout=1800)
sec = time.time() - t0
try:
    d = json.loads(cp.stdout)
except ValueError:
    print('unparsable', cp.returncode, cp.stdout[:300], cp.stderr[:300]); sys.exit(1)
res = d.get('result', ''); top = None
try:
    top = json.loads(res[res.index('{'):res.rindex('}') + 1])['top']
except Exception:
    pass
rk = {m: (top.index(m) + 1 if top and m in top else None) for m in mates}
out = {'model': MODEL, 'effort': EFF, 'input': INP, 'query_index': QI, 'group': g, 'query': q, 'mates': mates, 'mate_rank': rk,
       'n_top': len(top) if top else None, 'invalid_ids': [x for x in (top or []) if x not in cands],
       'seconds': round(sec, 1), 'usage': d.get('usage'), 'modelUsage': d.get('modelUsage'), 'total_cost_usd': d.get('total_cost_usd'),
       'is_error': d.get('is_error'), 'result_head': res[:200], 'prompt_chars': len(inst)}
p = O / f'probe_{MODEL}_{EFF}_{INP}_q{QI}.json'
json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('model', 'effort', 'input', 'mate_rank', 'n_top', 'invalid_ids', 'seconds', 'total_cost_usd', 'is_error', 'prompt_chars')}, ensure_ascii=False))
print('usage', out['usage'])
