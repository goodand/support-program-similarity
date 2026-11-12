"""격리 claude -p 순위 매기기, 질의 35개 전체. probe와 같은 격리 옵션(workflow-runner claude_adapter + --effort).
캐시를 위해 후보 목록(668개 전체, 고정 순서)을 앞에, 질의를 뒤에 둔다. 질의 자신의 번호는 답에서 빼라고 지시하고, 채점에서도 제외한다.
질의별 결과 파일이 있으면 건너뛴다(재개 가능).
usage: .venv/Scripts/python.exe -X utf8 runs/044/llm_rank_run_harness.py <model> <effort> <name|chunk>
출력: runs/044/llm_rank_harness/<model>_<effort>_<inp>/q<NN>.json, summary.json"""
import json, os, sys, shutil, subprocess, tempfile, time, hashlib
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.environ['WORKFLOW_RUNNER_DIR'])  # adapters._safe_child_env 제공
from adapters import _safe_child_env
MODEL, EFF, INP = sys.argv[1], sys.argv[2], sys.argv[3]
B = Path('runs/044/pool668_newreview_harness'); O = Path(f'runs/044/llm_rank_harness/{MODEL}_{EFF}_{INP}'); O.mkdir(parents=True, exist_ok=True)
R = [json.loads(l) for l in open(B / 'chunks.jsonl', encoding='utf-8')]
keys = json.load(open(B / 'texts.json', encoding='utf-8'))['keys']
assert keys == [r['type'] + '|' + json.dumps(r['id'], ensure_ascii=False) for r in R], 'keys 순서 불일치'
kix = {k: i for i, k in enumerate(keys)}
ev = json.load(open(B / 'eval_set_pool668.json', encoding='utf-8'))
T = defaultdict(set)
for g, a, b in ev['pairs']: T[g] |= {kix[a], kix[b]}
QS = sorted((g, q) for g, m in T.items() for q in m); assert len(QS) == 35
txt = (lambda r: r['name']) if INP == 'name' else (lambda r: r['text'])
W = Path(os.environ['ISO_CWD'])  # 11-12: 임시 폴더 이름이 호출마다 달라 환경 블록이 바뀌고 캐시 읽기가 0이었음 -> 고정 빈 폴더
SYS = '주어진 요청에만 답하세요.'
HEAD = ('[후보 목록]\n' + '\n'.join(f'{i}\t{txt(R[i])}' for i in range(len(R))) +
        '\n\n위 후보 목록에서, 아래 질의 사업과 가장 비슷한 사업을 비슷한 순서로 상위 30개 골라 후보 번호를 JSON으로만 답하라. '
        '질의 사업 자신의 번호는 넣지 마라. 형식: {"top": [번호, ...]}. 설명은 쓰지 마라.\n\n[질의]\n')
args = [shutil.which('claude') or 'claude', '-p', '--setting-sources', '', '--tools', '', '--strict-mcp-config',
        '--mcp-config', '{"mcpServers":{}}', '--system-prompt', SYS, '--exclude-dynamic-system-prompt-sections',
        '--disable-slash-commands', '--no-session-persistence', '--model', MODEL, '--effort', EFF] + (['--output-format', 'stream-json', '--verbose'] if os.environ.get('STREAM') == '1' else ['--output-format', 'json'])  # 11-12: STREAM=1이면 블록 전체 기록
print('start', MODEL, EFF, INP, 'head_sha', hashlib.sha256(HEAD.encode()).hexdigest()[:16], flush=True)
for qi, (g, q) in enumerate(QS):
    p = O / f'q{qi:02d}.json'
    if p.exists(): continue
    mates = sorted(T[g] - {q}); t0 = time.time(); d = None
    for attempt in range(3):
        W.mkdir(exist_ok=True); assert not any(W.iterdir()), '고정 작업 폴더가 비어 있지 않음'
        cp = subprocess.run(args, input=HEAD + f'{q}\t{txt(R[q])}', cwd=str(W), env=_safe_child_env(W),
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
        try:
            if os.environ.get('STREAM') == '1':  # 마지막 result 이벤트를 d로, 원문 스트림은 따로 저장
                ev_ = [json.loads(x) for x in cp.stdout.splitlines() if x.strip()]
                (O / f'q{qi:02d}_stream.jsonl').write_text(cp.stdout, encoding='utf-8')
                d = [e for e in ev_ if e.get('type') == 'result'][-1]
                if not d.get('result'):  # 결과 칸이 비면 assistant 텍스트 블록을 이어 붙여 사용(기록에 남김)
                    tx = [b.get('text', '') for e in ev_ if e.get('type') == 'assistant' for b in (e.get('message') or {}).get('content', []) if b.get('type') == 'text']
                    d = {**d, 'result': ''.join(tx), 'result_from_stream_text_blocks': True}
            else:
                d = json.loads(cp.stdout)
            break
        except ValueError:
            print('retry', qi, cp.returncode, cp.stderr[:200], flush=True); d = None
    res = (d or {}).get('result', ''); top = None
    try:  # 11-12: 첫 JSON 객체만 읽음(뒤에 다른 글이 붙으면 rindex 방식이 실패했음, Opus 7건)
        top = [int(x) for x in json.JSONDecoder().raw_decode(res[res.index('{'):])[0]['top']]
    except Exception:
        pass
    clean = list(dict.fromkeys(x for x in (top or []) if x != q and 0 <= x < len(R)))  # 11-12: 같은 번호 반복은 첫 등장만(350개 응답 중 43개에 반복)
    rk = {str(m): (clean.index(m) + 1 if m in clean else None) for m in mates}
    out = {'qi': qi, 'group': g, 'query': q, 'mates': mates, 'mate_rank': rk, 'top_raw': top,
           'self_in_top': bool(top and q in top), 'invalid': [x for x in (top or []) if not 0 <= x < len(R)],
           'seconds': round(time.time() - t0, 1), 'usage': (d or {}).get('usage'), 'modelUsage': (d or {}).get('modelUsage'),'total_cost_usd': (d or {}).get('total_cost_usd'),
           'is_error': (d or {}).get('is_error', True), 'result_head': res[:200], 'result': res}
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(qi, g, rk, out['seconds'], out['total_cost_usd'], flush=True)
rows = [json.load(open(O / f'q{i:02d}.json', encoding='utf-8')) for i in range(35)]
last = [None if any(v is None for v in r['mate_rank'].values()) else max(r['mate_rank'].values()) for r in rows]
known = [x for x in last if x is not None]
S = {'model': MODEL, 'effort': EFF, 'input': INP, 'n': 35, 'missed_queries': sum(x is None for x in last),
     'worst': (max(known) if known and None not in last else '30위 밖 있음'), 'worst_known': max(known) if known else None,
     'mean_known': sum(known) / len(known) if known else None, 'last': last,
     'cost_usd_cli': round(sum(r['total_cost_usd'] or 0 for r in rows), 4), 'seconds': round(sum(r['seconds'] for r in rows), 1),
     'errors': sum(bool(r['is_error']) for r in rows), 'self_in_top': sum(r['self_in_top'] for r in rows)}
json.dump(S, open(O / 'summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('SUMMARY', json.dumps({k: v for k, v in S.items() if k != 'last'}, ensure_ascii=False))
