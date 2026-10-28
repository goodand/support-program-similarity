#!/usr/bin/env bash
# Unattended chain (사용자 지시 2026-10-28: 추가 결정·승인 없이 진행). Runs after the GPU pipeline (eval_pipeline.py) exits.
# Order is fixed; each step logs to runs/003/autopilot/; a failed step is logged and the chain continues where independent.
#  1. kNN + Leiden with Qwen for the 'all' and 'purpose_content' chunks (CPU, embeddings now cached)
#  2. select_n (explainable N) on 'all' vn+qwen, k=5 and k=10
#  3. Docker vLLM parity check (GPU) -> 4. stage2 (6,000-char embeddings) only if parity passed
# Nothing is pushed, deleted or installed. Docker container is removed after each use.
set -u
P="<DEV_EVAL_ROOT>/jh_dev/<EXPERIMENT_DIR>"; R="$P/runs/003"; A="$R/autopilot"; mkdir -p "$A"
EPY="<DEV_EVAL_ROOT>/_embedding_v3_env/Scripts/python.exe"; LPY="<DEV_EVAL_ROOT>/jh_dev/program_mapping_nlp_v1/.venv/Scripts/python.exe"
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$A/autopilot.log"; }
log "waiting for eval_pipeline.py"
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'eval_pipeline.py' }) { exit 0 } else { exit 1 }"; do sleep 120; done
log "pipeline finished; exit status in $R/pipeline_001/run.log"
for t in all purpose_content; do
  "$EPY" -I -X utf8 "$P/knn_build.py" 003 $t 30 > "$A/knn_${t}.log" 2>&1 && "$LPY" -X utf8 "$P/leiden_eval.py" 003 $t 30 > "$A/leiden_${t}.log" 2>&1
  log "knn+leiden $t (vn+qwen) exit $?"
done
for k in 5 10; do
  "$LPY" -X utf8 "$P/select_n.py" 003 all 30 $k vn+qwen 50 10 200 > "$A/select_all_vnqwen_k$k.log" 2>&1; log "select_n all vn+qwen k$k exit $?"
done
"$EPY" -I -X utf8 "$P/docker_vllm_embed.py" 003 parity > "$A/docker_parity.log" 2>&1; log "docker parity exit $?"
if grep -q '"pass": true' "$R/docker/parity.json" 2>/dev/null; then
  "$EPY" -I -X utf8 "$P/docker_vllm_embed.py" 003 stage2 > "$A/docker_stage2.log" 2>&1; log "docker stage2 exit $?"
else
  log "parity not passed -> stage2 skipped"
fi
log "autopilot done"
