#!/usr/bin/env bash
# 2026-11-12 하네스 세션: 기준 모델(qwen_cropB3batch128_mem5_s42_20261106_night3 = B3 설정 + BATCH=128 + MAX_STEPS=5)과 같은 설정으로 seed만 바꿔 반복.
# night_queue2.sh의 drun/train이 만드는 docker 명령과 같은 env·mount(대기열·마감·훅 없이 순차 실행). 끝난 run은 건너뜀.
# usage: bash seed_repeat_b3b128_mem5_harness.sh 43 45 46
B=<DEV_EVAL_ROOT>/jh_dev/embedding_ft_docker; K=<DEV_EVAL_ROOT>/jh_dev/<EXPERIMENT_DIR>
MODEL_DIR='<MODEL_DIR>'
IMAGE=embed-ft:20261102-v1; TAG=20261112_harness
for S in "$@"; do
  R=qwen_cropB3batch128_mem5_s${S}_$TAG; out=$B/output/$R
  if [ -f $out/manifest.json ] && [ -f $out/qwen_chunk.npz ]; then echo "$(date +%T) skip $R"; continue; fi
  echo "$(date +%T) start $R"
  MSYS_NO_PATHCONV=1 docker run --rm --name "embed-ft-$R" --gpus all --shm-size=16g -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HOME=/tmp \
    -e RUN_ID=$R -e CORPUS=train_docs_v2.jsonl -e "HYPOTHESIS=11-12 하네스: 기준 모델(B3+batch128, 5스텝)의 seed 반복으로 668 풀 최악 순위의 seed 범위 확인" \
    -e PAIRS_FILE=crop -e TRAIN_LEN=512 -e LIMIT=9321 -e LR=2e-4 -e SEED=$S -e BATCH=128 -e MAX_STEPS=5 \
    -v "$MODEL_DIR:/models:ro" -v "$B/code:/code:ro" -v "$B/data:/data:ro" -v "$K/runs:/req:ro" -v "$B/output:/output" -w /code $IMAGE python /code/qwen_simcse.py \
    >> "$B/logs/$R.log" 2>&1
  echo "$(date +%T) done $R rc=$? manifest=$([ -f $out/manifest.json ] && echo yes || echo no)"
done
