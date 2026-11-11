#!/bin/bash
# Revision evaluation lane E: ERes2Net-base main set now (lane 2 of run_eval_rev.sh would reach it
# only after the trimming embeddings; its trimming ablation is analysed by 24_ablation_paired.py).
set -e
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
source experiments/run_eval_rev_lib.sh
run eres2net_base_zh main
echo "lane E all done $(date -Is)" > logs/eval_rev_laneE.done
