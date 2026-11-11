#!/bin/bash
# Revision evaluation lane 3: the two added 3D-Speaker VoxCeleb models (matched condition).
# Same logic as run_eval_rev.sh (waits for complete embeddings, eval_rev.done markers).
set -e
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
source experiments/run_eval_rev_lib.sh
run campplus_vox_cip0_d1 main
run eres2net_vox_d1 main
echo "lane 3 all done $(date -Is)" > logs/eval_rev_lane3.done
