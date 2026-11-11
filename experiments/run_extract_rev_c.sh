#!/bin/bash
# Revision queue C (11 Nov, after the machine rebooted at 08:10 and queues A/B2 died part-way).
# Finishes the matched-condition embeddings of the three non-WeSpeaker models with 6 workers,
# fastest model first: CAM++ patched + noise floor, TitaNet-small + noise floor, ERes2Net-base +
# noise floor. 02_extract.py resumes from the rows already embedded (NaN = not done).
# Dropped for lack of time: the two 3D-Speaker VoxCeleb models (campplus_vox_cip0_d1,
# eres2net_vox_d1), the trimming specs of eres2net_base_zh / campplus_zhen_adv_cip0 and the other
# ablation specs of campplus_zhen_adv_cip0.
set -e
source /home/ali/Desktop/Articles/.venv/bin/activate
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
step() {
  local n="$1"; shift
  if [ -f "logs/extract_rev_$n.done" ]; then echo "skip $n"; return; fi
  python experiments/02_extract.py "$@" >> "logs/extract_rev_$n.log" 2>&1
  date -Is > "logs/extract_rev_$n.done"
  echo "extracted $n $(date -Is)"
}
[ -f logs/extract_rev_cip0_d1.log.part ] && { cat logs/extract_rev_cip0_d1.log.part >> logs/extract_rev_cip0_d1.log; rm -f logs/extract_rev_cip0_d1.log.part; }
step cip0_d1 --set main --models campplus_zhen_adv_cip0_d1 --workers 6 --chunk 300
step titanet_d1 --set main --models titanet_small_d1 --workers 6 --chunk 300
step eres2net_d1 --set main --models eres2net_base_zh_d1 --workers 6 --chunk 300
echo "queue C done $(date -Is)" > logs/extract_rev_c.alldone
