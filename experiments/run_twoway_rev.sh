#!/bin/bash
# Two-way bootstrap cache, one configuration at a time (resumable: cached configurations are skipped).
# The matched-condition configurations wait until their embeddings are complete.
set -e
cd /home/ali/Desktop/Articles/artifacts/speaker-id-wake
source experiments/run_eval_rev_lib.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for c in eres2net_base_zh titanet_small wespeaker_campplus_vox_wsfe wespeaker_resnet34_vox_wsfe wespeaker_resnet34_cnc_wsfe \
         campplus_zhen_adv_d1 campplus_zhen_adv_cip0 campplus_zhen_adv_np128 campplus_zhen_adv_ort131 \
         campplus_zhen_adv__cw eres2net_base_zh__cw titanet_small__cw wespeaker_campplus_vox_wsfe__cw \
         wespeaker_resnet34_vox_wsfe__cw wespeaker_resnet34_cnc_wsfe__cw \
         campplus_zhen_adv_cip0_d1 titanet_small_d1 eres2net_base_zh_d1; do
  until ready "$c" main 2>/dev/null; do sleep 60; done
  python experiments/22_twoway_bootstrap.py --only "$c" >> logs/twoway_v2.log 2>&1
done
python experiments/22_twoway_bootstrap.py > logs/twoway_v2_assemble.log 2>&1
echo "two-way done $(date -Is)" > logs/twoway_v2.done
