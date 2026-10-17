#!/usr/bin/env bash
# Reproduce every number in the README: data -> train -> evaluate -> error analysis -> export
# -> latency benchmark -> temporal and false-alarm evaluation -> demo videos.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/download_data.py
python scripts/fetch_negatives.py --per-class 300

for cfg in unet_r34 deeplabv3p_r34 unet_r34_neg; do
  python -m polypseg.train --config "configs/${cfg}.yaml"
  ckpt="runs/${cfg}/best.pt"
  python -m polypseg.evaluate --ckpt "$ckpt" --dataset kvasir-val
  python -m polypseg.evaluate --ckpt "$ckpt" --dataset cvc
  python -m polypseg.error_analysis --eval-dir "outputs/${cfg}/cvc"
  python -m polypseg.video_eval --ckpt "$ckpt"
  python -m polypseg.export --ckpt "$ckpt"
done

# held-out polyp-free frames (the split is defined by the unet_r34_neg run)
for cfg in unet_r34 deeplabv3p_r34 unet_r34_neg; do
  python -m polypseg.evaluate --ckpt "runs/${cfg}/best.pt" --dataset negatives-val
  python -m polypseg.false_alarms --model "runs/${cfg}/best.pt" \
    --videos data/raw/hyperkvasir-videos/normal --out "outputs/${cfg}/false_alarms"
done

# sensitivity vs false alarms across thresholds
for cfg in unet_r34_neg unet_r34; do
  python -m polypseg.operating_point --ckpt "runs/${cfg}/best.pt" --videos data/raw/hyperkvasir-videos/normal
done

# latency on an otherwise idle machine
for cfg in unet_r34 deeplabv3p_r34; do
  python -m polypseg.benchmark --ckpt "runs/${cfg}/best.pt"
done

for v in data/raw/hyperkvasir-videos/polyps/*.avi; do
  n=$(basename "$v" .avi | cut -c1-8)
  python -m polypseg.video --model runs/unet_r34_neg/best.pt --input "$v" \
    --output "outputs/unet_r34_neg/videos/${n}.mp4" --events "outputs/unet_r34_neg/videos/${n}.events.csv"
done
