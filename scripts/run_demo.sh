#!/usr/bin/env bash
# One-shot demo: synthesize the data, run the pipeline, render the visualizations.
set -euo pipefail
DEVICE=${DEVICE:-auto}
dualsight make-demo --out data/demo
dualsight run --thermal data/demo/thermal.mp4 --rgb data/demo/rgb.mp4 \
  --out runs/demo --device "$DEVICE" --save-video --save-processed
dualsight render --run runs/demo --compare
echo "open runs/demo/report/index.html"
