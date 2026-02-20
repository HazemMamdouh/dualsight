#!/usr/bin/env bash
# Replay two recorded files as two *live, clock-synchronised* RTP/H.264 streams.
# Both branches live in ONE gst-launch pipeline, so they share a clock and start together -
# exactly how a dual-sensor head streams. Then consume them with:
#   dualsight run --thermal udp://0.0.0.0:5000 --rgb udp://0.0.0.0:5002 --gstreamer --show
set -euo pipefail
THERMAL=${1:-data/demo/thermal.mp4}
RGB=${2:-data/demo/rgb.mp4}
HOST=${HOST:-127.0.0.1}
gst-launch-1.0 -e \
  filesrc location="$THERMAL" ! decodebin ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 key-int-max=30 \
    ! rtph264pay config-interval=1 pt=96 ! udpsink host="$HOST" port=5000 sync=true \
  filesrc location="$RGB" ! decodebin ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=6000 key-int-max=30 \
    ! rtph264pay config-interval=1 pt=96 ! udpsink host="$HOST" port=5002 sync=true
