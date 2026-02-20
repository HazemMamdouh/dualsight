# Deployment notes

## Live dual-sensor ingest with GStreamer

`pip install opencv-contrib-python` ships **without** GStreamer. For live feeds use an OpenCV
build with `GStreamer: YES` (check with `python -c "import cv2; print(cv2.getBuildInformation())"`):

* **Jetson (JetPack)** – the system OpenCV already has GStreamer; install DualSight with
  `pip install --no-deps -e .` and add the remaining requirements without `opencv-*`.
* **x86 + NVIDIA** – build OpenCV with `-D WITH_GSTREAMER=ON -D WITH_CUDA=ON -D OPENCV_DNN_CUDA=ON`
  (this also enables the `cuda_farneback` / `nvof` optical-flow back-ends).

Without GStreamer, DualSight falls back to the FFmpeg backend automatically and logs a warning.

### Source forms accepted by `--thermal` / `--rgb`

| form | example | pipeline built |
|---|---|---|
| file | `data/demo/thermal.mp4` | `filesrc ! decodebin ! videoconvert ! appsink` |
| RTSP | `rtsp://10.0.0.21/stream1` | `rtspsrc latency=100 ! decodebin ! …` |
| RTP/UDP | `udp://0.0.0.0:5000` | `udpsrc ! rtpjitterbuffer ! rtph264depay ! h264parse ! avdec_h264 ! …` |
| V4L2 | `/dev/video0` or `0` | `v4l2src ! …` |
| raw | `gst:<your pipeline ending in appsink>` | used verbatim |

`--hw-decode jetson` swaps the decoder for `nvv4l2decoder ! nvvidconv`, `--hw-decode nvidia` for `nvh264dec`.

### Simulating the live rig from recordings

```bash
# terminal 1 - two clock-synchronised RTP streams from one gst-launch pipeline
scripts/stream_pair_udp.sh data/demo/thermal.mp4 data/demo/rgb.mp4
# terminal 2
dualsight run --thermal udp://0.0.0.0:5000 --rgb udp://0.0.0.0:5002 --gstreamer --show --save-video
```

Without a network hop, `--realtime` paces local files at their native frame rate and **drops
frames when processing falls behind**, i.e. it reproduces live-stream behaviour on a laptop.

### Synchronisation

The thermal channel is the master clock. For every thermal frame the RGB frame with the closest
presentation timestamp is chosen; differing frame rates are therefore resampled automatically.
Pairs whose skew exceeds `sources.sync_tolerance_ms` are counted in `run.json → streams.sync_violations`.

## Registration between the two sensors

Detections from both channels are fused in RGB pixel coordinates. For a co-boresighted head the
default pure-scale mapping is enough. Otherwise put the thermal→RGB homography (from a heated
checkerboard calibration) in the config:

```yaml
registration:
  thermal_to_rgb: [[2.01, 0.004, -11.3], [-0.002, 1.99, 6.8], [0, 0, 1]]
```

## GPU paths

| stage | CPU | GPU |
|---|---|---|
| thermal SR | – | Real-ESRGAN (`torch`, fp16, optional tiling for small VRAM) |
| RGB deblur | Wiener in `torch.fft` | same code on CUDA |
| optical flow | DIS | `cuda_farneback` / `nvof` (OpenCV CUDA) or RAFT-small (`torch`) |
| attribution | YOLO11 (`torch`) | YOLO11 fp16, or a TensorRT `.engine` exported with Ultralytics |

TensorRT for the detector: `yolo export model=yolo11n.pt format=engine half=True`, then
`dualsight run … --detector yolo11n.engine`.

## Calibrating the gates for a new camera

Every frame's gate inputs are logged in `frames.csv` (`fog_score`, `fog_spread`, `fog_grad`,
`cam_shift`). Run once on a clear and once on a foggy recording, look at the two distributions and
set `thermal.gate.on_score/off_score` (and `spread_ref/grad_ref`) between them. The same applies
to `rgb.gate.shift_px` for the camera-motion gate.
