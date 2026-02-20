# GPU image (x86_64 + NVIDIA). For Jetson, start from an L4T PyTorch base instead (see docs/deployment.md).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
      gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["dualsight"]
CMD ["--help"]
