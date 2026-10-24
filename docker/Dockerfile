# The organizers' base image: your code is re-run inside this image with no network.
# Build it once:   docker build -t eurs-base docker/
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
RUN pip install --no-cache-dir numpy pillow opencv-python-headless timm einops onnx pyyaml scipy scikit-image
WORKDIR /pkg
