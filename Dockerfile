# Real-Time Colonoscopy Polyp Segmentation System - Docker image
FROM python:3.11-slim

# System libraries required by OpenCV and the FFmpeg video pipeline
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch and torchvision first (keeps the image small), then the remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Install the polypseg package and copy configs, scripts and tests
COPY pyproject.toml README.md Makefile ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Run the real-time pipeline on a HyperKvasir polyp video (data/ and runs/ are mounted)
CMD ["python", "-m", "polypseg.video", \
     "--model", "runs/unet_r34_neg/best.onnx", \
     "--input", "data/raw/hyperkvasir-videos/polyps/c0c92730-bf69-4ee1-a625-f5a31c153034.avi", \
     "--output", "outputs/unet_r34_neg/videos/c0c92730.mp4", \
     "--events", "outputs/unet_r34_neg/videos/c0c92730.events.csv"]
