# ---------------------------------------------------------------------------
# Dockerfile — subliminal-learning thesis pipeline (QLoRA fine-tuning +
# generation + gradient selection + analysis). Built for a multi-GPU box so a
# 70B model in 4-bit can be sharded across GPUs via accelerate/device_map.
#
# The CUDA version below MUST be compatible with the HOST GPU driver on the
# target machine. 12.4.1 works with reasonably recent NVIDIA drivers (>=550).
# If `docker compose up` fails with a CUDA/driver mismatch, change the tag to
# match the host (check with `nvidia-smi` on the host: top-right "CUDA Version").
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# --- system deps -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# --- python deps (layer cached unless requirements change) -----------------
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# --- project code ----------------------------------------------------------
COPY . .

# Make the launcher executable inside the image
RUN chmod +x /app/run.sh || true

# No CMD that does work: the container is kept alive by docker-compose
# (command: sleep infinity) and jobs are started via `docker exec` + run.sh,
# OR run.sh is invoked directly as the compose command. See README.
