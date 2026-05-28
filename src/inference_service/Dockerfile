# ── Ribbit AI — SageMaker inference container ─────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY ribbit_ai/pyproject.toml /build/pyproject.toml
RUN cd /build && uv pip install --system --no-cache ".[backend]"

# Example of the folder structure
#/Users/{user}/src/ribbit_ai/BirdNET-Analyzer <- github clone
#/Users/{user}/src/ribbit_ai/V2.4 <- BirdNet Model folder
#/Users/{user}/src/ribbit_ai/ribbit_ai <- actual working folder

COPY V2.4/ /usr/local/lib/python3.11/site-packages/birdnet_analyzer/checkpoints/V2.4/
COPY ribbit_ai/src/inference_service/sagemaker.py  /app/sagemaker_handler.py
COPY ribbit_ai/src/inference_service/serve.py      /app/serve.py

WORKDIR /app

EXPOSE 8080

RUN printf '#!/bin/bash\nuvicorn serve:app --host 0.0.0.0 --port 8080\n' \
    > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

CMD ["serve"]