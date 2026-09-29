# ChatBoat — pinned build.
#
# Why Docker: the service's Python runtime is fixed at creation on Render, and the
# platform's default moved to 3.14, which has no wheels for the pinned native stack
# (tokenizers 0.20.3, chromadb 1.5.9, onnxruntime 1.30.0) — so native builds compile
# them from source and die on Render's read-only /usr/local/cargo. Pinning Python
# inside the image (3.12, the version the whole stack is tested on) removes the
# platform's runtime from the equation entirely.
#
# The image is self-sufficient: torch CPU, the pinned requirements, the committed
# offline ONNX encoder under models/, and the Chroma index BAKED IN at build time, so
# cold starts have nothing to download and nothing to build.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/render/project/src

# Compile toolchain as insurance for any wheel that lacks a cp312 build; otherwise
# every pinned package has a prebuilt cp312 wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (cache-friendly). torch from the CPU index, then the pinned set.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# The app: sources, data, committed ONNX model.
COPY . .

# Bake the search index into the image (offline: chunks + ONNX model are in the repo).
RUN python -m mf_rag.cli embed

# $PORT is injected by Render at runtime; shell form expands it.
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true