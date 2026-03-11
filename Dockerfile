# ============================================================================
# AI Service - Production Dockerfile
# ============================================================================
# Multi-stage build for a lean, secure production image.
#
# Build:  docker build -t ai-service .
# Run:    docker run --env-file .env -p 8000:8000 ai-service
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder - install dependencies into a virtual-env
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile native wheels (numpy, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual-env so we can copy only the finished env to the next stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (layer cache optimisation)
# --extra-index-url for CPU-only PyTorch builds (~200MB vs ~2GB+ with CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt && \
    # ── Post-install cleanup: strip tests, docs, and caches from site-packages ──
    find /opt/venv/lib -type d -name "tests"        -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv/lib -type d -name "test"         -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv/lib -type d -name "__pycache__"  -exec rm -rf {} + 2>/dev/null; \
    find /opt/venv/lib -type f -name "*.pyc"        -delete 2>/dev/null; \
    find /opt/venv/lib -type f -name "*.pyo"        -delete 2>/dev/null; \
    find /opt/venv/lib -type f -name "*.a"          -delete 2>/dev/null; \
    find /opt/venv/lib -type f -name "*.js.map"     -delete 2>/dev/null; \
    true

# Stage 2: Production
FROM python:3.12-slim AS production

LABEL maintainer="Mohammed Khalid <lordy.khalid@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/Moeen-Law/ai-service"
LABEL org.opencontainers.image.description="AI Services and LLMs for Moeen Law"
LABEL org.opencontainers.image.authors="Ahmed Alaa <AhmedAlaa277143@gmail.com, Zeyad Mahmoud <zeyadmahmoud2592004@gmail.com>"

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/false --create-home appuser

# Copy the pre-built virtual-env from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY main.py .

# Ensure the non-root user owns the app directory
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the default service port
EXPOSE 8000

# Health-check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health | grep "healthy" || exit 1

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]

# Start Uvicorn - production settings
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info --no-access-log"]
