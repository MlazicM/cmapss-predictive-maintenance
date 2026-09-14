# Inference image for the CMAPSS RUL service.
#
# Two stages. The builder compiles wheels into a virtualenv; the runtime copies
# that virtualenv and nothing else. It keeps the compilers, caches and headers
# that TensorFlow's install drags in out of the shipped image, which is worth
# doing when the dependency is this large.

# --- builder ---------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only the serving dependencies. requirements.txt also pins Jupyter, matplotlib
# and seaborn for the notebooks; none of them are reachable from api/ or src/,
# and shipping them would roughly double the image for code that never runs.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt


# --- runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # TensorFlow logs three INFO banners per import; on a serving container that
    # is noise in front of every real log line.
    TF_CPP_MIN_LOG_LEVEL=2

COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user. The service reads model files and answers HTTP; it has
# no reason to hold root inside the container, and an image that does is one
# escaped bug away from being a problem.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/

# Trained artifacts are deliberately NOT baked in. They are gitignored, they are
# produced by scripts/run_experiments.py, and an image that carries a model is an
# image that has to be rebuilt to ship a retrained one. Mount models/ at runtime:
#   docker run -v "$PWD/models:/app/models:ro" ...
# With nothing mounted the service still starts and /health reports
# model_loaded=false, which is the documented degraded state rather than a crash.
RUN mkdir -p /app/models && chown appuser:appuser /app/models

USER appuser
EXPOSE 8000

# Hits the endpoint that answers without a model, so the check measures whether
# the process is serving rather than whether an artifact happens to be mounted.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
