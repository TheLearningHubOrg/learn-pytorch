# Production Dockerfile for the Learn PyTorch companion project
FROM python:3.13-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

FROM python:3.13-slim AS runner
RUN useradd --create-home appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/appuser
USER appuser
CMD ["python", "-m", "learn_pytorch", "--epochs", "20", "--out", "artifacts"]
