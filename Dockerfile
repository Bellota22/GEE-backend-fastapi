# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Evita pyc y buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# Copia requisitos e instala
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copia código y credencial
COPY . /app/
# COPY key.json /app/key.json

# Señala credencial (puedes sobreescribir en compose)
# ENV GOOGLE_APPLICATION_CREDENTIALS=/app/key.json

EXPOSE 8000

# Para dev puedes sobreescribir este CMD en compose
CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
