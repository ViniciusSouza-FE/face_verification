FROM python:3.11-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Instalação robusta e completa das dependências do sistema para OpenCV
# libgl1-mesa-glx: Fornece o libGL.so.1, que é a causa exata do erro.
# libglib2.0-0: Dependência comum de várias bibliotecas de baixo nível.
# Outras libs: Bibliotecas de suporte que podem ser chamadas indiretamente.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/uploads

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]