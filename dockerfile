# MUDANÇA 1: Usando a imagem base completa em vez da 'slim' para incluir mais bibliotecas.
FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Instalação das dependências do sistema.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# MUDANÇA 2: Comando de depuração para verificar se a biblioteca foi instalada.
# Este comando será executado durante o build no Railway.
RUN echo "--- Verificando a instalação da biblioteca libGL.so.1 ---" && \
    find / -name "libGL.so.1" 2>/dev/null && \
    echo "--- Verificação concluída ---" || \
    echo "--- ALERTA: libGL.so.1 NÃO FOI ENCONTRADO APÓS A INSTALAÇÃO! ---"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/uploads

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]