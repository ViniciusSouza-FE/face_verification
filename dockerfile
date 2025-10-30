FROM python:3.11-bullseye

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primeiro para cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o resto da aplicação
COPY . .
# Criar diretório de uploads
RUN mkdir -p static/uploads

# Configurar variáveis de ambiente
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expor porta
EXPOSE 8080

# Comando de inicialização
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]