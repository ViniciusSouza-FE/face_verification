# Use uma base Python slim, que é leve e eficiente.
FROM python:3.11-slim-bullseye

# Define variáveis de ambiente para evitar prompts interativos durante a instalação.
ENV DEBIAN_FRONTEND=noninteractive

# Instala as dependências do sistema necessárias para OpenCV e outras bibliotecas.
# libgl1: Fornece o libGL.so.1, que era a causa do erro.
# libglib2.0-0: Dependência comum para aplicações GTK e outras.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do contêiner.
WORKDIR /app

# Copia o arquivo de requisitos primeiro para aproveitar o cache do Docker.
COPY requirements.txt .

# Instala as dependências do Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da sua aplicação.
COPY . .

# Cria o diretório de uploads para as imagens.
RUN mkdir -p static/uploads

# Define as variáveis de ambiente para a aplicação.
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expõe a porta que o Gunicorn irá usar.
EXPOSE 8080

# Comando para iniciar o servidor Gunicorn quando o contêiner for executado.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]