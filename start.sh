#!/bin/bash

# Configura variáveis de ambiente para modo headless
export TF_CPP_MIN_LOG_LEVEL=3
export OPENCV_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=-1

echo "Iniciando Face Confirmation no Render..."
echo "Verificando dependências..."

# Inicia a aplicação
exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 app:app