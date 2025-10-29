#!/bin/bash

# Configura variáveis de ambiente para modo headless
export TF_CPP_MIN_LOG_LEVEL=3
export OPENCV_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=-1

echo "🚀 Iniciando aplicação no Railway..."
echo "📦 Verificando dependências..."

# Verifica se as bibliotecas do sistema estão presentes
ldd /app/.venv/lib/python3.11/site-packages/cv2/*.so | grep -i "not found" && echo "❌ Bibliotecas faltando!" || echo "✅ Bibliotecas do sistema OK"

# Inicia a aplicação
exec gunicorn --bind 0.0.0.0:8080 --workers 1 --threads 2 --timeout 120 app:app