#!/bin/bash

# Configurações otimizadas para Render free
export TF_CPP_MIN_LOG_LEVEL=3
export OPENCV_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=-1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export DEEPFACE_BACKEND=opencv

echo "🚀 Iniciando Face Confirmation no Render (Configuração Conservadora)..."
echo "📦 Python version: $(python --version)"

# Configurações conservadoras para Gunicorn no Render free
exec gunicorn --bind 0.0.0.0:$PORT \
              --workers 1 \
              --threads 1 \
              --timeout 180 \
              --max-requests 50 \
              --max-requests-jitter 10 \
              app:app