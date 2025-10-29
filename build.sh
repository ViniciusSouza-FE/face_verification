#!/bin/bash

echo "🔧 Iniciando build otimizado no Render..."
echo "📦 Python version: $(python --version)"

# Configurações de ambiente
export TF_CPP_MIN_LOG_LEVEL=3
export OPENCV_LOG_LEVEL=ERROR
export CUDA_VISIBLE_DEVICES=-1

echo "📦 Instalando dependências Python..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "✅ Build concluído!"