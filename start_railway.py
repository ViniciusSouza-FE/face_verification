#!/usr/bin/env python3
import os
import subprocess
import sys

if __name__ == "__main__":
    port = os.environ.get("PORT", "8080")
    
    # Configurações otimizadas para Railway
    cmd = [
        "gunicorn", 
        "--bind", f"0.0.0.0:{port}",
        "--workers", "1",
        "--threads", "2",
        "--timeout", "120",
        "app:app"
    ]
    
    print(f"🚀 Iniciando servidor na porta {port}")
    subprocess.run(cmd)