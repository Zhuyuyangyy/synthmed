#!/bin/bash
# SynthMed 启动脚本
# AI医学影像合成平台 — Diffusion + ControlNet + DICOM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  SynthMed - AI医学影像合成平台"
echo "  Diffusion + ControlNet + DICOM处理"
echo "=============================================="
echo

# 虚拟环境
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install -q fastapi uvicorn torch pillow pyyaml 2>/dev/null

echo "[启动] 服务运行于 http://localhost:8090"
echo "[启动] API文档: http://localhost:8090/docs"
cd "$SCRIPT_DIR/backend"
python3 app.py