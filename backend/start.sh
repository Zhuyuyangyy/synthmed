#!/bin/bash
# 启动多模态手语翻译系统后端

cd /mnt/d/ZYY\ Project/synthmed/backend

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
