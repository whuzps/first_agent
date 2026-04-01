#!/bin/bash
# 关闭前端服务脚本（Vite 开发服务器 / 预览服务器）

echo "🔍 正在查找前端服务进程..."

# 查找 vite 相关进程（dev 和 preview 模式）
PIDS=$(lsof -ti :5173 2>/dev/null)

if [ -z "$PIDS" ]; then
  echo "✅ 没有发现正在运行的前端服务（端口 5173）"
  exit 0
fi

echo "📋 发现以下进程占用端口 5173："
lsof -i :5173 2>/dev/null

echo ""
echo "⏳ 正在关闭进程..."

for PID in $PIDS; do
  kill "$PID" 2>/dev/null
  if [ $? -eq 0 ]; then
    echo "  ✔ 已发送终止信号给进程 $PID"
  else
    echo "  ✘ 无法终止进程 $PID，尝试强制关闭..."
    kill -9 "$PID" 2>/dev/null
  fi
done

sleep 1

# 验证是否已关闭
REMAINING=$(lsof -ti :5173 2>/dev/null)
if [ -z "$REMAINING" ]; then
  echo "✅ 前端服务已成功关闭"
else
  echo "⚠️  仍有进程未关闭，尝试强制终止..."
  kill -9 $REMAINING 2>/dev/null
  sleep 1
  FINAL=$(lsof -ti :5173 2>/dev/null)
  if [ -z "$FINAL" ]; then
    echo "✅ 前端服务已强制关闭"
  else
    echo "❌ 无法关闭前端服务，请手动处理"
    exit 1
  fi
fi
