#!/usr/bin/env bash
set -euo pipefail

# 停止 Celery worker 脚本（客服 Agent 项目）
# 默认匹配：celery -A celery_app worker --queues=chat_queue
# 可选参数：
#   第1个参数：Celery app 名称（默认 celery_app）
#   第2个参数：队列名（默认 chat_queue）
#
# 使用示例：
#   bash stop_celery.sh
#   bash stop_celery.sh celery_app chat_queue

APP_NAME="${1:-celery_app}"
QUEUE_NAME="${2:-chat_queue}"

uniq_pids() {
  tr ' ' '\n' | sed '/^$/d' | sort -u
}

collect_celery_pids() {
  local pids_a pids_b pids_c

  # 精准匹配当前项目常用启动参数
  pids_a="$(pgrep -f "celery -A ${APP_NAME} worker.*--queues[= ]${QUEUE_NAME}" 2>/dev/null || true)"

  # 兜底匹配：只要是该 app 的 worker 也尝试纳入
  pids_b="$(pgrep -f "celery -A ${APP_NAME} worker" 2>/dev/null || true)"

  # 再兜底：匹配所有 celery worker（避免僵尸进程残留）
  pids_c="$(pgrep -f "celery.*worker" 2>/dev/null || true)"

  printf "%s\n%s\n%s\n" "${pids_a}" "${pids_b}" "${pids_c}" | uniq_pids
}

stop_pid_gracefully() {
  local pid="$1"

  echo "[停止Celery] 正在优雅停止进程，PID=${pid} ..."
  kill -TERM "${pid}" 2>/dev/null || true

  # 等待最多 5 秒，让 worker 有时间处理收尾逻辑
  for _ in {1..10}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[停止Celery] 进程已停止，PID=${pid}"
      return 0
    fi
    sleep 0.5
  done

  echo "[停止Celery] 优雅停止超时，尝试强制结束，PID=${pid}"
  kill -KILL "${pid}" 2>/dev/null || true
  echo "[停止Celery] 已强制结束进程，PID=${pid}"
}

main() {
  echo "============================================================"
  echo "[停止Celery] 开始停止 Celery Worker"
  echo "[停止Celery] APP_NAME=${APP_NAME}"
  echo "[停止Celery] QUEUE_NAME=${QUEUE_NAME}"
  echo "============================================================"

  local pids
  pids="$(collect_celery_pids)"

  if [[ -z "${pids}" ]]; then
    echo "[停止Celery] 未发现需要停止的 Celery worker 进程。"
    exit 0
  fi

  echo "[停止Celery] 检测到以下 Celery worker 进程："
  echo "${pids}" | sed 's/^/[停止Celery] PID=/'

  echo "${pids}" | while read -r pid; do
    [[ -n "${pid}" ]] && stop_pid_gracefully "${pid}"
  done

  echo "============================================================"
  echo "[停止Celery] Celery Worker 停止流程执行完成。"
  echo "============================================================"
}

main "$@"
