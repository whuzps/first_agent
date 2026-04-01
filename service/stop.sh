#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"

uniq_pids() {
  tr ' ' '\n' | sed '/^$/d' | sort -u
}

collect_pids() {
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -nP -tiTCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)"
  fi
  local p1 p2
  p1="$(pgrep -f "python3 -m uvicorn.*work\.app:app.*--port[= ]${PORT}" 2>/dev/null || true)"
  p2="$(pgrep -f "[u]vicorn.*work\.app:app.*--port[= ]${PORT}" 2>/dev/null || true)"
  printf "%s\n%s\n%s\n" "${pids}" "${p1}" "${p2}" | uniq_pids
}

stop_pid() {
  local pid="$1"
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in {1..10}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "stopped ${pid}"
      return 0
    fi
    sleep 0.3
  done
  kill -KILL "${pid}" 2>/dev/null || true
  echo "killed ${pid}"
}

main() {
  local pids
  pids="$(collect_pids)"
  if [[ -z "${pids}" ]]; then
    echo "no service found on port ${PORT}"
    exit 0
  fi
  echo "${pids}" | while read -r pid; do
    [[ -n "${pid}" ]] && stop_pid "${pid}"
  done
}

main "$@"
