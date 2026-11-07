#!/usr/bin/env bash
# One-command local demo: API + Vite. Ctrl+C stops both.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-8765}"
FRONTEND_PORT="${FRONTEND_PORT:-5175}"
API_HOST="${API_HOST:-0.0.0.0}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export API_PORT API_HOST
export VITE_PROXY_TARGET="http://127.0.0.1:${API_PORT}"
export VITE_API_URL="${VITE_API_URL:-}"

echo "Starting VolatilityLab"
echo "  API:      http://127.0.0.1:${API_PORT}"
echo "  Frontend: http://127.0.0.1:${FRONTEND_PORT}"

cleanup() {
  if [[ -n "${BACK_PID:-}" ]]; then kill "${BACK_PID}" 2>/dev/null || true; fi
  if [[ -n "${FRONT_PID:-}" ]]; then kill "${FRONT_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m uvicorn backend.app.main:app --host "${API_HOST}" --port "${API_PORT}" &
BACK_PID=$!

(
  cd "${ROOT}/frontend"
  npm run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT}"
) &
FRONT_PID=$!

# Wait until the API answers so a fresh clone can be checked with curl.
for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    echo "API healthy on :${API_PORT}"
    break
  fi
  sleep 0.25
done

echo "Open http://127.0.0.1:${FRONTEND_PORT} — Ctrl+C to stop."
wait
