#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${APP_DIR}/.." && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
API_BASE_URL="${VITE_API_BASE_URL:-http://${BACKEND_HOST}:${BACKEND_PORT}}"

backend_pid=""
frontend_pid=""

cleanup() {
  echo
  echo "Stopping app..."
  if [[ -n "${frontend_pid}" ]] && kill -0 "${frontend_pid}" 2>/dev/null; then
    kill "${frontend_pid}" 2>/dev/null || true
  fi
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    kill "${backend_pid}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn is not installed. Install backend dependencies first."
  echo "Example: pip install fastapi uvicorn xarray zarr pandas numpy"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is not installed."
  exit 1
fi

if [[ ! -d "${APP_DIR}/frontend/node_modules" ]]; then
  echo "data_viewer/frontend/node_modules is missing. Run: cd data_viewer/frontend && npm install"
  exit 1
fi

echo "Starting backend: http://${BACKEND_HOST}:${BACKEND_PORT}"
(
  cd "${ROOT_DIR}"
  uvicorn data_viewer.backend.app:app --reload --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
) &
backend_pid="$!"

echo "Starting frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
(
  cd "${APP_DIR}/frontend"
  VITE_API_BASE_URL="${API_BASE_URL}" npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}"
) &
frontend_pid="$!"

echo
echo "App is starting."
echo "Open: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "Backend API: ${API_BASE_URL}/api"
echo "Press Ctrl-C to stop both servers."
echo

wait -n "${backend_pid}" "${frontend_pid}"
