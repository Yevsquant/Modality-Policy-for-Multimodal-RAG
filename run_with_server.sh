#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
SERVER_CMD="./start_server.sh"
CLIENT_CMD="python run_mmdocrag_baseline.py"
HEALTH_URL="http://127.0.0.1:8000/v1/models"
WAIT_SECONDS=600
POLL_INTERVAL=5
LOG_DIR="logs"
SERVER_LOG="${LOG_DIR}/server.log"
CLIENT_LOG="${LOG_DIR}/client.log"
# ------------------------

mkdir -p "${LOG_DIR}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "Stopping server PID ${SERVER_PID} ..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting server..."
bash -lc "${SERVER_CMD}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

echo "Waiting for server readiness at ${HEALTH_URL} ..."
elapsed=0
until curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server exited early. Last server log lines:"
        tail -n 50 "${SERVER_LOG}" || true
        exit 1
    fi

    if (( elapsed >= WAIT_SECONDS )); then
        echo "Timed out waiting for server."
        tail -n 50 "${SERVER_LOG}" || true
        exit 1
    fi

    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
done

echo "Server is ready."
echo "Running client..."
bash -lc "${CLIENT_CMD}" | tee "${CLIENT_LOG}"

echo "Client finished successfully."