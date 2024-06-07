#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config /home/model-server/config.properties
    echo "TorchServe started, waiting for it to stabilize..."
    sleep 10  # Waits for 10 seconds before proceeding
else
    eval "$@"
fi

# Start FastAPI app
uvicorn api.redoc:app --host 0.0.0.0 --port 5000 &

# Prevent Docker container from exiting
wait -n

# Exit with status of process that exited first
exit $?