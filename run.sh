#!/bin/bash

# Run anti-cheat service
cd "$(dirname "$0")"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8081

