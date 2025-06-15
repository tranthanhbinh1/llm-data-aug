#!/bin/bash

# Start Dagster development server for LLM Data Augmentation Pipeline
#
# This script starts the Dagster webserver in development mode, 
# loads our asset definitions, and sets up the environment.

# Get absolute path to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set absolute DAGSTER_HOME path
export DAGSTER_HOME="${PROJECT_ROOT}/.dagster_home"

# Ensure DAGSTER_HOME directory exists
mkdir -p "$DAGSTER_HOME"

# Start Dagster development server
echo "Starting Dagster development server..."
echo "DAGSTER_HOME: $DAGSTER_HOME"
echo "Visit: http://localhost:3001"

cd "$PROJECT_ROOT"
dagster dev -m src.worker -a defs --port 3001 
