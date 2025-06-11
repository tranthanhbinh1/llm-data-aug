#!/bin/bash

# Set up Dagster environment
export DAGSTER_HOME=$(pwd)/.dagster_home

# Ensure .dagster_home directory exists
mkdir -p .dagster_home

echo "Starting Dagster development server..."
echo "DAGSTER_HOME: $DAGSTER_HOME"
echo "Visit http://localhost:3001 to view the Dagster UI"
echo ""

# Start Dagster dev server
dagster dev --port 3001 -m src.worker -a defs 
