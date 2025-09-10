#!/bin/bash

set -e

# Sync the environment, installing the project editable and including dev dependencies
uv sync

# Set Git safe directory to avoid ownership issues
git config --global --add safe.directory "$PWD"

# Optional: If you need to reset GitHub host key for SSH (uncomment if necessary)
# ssh-keygen -f "/home/vscode/.ssh/known_hosts" -R "github.com"
