#!/bin/bash
# entrypoint.sh

# Make sure artifacts directory exists
mkdir -p /app/artifacts

# Run the specified command with arguments
exec "$@"