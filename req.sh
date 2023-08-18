#!/bin/bash

# Check if the directory is provided
if [ -z "$1" ]; then
  echo "Please provide the directory to scan for Python files."
  exit 1
fi

# Check if pipreqs is installed
if ! command -v pipreqs &> /dev/null; then
  echo "pipreqs is not installed. Please install it using 'pip install pipreqs' and rerun the script."
  exit 1
fi

# Directory to scan
DIRECTORY="$1"

# Generate the requirements.txt file
pipreqs "$DIRECTORY" --force

echo "requirements.txt has been generated successfully in $DIRECTORY"
