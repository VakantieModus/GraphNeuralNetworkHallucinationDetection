#!/usr/bin/env bash
set -e  # exit on error

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "Virtual environment already exists."

source venv/bin/activate
fi
