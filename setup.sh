#!/bin/bash

# Setup pip environment for development.

python -m pip install --upgrade pip
pip install -r build_requirements.txt
pip install -r run_requirements.txt
pip install -r test_requirements.txt
