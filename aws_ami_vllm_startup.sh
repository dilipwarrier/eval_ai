#!/bin/bash
# This is a startup script that can execute on an AWS Nvidia AMI
# to set up the virtual environment for vLLM.
echo "--- Starting Startup Script: \$(date) ---"

sudo apt-get install emacs

# 2. Clone repository into the user space
TARGET_DIR="/home/ubuntu/eval_ai"
if [ ! -d $TARGET_DIR ]; then
    git clone https://github.com/dilipwarrier/eval_ai $TARGET_DIR
fi

# 3. Setup Virtual Environment and Dependencies
cd $TARGET_DIR
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo "--- Startup Script Complete: \$(date) ---"
