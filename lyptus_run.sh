#!/bin/bash
# lyptus_run.sh: Syncs code to remote, executes with timestamps, and pulls results back.

REMOTE_ALIAS="lyptus"
REMOTE_DIR="~/projects/eval_ai"
LOCAL_DIR="."

echo "--- Execution Started: $(date '+%Y-%m-%d %H:%M:%S') ---"

# 1. Push code to Lyptus
rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    "$LOCAL_DIR/" "$REMOTE_ALIAS:$REMOTE_DIR/"

# 2. Execute remote command
ssh "$REMOTE_ALIAS" "cd $REMOTE_DIR && . .venv/bin/activate && $1"

# 3. Pull output files back
echo "--- Pulling remote output files back to local ---"
rsync -avz --include='*.txt' --exclude='*' "$REMOTE_ALIAS:$REMOTE_DIR/" "$LOCAL_DIR/"

echo "--- Execution Finished: $(date '+%Y-%m-%d %H:%M:%S') ---"
