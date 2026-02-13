#!/bin/bash
# remote_sync_and_run.sh: Syncs code to remote, executes command, and pulls results.
# Usage: ./remote_sync_and_run.sh <remote_alias> <command>

# 1. Validate Arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <remote_alias> <command>"
    echo "Example: $0 lyptus \"promptfoo eval -c split_vllm_config.yaml\""
    exit 1
fi

REMOTE_ALIAS="$1"
shift
REMOTE_CMD="$*" # Captures the rest of the arguments as the command

REMOTE_DIR="~/projects/eval_ai"
LOCAL_DIR="."

echo "--- Syncing files to $REMOTE_ALIAS: $(date '+%Y-%m-%d +%H:%M:%S') ---"

# 2. Push code to remote location
rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
      --exclude '*~' \
      "$LOCAL_DIR/" "$REMOTE_ALIAS:$REMOTE_DIR/"

# 3. Execute remote command
echo "--- Starting execution: $(date '+%Y-%m-%d %H:%M:%S') ---"
# We execute the passed command after activating the environment
ssh "$REMOTE_ALIAS" "cd $REMOTE_DIR && source .venv/bin/activate && $REMOTE_CMD"

# 4. Pull output files back
echo "--- Pulling results: $(date '+%Y-%m-%d +%H:%M:%S') ---"
rsync -avz --include='autogen_*.txt' --exclude='*' "$REMOTE_ALIAS:$REMOTE_DIR/" "$LOCAL_DIR/"

echo "--- Completing: $(date '+%Y-%m-%d %H:%M:%S') ---"
