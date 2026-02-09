#!/bin/bash

################################################################################
# Script Name:  lyptus_run.sh
# Description:  Automates the workflow for remote AI/ML development.
#               1. Verifies the 'lyptus' SSH alias exists.
#               2. Syncs local code to the Lyptus server via rsync.
#               3. Executes a specified command on the remote server.
#
# Usage:        ./lyptus_run.sh [local_dir] [remote_dir] "<command>"
# Defaults:     If only one arg is provided, it is treated as the command.
#               local_dir defaults to "."
#               remote_dir defaults to local_dir.
################################################################################

# Configuration
REMOTE_ALIAS="lyptus"

# 1. Check if SSH alias exists
if ! ssh -G "$REMOTE_ALIAS" > /dev/null 2>&1; then
    echo "ERROR: SSH alias '$REMOTE_ALIAS' not found."
    echo "Please add it to your ~/.ssh/config. Example:"
    echo "Host $REMOTE_ALIAS"
    echo "    HostName <remote-ip-address>"
    echo "    User user3"
    echo "    IdentityFile ~/.ssh/id_rsa"
    exit 1
fi

# 2. Handle Argument Logic
if [ $# -eq 1 ]; then
    # Only command provided
    LOCAL_DIR="."
    REMOTE_DIR="~/projects/eval_ai/" # Your established project path
    REMOTE_COMMAND="$1"
elif [ $# -eq 3 ]; then
    # Full path override: local_dir, remote_dir, command
    LOCAL_DIR="$1"
    REMOTE_DIR="$2"
    REMOTE_COMMAND="$3"
else
    echo "Usage: $0 [local_dir] [remote_dir] \"<command>\""
    echo "Example (standard): $0 \"python basic_vllm.py\""
    echo "Example (explicit): $0 ./ ~/projects/custom/ \"python main.py\""
    exit 1
fi

# 3. Sync and Execute
echo "--- Syncing files: $LOCAL_DIR -> $REMOTE_ALIAS:$REMOTE_DIR ---"

rsync --archive --verbose --compress \
      --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '*~' \
      "$LOCAL_DIR" "$REMOTE_ALIAS:$REMOTE_DIR" && {

    echo "--- Sync Successful. Executing on Lyptus: $REMOTE_COMMAND ---"

    # Execute the provided command remotely
    ssh "$REMOTE_ALIAS" "cd $REMOTE_DIR && [ -d .venv ] && source .venv/bin/activate; $REMOTE_COMMAND"
} || {
    echo "CRITICAL: Sync failed. Remote execution aborted."
    exit 1
}
