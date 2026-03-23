#!/bin/bash
# DESCRIPTION:
# This script automates LLM benchmarking tasks:
# 0. Quick Test: Pre-loads the model and verifies the environment.
# 1. Translation: Translates a segment of a French book.
# 2. Summarization: Summarizes a segment of an English book.
# 3. Generation: Generates an essay on the Enlightenment.
# It saves all outputs, logs, and machine hardware configuration to a timestamped sub-directory.

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================

# Model Configuration
MODEL_NAME="unsloth/llama-3-8b-instruct-bnb-4bit"

# Directory Setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="autogen_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# File Paths (Stored inside the results directory)
machine_config_log="${RESULTS_DIR}/machine_config.log"
quick_test_stdout_log="${RESULTS_DIR}/quick_test_stdout.log"
quick_test_stderr_log="${RESULTS_DIR}/quick_test_stderr.log"

translation_input="${RESULTS_DIR}/translation_input.txt"
translation_output="${RESULTS_DIR}/translation_output.txt"
translation_stdout="${RESULTS_DIR}/translation_stdout.log"
translation_stderr="${RESULTS_DIR}/translation_stderr.log"

summarization_input="${RESULTS_DIR}/summarization_input.txt"
summarization_output="${RESULTS_DIR}/summarization_output.txt"
summarization_stdout="${RESULTS_DIR}/summarization_stdout.log"
summarization_stderr="${RESULTS_DIR}/summarization_stderr.log"

generation_output="${RESULTS_DIR}/generation_output.txt"
generation_stdout="${RESULTS_DIR}/generation_stdout.log"
generation_stderr="${RESULTS_DIR}/generation_stderr.log"

# Task Settings
num_words_for_translation=1000
num_words_for_summarization=1000
num_words_for_generation=500

# URLs
FRENCH_URL="https://www.gutenberg.org/cache/epub/11450/pg11450.txt"
ENGLISH_URL="https://www.gutenberg.org/files/1661/1661-0.txt"

# ==========================================
# STEP: SAVE MACHINE CONFIGURATION
# ==========================================
echo "Saving machine configuration to $machine_config_log..."
{
    echo "=== System Timestamp: $(date) ==="
    echo ""
    echo "--- CPU Information ---"
    lscpu | grep -E "Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core"

    echo ""
    echo "--- CPU RAM Information ---"
    free -h | grep -E "Mem:|Total"

    echo ""
    echo "--- GPU & VRAM Information ---"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        echo "Nvidia GPU driver not detected."
    fi
} > "$machine_config_log"

# ==========================================
# HELPER FUNCTION: Fetch & Extract Words
# ==========================================
fetch_and_extract() {
    local url="$1"
    local output_file="$2"
    local max_words="$3"
    local raw_file="${RESULTS_DIR}/temp_raw_$$.txt"

    local start_time=$SECONDS
    echo "Downloading from: $url"
    if curl -s -f -L "$url" -o "$raw_file"; then
        echo "Cleaning headers and extracting first $max_words words..."
        sed -n '/\*\*\* START OF THE PROJECT GUTENBERG EBOOK/,/\*\*\* END OF THE PROJECT GUTENBERG EBOOK/p' "$raw_file" | sed '1d;$d' | \
        awk -v max="$max_words" '
            BEGIN { word_count = 0 }
            {
                for (i = 1; i <= NF; i++) {
                    printf "%s ", $i > "'"$output_file"'"
                    word_count++
                    if (word_count >= max) {
                        printf "\n" > "'"$output_file"'"
                        exit
                    }
                }
            }
        '
        rm "$raw_file"
        local end_time=$SECONDS
        echo "-> Time taken for fetch and extract: $((end_time - start_time)) seconds"
    else
        echo "Error: Failed to download from $url"
        exit 1
    fi
}

# ==========================================
# TASK 0: QUICK TEST (PRE-LOAD MODEL)
# ==========================================
echo -e "\nExecuting Task 0: Quick Test (Model: $MODEL_NAME)"
python split_vllm.py \
    --model "$MODEL_NAME" \
    --prompt "Write a 100-word essay on the universe" \
    > "$quick_test_stdout_log" 2> "$quick_test_stderr_log"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "CRITICAL ERROR: Quick Test failed (Code $exit_code). Aborting tasks."
    exit $exit_code
fi
echo "-> Quick Test successful!"

# ==========================================
# TASK 1: TRANSLATION
# ==========================================
echo -e "\nExecuting Task 1: Translation"
fetch_and_extract "$FRENCH_URL" "$translation_input" "$num_words_for_translation"

task1_start=$SECONDS
python split_vllm.py \
    --model "$MODEL_NAME" \
    --prompt "Act as a professional French-to-English translator. Translate the following text: " \
    --prompt-file "$translation_input" \
    --save-output "$translation_output" \
    > "$translation_stdout" 2> "$translation_stderr"

exit_code=$?
task1_end=$SECONDS
[ $exit_code -ne 0 ] && echo "ERROR: Task 1 failed (Code $exit_code)." || echo "-> Task 1 Complete."
echo "-> Inference time: $((task1_end - task1_start)) seconds"

# ==========================================
# TASK 2: SUMMARIZATION
# ==========================================
echo -e "\nExecuting Task 2: Summarization"
fetch_and_extract "$ENGLISH_URL" "$summarization_input" "$num_words_for_summarization"

task2_start=$SECONDS
python split_vllm.py \
    --model "$MODEL_NAME" \
    --prompt "Summarize the following English text: " \
    --prompt-file "$summarization_input" \
    --save-output "$summarization_output" \
    > "$summarization_stdout" 2> "$summarization_stderr"

exit_code=$?
task2_end=$SECONDS
[ $exit_code -ne 0 ] && echo "ERROR: Task 2 failed (Code $exit_code)." || echo "-> Task 2 Complete."
echo "-> Inference time: $((task2_end - task2_start)) seconds"

# ==========================================
# TASK 3: GENERATION
# ==========================================
echo -e "\nExecuting Task 3: Generation"
GENERATION_PROMPT="System: You are a professional essay writer. User: Write a $num_words_for_generation word essay on the Enlightenment movement."

task3_start=$SECONDS
python split_vllm.py \
    --model "$MODEL_NAME" \
    --prompt "$GENERATION_PROMPT" \
    --save-output "$generation_output" \
    > "$generation_stdout" 2> "$generation_stderr"

exit_code=$?
task3_end=$SECONDS
[ $exit_code -ne 0 ] && echo "ERROR: Task 3 failed (Code $exit_code)." || echo "-> Task 3 Complete."
echo "-> Inference time: $((task3_end - task3_start)) seconds"

echo -e "\n=========================================="
echo "All tasks finished. Results saved in: $RESULTS_DIR"
echo "=========================================="
