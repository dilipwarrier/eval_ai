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

# Ensure the virtual environment is deactivated on script exit
trap "deactivate" EXIT

# Default number of iterations for tasks
NUM_ITERATIONS=1

# Default processor type
PROCESSOR_TYPE="Nvidia"

# Function to display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -n, --num_iter NUM       Number of iterations for tasks (default: 1)"
    echo "  -p, --processor_type TYPE Processor type: Intel or Nvidia (default: Nvidia)"
    echo "  -h, --help               Display this help message"
    exit 0
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--num_iter) NUM_ITERATIONS="$2"; shift ;;
        -p|--processor_type) PROCESSOR_TYPE="$(echo "$2" | tr '[:upper:]' '[:lower:]' | sed 's/.*/\u&/')" ; shift ;;
        -h|--help) show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help ;;
    esac
    shift
done
MODEL_NAME="unsloth/llama-3-8b-instruct-bnb-4bit"

# Directory Setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="autogen_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# File Paths (Stored inside the results directory)
top_level_stdout_log="${RESULTS_DIR}/top_level_stdout.log"
top_level_stderr_log="${RESULTS_DIR}/top_level_stderr.log"
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
echo "Saving machine configuration..." | tee -a "$top_level_stdout_log"
{
    echo "=== System Timestamp: $(date) ==="
    echo ""
    echo "--- CPU Information ---"
    lscpu | grep -E "Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core" | column -t -s ":"

    echo ""
    echo "--- CPU RAM Information ---"
    free -h | grep -E "Mem:|Total" | column -t

    echo ""
    echo "--- GPU & VRAM Information ---"
    if [ "$PROCESSOR_TYPE" == "Nvidia" ]; then
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | column -t -s ","
        else
            echo "Nvidia GPU driver not detected."
        fi
    elif [ "$PROCESSOR_TYPE" == "Intel" ]; then
        if command -v clinfo &> /dev/null; then
            clinfo --raw | grep -E "Device Name|Global Memory size" | column -t -s ":"
        else
            echo "Intel XPU driver not detected."
        fi
    fi
} >> "$top_level_stdout_log" 2>> "$top_level_stderr_log"

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
# SETUP VIRTUAL ENVIRONMENT
# ==========================================
if [ "$PROCESSOR_TYPE" == "Nvidia" ]; then
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment for Nvidia GPU..." | tee -a "$top_level_stdout_log"
        python3 -m venv .venv
    fi

    echo "Setting up virtual environment for Nvidia GPU..." | tee -a "$top_level_stdout_log"
    source .venv/bin/activate >> "$top_level_stdout_log" 2>> "$top_level_stderr_log"
    pip install -r requirements.txt >> "$top_level_stdout_log" 2>> "$top_level_stderr_log"

elif [ "$PROCESSOR_TYPE" == "Intel" ]; then
    echo "Setting up environment for Intel XPU..." | tee -a "$top_level_stdout_log"
    source /home/lyptusadmin/vllm-xpu-env/bin/activate
fi

# ==========================================
# TASK 0: QUICK TEST (PRE-LOAD MODEL)
# ==========================================
echo -e "\nExecuting Task 0: Quick Test (Model: $MODEL_NAME)"
python split_vllm.py \
    --model "$MODEL_NAME" \
    --prompt "Write a 100-word essay on the universe" \
    --enforce-eager \
    > "$quick_test_stdout_log" 2> "$quick_test_stderr_log"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "CRITICAL ERROR: Quick Test failed (Code $exit_code). Aborting tasks."
    exit $exit_code
fi
echo "-> Quick Test successful!"

# Fetch and extract data once before the loop
fetch_and_extract "$FRENCH_URL" "$translation_input" "$num_words_for_translation"
fetch_and_extract "$ENGLISH_URL" "$summarization_input" "$num_words_for_summarization"

# Arrays to store file paths for each run
translation_stdout_runs=()
summarization_stdout_runs=()
generation_stdout_runs=()

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    echo -e "\nIteration $i of $NUM_ITERATIONS"

    # Prepare file paths for this iteration
    translation_input_run="${RESULTS_DIR}/translation_input_run${i}.txt"
    translation_output_run="${RESULTS_DIR}/translation_output_run${i}.txt"
    translation_stdout_run="${RESULTS_DIR}/translation_stdout_run${i}.log"
    translation_stderr_run="${RESULTS_DIR}/translation_stderr_run${i}.log"

    summarization_input_run="${RESULTS_DIR}/summarization_input_run${i}.txt"
    summarization_output_run="${RESULTS_DIR}/summarization_output_run${i}.txt"
    summarization_stdout_run="${RESULTS_DIR}/summarization_stdout_run${i}.log"
    summarization_stderr_run="${RESULTS_DIR}/summarization_stderr_run${i}.log"

    generation_output_run="${RESULTS_DIR}/generation_output_run${i}.txt"
    generation_stdout_run="${RESULTS_DIR}/generation_stdout_run${i}.log"
    generation_stderr_run="${RESULTS_DIR}/generation_stderr_run${i}.log"

    # Store stdout file paths in arrays
    translation_stdout_runs+=("$translation_stdout_run")
    summarization_stdout_runs+=("$summarization_stdout_run")
    generation_stdout_runs+=("$generation_stdout_run")

    # Copy the input files for this iteration
    cp "$translation_input" "$translation_input_run"
    cp "$summarization_input" "$summarization_input_run"

    # ==========================================
    # TASK 1: TRANSLATION
    # ==========================================
    echo -e "\nExecuting Task 1: Translation"

    task1_start=$SECONDS
    python split_vllm.py \
        --model "$MODEL_NAME" \
        --prompt "Act as a professional French-to-English translator. Translate the following text: " \
        --prompt-file "$translation_input_run" \
        --save-output "$translation_output_run" \
        --enforce-eager \
        > "$translation_stdout_run" 2> "$translation_stderr_run"

    exit_code=$?
    task1_end=$SECONDS
    [ $exit_code -ne 0 ] && echo "ERROR: Task 1 failed (Code $exit_code)." || echo "-> Task 1 Complete."
    echo "-> Inference time: $((task1_end - task1_start)) seconds"

    # ==========================================
    # TASK 2: SUMMARIZATION
    # ==========================================
    echo -e "\nExecuting Task 2: Summarization"

    task2_start=$SECONDS
    python split_vllm.py \
        --model "$MODEL_NAME" \
        --prompt "Summarize the following English text: " \
        --prompt-file "$summarization_input_run" \
        --save-output "$summarization_output_run" \
        --enforce-eager \
        > "$summarization_stdout_run" 2> "$summarization_stderr_run"

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
        --save-output "$generation_output_run" \
        --enforce-eager \
        > "$generation_stdout_run" 2> "$generation_stderr_run"

    exit_code=$?
    task3_end=$SECONDS
    [ $exit_code -ne 0 ] && echo "ERROR: Task 3 failed (Code $exit_code)." || echo "-> Task 3 Complete."
    echo "-> Inference time: $((task3_end - task3_start)) seconds"
done

# ==========================================
# GENERATE REPORT
# ==========================================
echo -e "\nGenerating report..."

report_file="${RESULTS_DIR}/top_level_stats.txt"
echo -e "Task\tRun\tPhase\tTime (s)\tInput tokens\tOutput tokens" > "$report_file"

parse_and_append_stats() {
    local task_name="$1"
    local run_number="$2"
    local output_file="$3"

    # Extract relevant lines and append to report
    grep -E "Initialization|Prefill|Decode" "$output_file" | while read -r line; do
        phase=$(echo "$line" | awk '{print $1}')
        time=$(echo "$line" | awk '{print $3}')
        input_tokens=$(echo "$line" | awk '{print $5}')
        output_tokens=$(echo "$line" | awk '{print $7}')
        echo -e "${task_name}\t${run_number}\t${phase}\t${time}\t${input_tokens}\t${output_tokens}" >> "$report_file"
    done
}

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    parse_and_append_stats "Translation" "$i" "${translation_stdout_runs[$i-1]}"
    parse_and_append_stats "Summarization" "$i" "${summarization_stdout_runs[$i-1]}"
    parse_and_append_stats "Generation" "$i" "${generation_stdout_runs[$i-1]}"
done

# Sort the report by Task, Run, and Phase, keeping the header at the top
{
    head -n 1 "$report_file"
    tail -n +2 "$report_file" | sort -k1,1 -k2,2n -k3,3
} > "${report_file}.tmp" && mv "${report_file}.tmp" "$report_file"

echo "Report generated: $report_file"
echo "All tasks finished. Results saved in: $RESULTS_DIR"
echo "=========================================="
