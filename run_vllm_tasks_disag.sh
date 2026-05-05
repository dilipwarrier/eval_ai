#!/bin/bash
# DESCRIPTION:
# This script automates LLM benchmarking tasks:
# 1. Translation: Translates a segment of a French book.
# 2. Summarization: Summarizes a segment of an English book.
# 3. Generation: Generates an essay on the Enlightenment.
# Each task runs NUM_INNER_RUNS times per word-count, across NUM_ITERATIONS word-count levels.
# Order: Translation x3 @ 500w -> Summarization x3 @ 500w -> Generation x3 @ 500w -> repeat at 1000w -> ...

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================

trap "deactivate" EXIT

NUM_ITERATIONS=6
NUM_INNER_RUNS=3
PROCESSOR_TYPE="Intel"

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -n, --num_iter NUM        Number of word-count iterations (default: 6)"
    echo "  -r, --inner_runs NUM      Number of runs per task per word-count (default: 3)"
    echo "  -p, --processor_type TYPE Processor type: Intel or Nvidia (default: Intel)"
    echo "  -h, --help                Display this help message"
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--num_iter) NUM_ITERATIONS="$2"; shift ;;
        -r|--inner_runs) NUM_INNER_RUNS="$2"; shift ;;
        -p|--processor_type) PROCESSOR_TYPE="$(echo "$2" | tr '[:upper:]' '[:lower:]' | sed 's/.*/\u&/')" ; shift ;;
        -h|--help) show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help ;;
    esac
    shift
done

MODEL_NAME="unsloth/llama-3-8b-instruct-bnb-4bit"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="autogen_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

top_level_stdout_log="${RESULTS_DIR}/top_level_stdout.log"
top_level_stderr_log="${RESULTS_DIR}/top_level_stderr.log"

translation_input="${RESULTS_DIR}/translation_input.txt"
summarization_input="${RESULTS_DIR}/summarization_input.txt"

base_words=500

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
    lscpu | grep -E "Model name" | column -t -s ":"
    echo ""
    echo "--- CPU RAM Information ---"
    free -h | grep -E -i "mem:|total" | column -t
    if [ "$PROCESSOR_TYPE" == "Nvidia" ]; then
        echo ""
        echo "--- NVIDIA GPU & VRAM Information ---"
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | column -t -s ","
        else
            echo "Nvidia GPU driver not detected."
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

# Arrays to store stdout paths for report generation
translation_stdout_runs=()
summarization_stdout_runs=()
generation_stdout_runs=()

# ==========================================
# MAIN LOOPS
# Order: Translation x3 -> Summarization x3 -> Generation x3, then +500 words, repeat
# ==========================================
for ((i=1; i<=NUM_ITERATIONS; i++)); do
    w=$((base_words * i))

    echo -e "\n=========================================="
    echo "=== Word-count level $i of $NUM_ITERATIONS: $w words ==="
    echo "=========================================="

    # Fetch input files once per word-count level
    echo "Fetching French input ($w words)..."
    fetch_and_extract "$FRENCH_URL" "$translation_input" "$w"
    echo "Fetching English input ($w words)..."
    fetch_and_extract "$ENGLISH_URL" "$summarization_input" "$w"

    # ==========================================
    # TASK 1: TRANSLATION x NUM_INNER_RUNS
    # ==========================================
    echo -e "\n--- Task 1: Translation @ ${w} words ---"
    for ((r=1; r<=NUM_INNER_RUNS; r++)); do
        echo -e "\n  Translation run $r of $NUM_INNER_RUNS"
        RUN_TAG="iter${i}_run${r}"

        translation_input_run="${RESULTS_DIR}/translation_input_${RUN_TAG}.txt"
        translation_output_run="${RESULTS_DIR}/translation_output_${RUN_TAG}.txt"
        translation_stdout_run="${RESULTS_DIR}/translation_stdout_${RUN_TAG}.log"
        translation_stderr_run="${RESULTS_DIR}/translation_stderr_${RUN_TAG}.log"

        translation_stdout_runs+=("$translation_stdout_run")
        cp "$translation_input" "$translation_input_run"

        task_start=$SECONDS
        python dissag_split_vllm.py \
            --model "$MODEL_NAME" \
            --prompt "Act as a professional French-to-English translator. Translate the following text: " \
            --prompt-file "$translation_input_run" \
            --save-output "$translation_output_run" \
            --enforce-eager \
            --max-model-len 8192 \
            > "$translation_stdout_run" 2> "$translation_stderr_run"
        exit_code=$?
        task_end=$SECONDS
        [ $exit_code -ne 0 ] && echo "  ERROR: Translation run $r failed (Code $exit_code)." || echo "  -> Translation run $r complete."
        echo "  -> Inference time: $((task_end - task_start)) seconds"
    done

    # ==========================================
    # TASK 2: SUMMARIZATION x NUM_INNER_RUNS
    # ==========================================
    echo -e "\n--- Task 2: Summarization @ ${w} words ---"
    for ((r=1; r<=NUM_INNER_RUNS; r++)); do
        echo -e "\n  Summarization run $r of $NUM_INNER_RUNS"
        RUN_TAG="iter${i}_run${r}"

        summarization_input_run="${RESULTS_DIR}/summarization_input_${RUN_TAG}.txt"
        summarization_output_run="${RESULTS_DIR}/summarization_output_${RUN_TAG}.txt"
        summarization_stdout_run="${RESULTS_DIR}/summarization_stdout_${RUN_TAG}.log"
        summarization_stderr_run="${RESULTS_DIR}/summarization_stderr_${RUN_TAG}.log"

        summarization_stdout_runs+=("$summarization_stdout_run")
        cp "$summarization_input" "$summarization_input_run"

        task_start=$SECONDS
        python dissag_split_vllm.py \
            --model "$MODEL_NAME" \
            --prompt "Summarize the following English text in exactly 300 words: " \
            --prompt-file "$summarization_input_run" \
            --save-output "$summarization_output_run" \
            --enforce-eager \
            --max-model-len 8192 \
            > "$summarization_stdout_run" 2> "$summarization_stderr_run"
        exit_code=$?
        task_end=$SECONDS
        [ $exit_code -ne 0 ] && echo "  ERROR: Summarization run $r failed (Code $exit_code)." || echo "  -> Summarization run $r complete."
        echo "  -> Inference time: $((task_end - task_start)) seconds"
    done

    # ==========================================
    # TASK 3: GENERATION x NUM_INNER_RUNS
    # ==========================================
    echo -e "\n--- Task 3: Generation @ ${w} words ---"
    for ((r=1; r<=NUM_INNER_RUNS; r++)); do
        echo -e "\n  Generation run $r of $NUM_INNER_RUNS"
        RUN_TAG="iter${i}_run${r}"

        generation_output_run="${RESULTS_DIR}/generation_output_${RUN_TAG}.txt"
        generation_stdout_run="${RESULTS_DIR}/generation_stdout_${RUN_TAG}.log"
        generation_stderr_run="${RESULTS_DIR}/generation_stderr_${RUN_TAG}.log"

        generation_stdout_runs+=("$generation_stdout_run")

        GENERATION_PROMPT="System: You are a professional essay writer. User: Write a $w word essay on the Enlightenment movement."

        task_start=$SECONDS
        python dissag_split_vllm.py \
            --model "$MODEL_NAME" \
            --prompt "$GENERATION_PROMPT" \
            --save-output "$generation_output_run" \
            --enforce-eager \
            --max-model-len 8192 \
            > "$generation_stdout_run" 2> "$generation_stderr_run"
        exit_code=$?
        task_end=$SECONDS
        [ $exit_code -ne 0 ] && echo "  ERROR: Generation run $r failed (Code $exit_code)." || echo "  -> Generation run $r complete."
        echo "  -> Inference time: $((task_end - task_start)) seconds"
    done

done  # end word-count iteration loop

# ==========================================
# GENERATE REPORT
# ==========================================
echo -e "\nGenerating report..."

report_file="${RESULTS_DIR}/top_level_stats.txt"
echo -e "Task\tIteration\tWords\tInnerRun\tPhase\tTime (s)\tInput tokens\tOutput tokens" > "$report_file"

parse_and_append_stats() {
    local task_name="$1"
    local iter_num="$2"
    local word_count="$3"
    local run_num="$4"
    local output_file="$5"

    grep -E "Initialization|Prefill|Decode" "$output_file" | while read -r line; do
        phase=$(echo "$line" | awk '{print $1}')
        time=$(echo "$line" | awk '{print $3}')
        input_tokens=$(echo "$line" | awk '{print $5}')
        output_tokens=$(echo "$line" | awk '{print $7}')
        echo -e "${task_name}\t${iter_num}\t${word_count}\t${run_num}\t${phase}\t${time}\t${input_tokens}\t${output_tokens}" >> "$report_file"
    done
}

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    w=$((base_words * i))
    for ((r=1; r<=NUM_INNER_RUNS; r++)); do
        run_index=$(( (i-1) * NUM_INNER_RUNS + (r-1) ))
        parse_and_append_stats "Translation"   "$i" "$w" "$r" "${translation_stdout_runs[$run_index]}"
        parse_and_append_stats "Summarization" "$i" "$w" "$r" "${summarization_stdout_runs[$run_index]}"
        parse_and_append_stats "Generation"    "$i" "$w" "$r" "${generation_stdout_runs[$run_index]}"
    done
done

{
    head -n 1 "$report_file"
    tail -n +2 "$report_file" | sort -k1,1 -k2,2n -k3,3n -k4,4n
} > "${report_file}.tmp" && mv "${report_file}.tmp" "$report_file"

echo "Report generated: $report_file"
echo "All tasks finished. Results saved in: $RESULTS_DIR"
echo "=========================================="
