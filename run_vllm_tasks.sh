#!/bin/bash
# DESCRIPTION:
# This script automates three LLM benchmarking tasks:
# 1. Translation: Downloads a French book, cleans it, and translates the first segment using vLLM.
# 2. Summarization: Downloads an English book, cleans it, and summarizes the first segment using vLLM.
# 3. Generation: Prompts vLLM to generate a professional essay on the Enlightenment.
# It utilizes 'split_vllm.py' for phase profiling (Prefill/Decode) and performance tracking.
# All stdout and stderr for each task are captured in individual log files with the 'autogen_' prefix.
# Performance timings for each sub-task are printed to the console.

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================

# Model Configuration
MODEL_NAME="unsloth/llama-3-8b-instruct-bnb-4bit"

# Task 1: Translation Settings
num_words_for_translation=1000
translation_task_input_filename="autogen_translation_input.txt"
translation_task_output_filename="autogen_translation_output.txt"
translation_stdout_log="autogen_translation_stdout.log"
translation_stderr_log="autogen_translation_stderr.log"

# Task 2: Summarization Settings
num_words_for_summarization=1000
summarization_task_input_filename="autogen_summarization_input.txt"
summarization_task_output_filename="autogen_summarization_output.txt"
summarization_stdout_log="autogen_summarization_stdout.log"
summarization_stderr_log="autogen_summarization_stderr.log"

# Task 3: Generation Settings
num_words_for_generation=500
generation_task_output_filename="autogen_generation_output.txt"
generation_stdout_log="autogen_generation_stdout.log"
generation_stderr_log="autogen_generation_stderr.log"

# URLs
FRENCH_URL="https://www.gutenberg.org/cache/epub/11450/pg11450.txt"
ENGLISH_URL="https://www.gutenberg.org/files/1661/1661-0.txt"

# ==========================================
# HELPER FUNCTION: Fetch & Extract Words
# ==========================================
fetch_and_extract() {
    local url="$1"
    local output_file="$2"
    local max_words="$3"
    local raw_file="autogen_temp_raw_$$.txt"

    local start_time=$SECONDS

    echo "Downloading from: $url"
    if curl -s -f -L "$url" -o "$raw_file"; then
        echo "Download successful. Cleaning headers and extracting first $max_words words..."

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
        echo "-> Saved exactly $(wc -w < "$output_file") words to $output_file"
        echo "-> Time taken for fetch and extract: $((end_time - start_time)) seconds"
    else
        echo "Error: Failed to download from $url"
        exit 1
    fi
}

# ==========================================
# TASK 1: TRANSLATION
# ==========================================
echo ""
echo "=========================================="
echo "Executing Task 1: Translation (Model: $MODEL_NAME)"
echo "=========================================="
fetch_and_extract "$FRENCH_URL" "$translation_task_input_filename" "$num_words_for_translation"

task1_start=$SECONDS
python split_vllm.py \
    --model "$MODEL_NAME" \
    --prompt "Act as a professional French-to-English translator. Translate the following text into natural, fluent English. Maintain the original tone and do not add conversational filler. French Text:" \
    --prompt-file "$translation_task_input_filename" \
    --save-output "$translation_task_output_filename" \
    > "$translation_stdout_log" 2> "$translation_stderr_log"
task1_end=$SECONDS

echo "-> Task 1 Complete! Output saved to: $translation_task_output_filename"
echo "-> Time taken for translation inference: $((task1_end - task1_start)) seconds"
echo "-> Logs: $translation_stdout_log, $translation_stderr_log"

# ==========================================
# TASK 2: SUMMARIZATION
# ==========================================
echo ""
echo "=========================================="
echo "Executing Task 2: Summarization (Model: $MODEL_NAME)"
echo "=========================================="
fetch_and_extract "$ENGLISH_URL" "$summarization_task_input_filename" "$num_words_for_summarization"

task2_start=$SECONDS
python split_vllm.py \
    --model "$MODEL_NAME" \
    --prompt "Summarize the following English text into a concise summary. Capture the main themes, plot points, and tone of the narrative. English Text:" \
    --prompt-file "$summarization_task_input_filename" \
    --save-output "$summarization_task_output_filename" \
    > "$summarization_stdout_log" 2> "$summarization_stderr_log"
task2_end=$SECONDS

echo "-> Task 2 Complete! Output saved to: $summarization_task_output_filename"
echo "-> Time taken for summarization inference: $((task2_end - task2_start)) seconds"
echo "-> Logs: $summarization_stdout_log, $summarization_stderr_log"

# ==========================================
# TASK 3: GENERATION
# ==========================================
echo ""
echo "=========================================="
echo "Executing Task 3: Generation (Model: $MODEL_NAME)"
echo "=========================================="

GENERATION_PROMPT="System: You are a professional essay writer. User: Write a $num_words_for_generation word essay on the Enlightenment movement."

echo "Sending generation prompt to vLLM..."
task3_start=$SECONDS
python split_vllm.py \
    --model "$MODEL_NAME" \
    --prompt "$GENERATION_PROMPT" \
    --save-output "$generation_task_output_filename" \
    > "$generation_stdout_log" 2> "$generation_stderr_log"
task3_end=$SECONDS

echo "-> Task 3 Complete! Output saved to: $generation_task_output_filename"
echo "-> Time taken for generation: $((task3_end - task3_start)) seconds"
echo "-> Logs: $generation_stdout_log, $generation_stderr_log"
echo "All tasks finished successfully."
echo "=========================================="
