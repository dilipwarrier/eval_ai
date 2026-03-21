#!/bin/bash

# Configuration
# https://www.gutenberg.org/cache/epub/11450/pg11450.txt: Fort comme la mort by Guy de Maupassant (French)
# https://www.gutenberg.org/files/1661/1661-0.txt: The Adventures of Sherlock Holmes by Arthur Conan Doyle
SOURCE_URL="https://www.gutenberg.org/cache/epub/11450/pg11450.txt"
RAW_FILE="source_raw.txt"
CLEAN_FILE="source_clean.txt"

# Parameter: Max word length per segment
MAX_WORDS="${1:-1000}"  # Defaults to 1000 if no argument provided

echo "Downloading from: $SOURCE_URL"

# -f (fail silently on server errors) -L (follow redirects)
if curl -f -L "$SOURCE_URL" -o "$RAW_FILE"; then
    echo "Download successful. Cleaning headers and footers..."

    # 1. Extract everything between the Gutenberg markers
    # 2. Delete the first and last lines (the markers themselves)
    sed -n '/\*\*\* START OF THE PROJECT GUTENBERG EBOOK/,/\*\*\* END OF THE PROJECT GUTENBERG EBOOK/p' "$RAW_FILE" | sed '1d;$d' > "$CLEAN_FILE"

    # Verify the results
    TOTAL_WORDS=$(wc -w < "$CLEAN_FILE")
    echo "------------------------------------------"
    echo "Clean file saved as: $CLEAN_FILE"
    echo "Total word count: $TOTAL_WORDS"
    echo "Target words per segment: $MAX_WORDS"
    echo "------------------------------------------"

    # Splitting logic using awk
    echo "Splitting into segments..."
    awk -v max="$MAX_WORDS" '
        BEGIN { file_idx = 0; word_count = 0; filename = sprintf("segment_%02d.txt", file_idx) }
        {
            for (i = 1; i <= NF; i++) {
                printf "%s%s", $i, (i == NF ? ORS : OFS) >> filename
                word_count++
                if (word_count >= max) {
                    close(filename)
                    file_idx++
                    filename = sprintf("segment_%02d.txt", file_idx)
                    word_count = 0
                }
            }
        }
    ' "$CLEAN_FILE"

    echo "Successfully created $(ls segment_*.txt | wc -l) segments."
    echo "------------------------------------------"

    rm "$RAW_FILE"
else
    echo "Error: Failed to download the file. Please check the URL or your connection."
    exit 1
fi
