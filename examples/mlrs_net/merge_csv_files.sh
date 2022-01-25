#!/bin/sh
set -eu

CSV_DIR=$1
OUTPUT_FILE=$2

cat "$CSV_DIR"/*.csv | head -n 1 > "$OUTPUT_FILE"
find "$CSV_DIR" -name '*.csv' -print0 | xargs -0 -n 1 tail -n +2 >> "$OUTPUT_FILE"
