#!/bin/sh
set -eu

IMAGE_DIR=$1

rar_files=$(find "$IMAGE_DIR" -name '*.rar')
for rar_file in $rar_files
do
    unrar e "$rar_file" "$IMAGE_DIR"
done
