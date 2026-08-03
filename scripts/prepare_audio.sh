#!/usr/bin/env bash
set -euo pipefail

IN_DIR=${1:?Usage: prepare_audio.sh <raw_dir> <out_dir>}
OUT_DIR=${2:?Usage: prepare_audio.sh <raw_dir> <out_dir>}

mkdir -p "$OUT_DIR"

find "$IN_DIR" -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.m4a' \) -print0 |
while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    name="${filename%.*}"
    out="$OUT_DIR/${name}.16k.wav"

    echo "Converting $file -> $out"

    ffmpeg -hide_banner -loglevel error -y \
        -i "$file" \
        -ac 1 \
        -ar 16000 \
        -sample_fmt s16 \
        "$out"
done

echo "Done."
