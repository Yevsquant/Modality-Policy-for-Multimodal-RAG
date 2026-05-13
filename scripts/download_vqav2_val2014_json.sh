#!/usr/bin/env bash
# Download official VQAv2 val2014 questions + annotations (JSON only, small).
# For images, download and unzip COCO val2014 into e.g. data/coco/val2014/ :
#   wget http://images.cocodataset.org/zips/val2014.zip
#   unzip val2014.zip -d data/coco
set -euo pipefail

ROOT="${1:-data/vqav2/official}"
mkdir -p "$ROOT"
cd "$ROOT"

BASE="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"
for f in \
  v2_OpenEnded_mscoco_val2014_questions.json \
  v2_mscoco_val2014_annotations.json
do
  if [[ -f "$f" ]]; then
    echo "exists: $f"
  else
    wget -q --show-progress "${BASE}/${f}" -O "$f"
  fi
done

echo "Done. JSON files are in: $(pwd)"
echo "Point the runner at:"
echo "  --questions-json $(pwd)/v2_OpenEnded_mscoco_val2014_questions.json \\"
echo "  --annotations-json $(pwd)/v2_mscoco_val2014_annotations.json \\"
echo "  --coco-images-dir <path-to-extracted>/val2014"
