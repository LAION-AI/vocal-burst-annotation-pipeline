#!/usr/bin/env bash
# Reassemble the split model weights (each file is split into <95 MB parts to fit GitHub's
# 100 MB/file limit). Run from the repo root or from weights/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

assemble () {   # $1 = dir  $2 = output filename  $3 = sha256 file
  echo ">>> assembling $1/$2"
  cat "$1/$2".*.part > "$1/$2"
  if [ -f "$1/$3" ]; then
    have=$(sha256sum "$1/$2" | awk '{print $1}')
    want=$(cat "$1/$3")
    [ "$have" = "$want" ] && echo "    sha256 OK" || { echo "    sha256 MISMATCH ($have != $want)"; exit 1; }
  fi
}

assemble locator   model.pt          model.pt.sha256
assemble captioner model.safetensors model.safetensors.sha256

cat <<EOF

Done. Weights assembled:
  $HERE/locator/model.pt
  $HERE/captioner/model.safetensors
Use them offline with:   export VB_WEIGHTS_DIR="$HERE"
(otherwise the code auto-downloads the same weights from the HuggingFace hub.)
EOF
