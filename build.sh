#!/bin/bash
# Build ANE prefill demo — Apple Silicon only
# Usage: ./build.sh
# Then:  ./ane_prefill_27b <path-to-gguf> [seq_len]

set -e

if [ "$(uname -m)" != "arm64" ]; then
    echo "Error: Apple Silicon (arm64) required"
    exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"

# Check for libomp (required for parallel DeltaNet recurrence)
OMP_FLAGS=""
if [ -d "/opt/homebrew/opt/libomp" ]; then
    OMP_FLAGS="-I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
elif [ -d "/usr/local/opt/libomp" ]; then
    OMP_FLAGS="-I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib -lomp"
else
    echo "Warning: libomp not found. Install with: brew install libomp"
    echo "Continuing without OpenMP (DeltaNet recurrence will be slower)..."
fi

echo "Building ANE prefill demo..."
xcrun clang -O2 -fobjc-arc \
    "$DIR/ane_prefill_27b.m" \
    "$DIR/ane_bridge.m" \
    -framework Foundation -framework IOSurface -framework Accelerate \
    $OMP_FLAGS \
    -o "$DIR/ane_prefill_27b"

echo "Built: $DIR/ane_prefill_27b"
echo ""
echo "Usage: $DIR/ane_prefill_27b <model.gguf> [seq_len=256]"
echo ""
echo "Get the model:"
echo "  pip install huggingface-hub"
echo "  huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF qwen3-30b-a3b-q4_k_m.gguf --local-dir ."
