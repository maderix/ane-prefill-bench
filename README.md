# ANE Prefill Benchmark — Qwen 3.5 27B on Apple Neural Engine

Standalone benchmark that runs the full Qwen 3.5 27B (64 layers, 48 DeltaNet + 16 Attention) prefill pipeline entirely on the Apple Neural Engine. No ML frameworks, no Python — just Objective-C talking directly to ANE private APIs.

**What it does:** Loads a GGUF model, dequantizes Q4K/Q5K/Q6K weights per-layer to FP16, and dispatches all projections + FFN through ANE conv1x1 kernels with pipelined DMA/compute staging. DeltaNet recurrence and causal attention run on CPU via Accelerate/BLAS.

## Results (M4, 10 cores, 24 GB)

| Seq Len | tok/s | TTFT |
|---------|-------|------|
| 256 | 20.2 | 12.7s |
| 512 | 24.3 | 21.1s |
| 1024 | 27.8 | 36.8s |

**Want to see what M4 Pro / Max / Ultra / M5 can do?** Run the benchmark and share your numbers.

## Quick Start

### 1. Install libomp

```bash
brew install libomp
```

### 2. Get the model (~16 GB)

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF qwen3-30b-a3b-q4_k_m.gguf --local-dir .
```

### 3. Build and run

```bash
./build.sh
./ane_prefill_27b qwen3-30b-a3b-q4_k_m.gguf
```

For longer sequences:

```bash
./ane_prefill_27b qwen3-30b-a3b-q4_k_m.gguf 512
./ane_prefill_27b qwen3-30b-a3b-q4_k_m.gguf 1024
```

## Sample Output

```
═══════════════════════════════════════════════════════════════
  ANE Prefill Benchmark — Qwen 3.5 27B
  Chip: Apple M4 (10 cores, 24 GB)
  S=256, CHUNK=256, 64 layers (48 DN + 16 Attn)
═══════════════════════════════════════════════════════════════

Compiling ANE kernels...
  12 kernels compiled in 50 ms

  Total: 12695 ms → 20.2 tok/s (S=256)
  ANE proj:     3282 ms  (208 dispatches)
  ANE FFN:      4390 ms
  CPU attn:     1397 ms
  CPU recur:    745 ms
```

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 14+ (Sonoma)
- Xcode Command Line Tools (`xcode-select --install`)
- libomp (`brew install libomp`)
- ~16 GB free RAM for the model

## Architecture

The pipeline runs per-layer:

```
RMSNorm (CPU) → Projections (ANE conv1x1) → Recurrence/Attention (CPU)
  → O_proj (ANE) → Residual (CPU) → RMSNorm (CPU) → Fused FFN (ANE) → Residual (CPU)
```

Key techniques:
- **Zero-copy output reads** — read ANE output IOSurface directly with NEON transpose, no intermediate buffer
- **Pipelined DMA/compute** — pthread stages next kernel's weights while current ANE eval runs (double-buffered kernels)
- **Adaptive parallel staging** — blocked dispatch_apply for large IOSurface copies (>4 MB)
- **K-tiled down projection** — [17408→5120] split into 2048-channel tiles with zero-copy NEON accumulation

ANE access is through `_ANEInMemoryModel` private APIs (see `ane_bridge.m`). MIL graphs are generated at runtime for each unique projection shape, compiled once, and reused across all 64 layers.

## Sharing Results

Please include in your report:
1. The chip line from the output (e.g. `Apple M4 Max (16 cores, 128 GB)`)
2. tok/s at S=256, S=512, S=1024
3. macOS version (`sw_vers`)

## License

MIT
