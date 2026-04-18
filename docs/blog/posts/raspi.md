---
date: 2026-04-18
comments: true
image: blog/assets/raspi/smollm.png
social_title: SmolLM on a Smol Machine
---

# SmolLM on a Smol Machine: Optimizing LLM Inference on a $15 Computer

I got SmolLM, a 360M-parameter LLM, running at **2.6 tokens/sec on a $15 Raspberry Pi Zero 2W**. The naive version ran at 0.015 tok/s, effectively unusable. This post breaks down exactly what made it **~170x faster**.

![SmolLM generating on a Raspberry Pi Zero 2W](assets/raspi/raspi-demo.gif)

<!-- more -->

Each technique here is used by production inference engines (llama.cpp, vLLM, TensorRT-LLM). Here, I'll be showing them one at a time so you can see what each one contributes. [Check the code](https://github.com/AyoubMDL/rust_llm).

!!! note "Inspiration"
    This project was inspired by [picolm](https://github.com/RightNow-AI/picolm).

---

**Prerequisites:** familiarity with transformers at a high level (attention, FFN, embeddings). If you're new to them, [Karpathy's "Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) is a great starting point. SmolLM is a standard decoder-only transformer (LLaMA-style); I'll only cover details that matter for performance.


## The setup

* **Engine:** [rustllm](https://github.com/AyoubMDL/rust_llm). Supports any LLaMA-architecture model in safetensors or GGUF format (Q4_K, Q5_0, Q6_K, Q8_0; **not all GGUF quantization types are supported**).

* **Hardware:** Raspberry Pi Zero 2W, 512 MB RAM, quad-core ARM Cortex-A53 at 1 GHz.

* **Model:** SmolLM 360M (360 million parameters, 32 layers, 960 hidden dim, 15 query heads, 5 KV heads).

---

## The starting point: why format matters

Here are the key fields from SmolLM 360M's [`config.json`](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct/blob/main/config.json) we'll reference throughout the post:

```json
{
  "hidden_size": 960,
  "intermediate_size": 2560,
  "num_attention_heads": 15,
  "num_key_value_heads": 5,
  "num_hidden_layers": 32,
  "max_position_embeddings": 2048,
  "vocab_size": 49152,
  "tie_word_embeddings": true
}
```

Here's what SmolLM 360M looks like in three different weight formats on the Pi:

| Format | Weight size | Model size | Fits in RAM? | Generation |
|--------|-----------|-----------|-------------|------------|
| f32 (safetensors) | 4 bytes | ~1.4 GB | No | 0.015 tok/s |
| bf16 (safetensors) | 2 bytes | ~720 MB | No | 0.03 tok/s |
| Q4_K_M (GGUF) | ~0.56 bytes | ~210 MB | Yes | **2.6 tok/s** |

Two things stand out:

* **bf16 is exactly 2x faster than f32.** Half the bytes, double the speed. Both models exceed the Pi's 512 MB RAM, so every forward pass reads weights from the SD card via mmap. The Pi Zero 2W's SD card bus is limited to about [23 MB/s](https://forums.raspberrypi.com/viewtopic.php?t=232148), and bf16 reads half as many bytes. This proves that these runs are purely I/O bound.

* **GGUF is 87x faster than bf16.** Two things stack to get there. First, 210 MB fits entirely in the Pi's RAM, so weights stream from LPDDR2 DRAM (~3.2 GB/s peak, 32-bit bus × 800 MT/s) instead of the SD card. That alone shifts the workload from I/O-bound to compute-bound. Second, *only because that shift happens* do the rest of the techniques in this post actually pay off: on the bf16 run, parallel matmul and NEON SIMD bought nothing because the cores were stalled on storage. Strip those optimizations from the Q4_K_M run and you'd be well below 1 tok/s, but nowhere near 2.6.

The 2.6 tok/s number includes every optimization in this post (KV cache, parallel matmul, fused dequantization, NEON SIMD). The sections below show what happens when you disable each one.

For context: 2.6 tok/s is slow compared to a desktop CPU (~20-50 tok/s for similar-size models) or any GPU, but it's fast enough for interactive use with short prompts. On a $15 device with 512 MB of RAM, that's usable.

---

## The optimizations, one by one

### 1. mmap: zero-copy weight loading

mmap is what makes this possible on 512 MB.

**The problem:** Loading model weights normally means: allocate memory, read all bytes, parse. SmolLM f32 at 1.4 GB can't fit in 512 MB that way.

**The fix:** Memory-mapped I/O (`mmap`) maps the file into virtual address space without reading it. The OS loads pages on demand and evicts them under memory pressure. Weight bytes are used directly from the mmap'd region. No copying, no deserialization.

**What actually lives in RAM (SmolLM 360M Q4_K_M on Pi Zero 2W):**

| Component | Size |
|-----------|------|
| Model weights (mmap'd) | ~210 MB |
| KV cache (f32, 2048 context) | ~160 MB |
| Activation buffers | ~0.5 MB |
| Tokenizer | ~2 MB |
| **Total** | **~373 MB** (of 512 MB) |

The KV cache stores Keys and Values for every past token, across every layer:

```
KV cache = max_seq_len × 2 (K and V) × n_layers × n_kv_heads × head_dim × 4 bytes
         = 2048        × 2            × 32       × 5           × 64       × 4
         = 160 MB
```

With GQA (see next section), SmolLM uses only 5 KV heads instead of 15 query heads. Without GQA this would be 480 MB, which wouldn't fit on the Pi at all. The cache could also be stored in lower precision (FP16, or even quantized) to halve or quarter the size, but this engine uses f32 for simplicity.

---

### 2. GQA: Grouped Query Attention

**The problem:** Standard multi-head attention gives every head its own K and V projections. SmolLM with 15 heads would need 15 K/V pairs per layer, expensive to compute and store.

**The fix:** GQA shares K/V heads across groups of query heads. SmolLM uses 15 query heads but only 5 KV heads, a 3:1 ratio. A more aggressive variant, Multi-Query Attention (MQA), uses a single KV head for all query heads, but at the cost of degraded output quality. GQA is a good middle ground:

```
MHA (15 KV heads):  Q0→K0,V0  Q1→K1,V1  ...  Q14→K14,V14
GQA (5 KV heads):   Q0..Q2→K0,V0  Q3..Q5→K1,V1  ...  Q12..Q14→K4,V4
```

If you come from a computer vision background, the pattern is the same idea as **grouped convolutions**: instead of every output channel getting its own set of filter weights (standard convolution / MHA), groups of output channels share one filter (grouped conv / GQA). The extreme case, **depthwise convolution** (one filter per group), is the direct analog of Multi-Query Attention.

**Impact:**

| Metric | MHA (15 KV heads) | GQA (5 KV heads) |
|--------|-------------------|-------------------|
| KV cache (SmolLM, 2048 ctx, f32) | ~480 MB | ~160 MB |
| KV projection compute per layer | 15 K + 15 V | 5 K + 5 V |

GQA is an architecture choice made during training, not a runtime optimization you apply. But understanding it explains why new architectures adopt it.

---

### 3. KV Cache: O(n) inference instead of O(n^2)

**The problem:** Without caching, generating the 100th token requires re-computing Keys and Values for all 100 previous tokens. The 101st token re-computes for 101 tokens. Cost grows quadratically with sequence length.

**The fix:** Keys and Values for past tokens never change (the model is causal, token 42 can't attend to token 43). Cache them. Each new token computes only its own K and V, appends to the cache, then attends over the full history. Without a KV cache, the model “re-reads the whole book” for every new word. With it, it just appends a new page.

```
Without cache:  token 100 → compute K,V for all 100 tokens → attend
With cache:     token 100 → compute K,V for token 100 only → append → attend
```

The cost is memory: for SmolLM 360M at 2048 context, the KV cache is ~160 MB in f32 (see section 1). On a Pi with 512 MB, that's ~31% of total RAM.

**Impact on Pi Zero 2W (SmolLM 360M Q4_K_M):**

All other optimizations (NEON SIMD, parallel matmul, fused dequant) are enabled in both rows. The only variable is the KV cache.

| Config | tok/s |
|--------|-------|
| No KV cache (recompute everything) | 0.05 |
| With KV cache | 2.6 |

Every production inference engine uses a KV cache. It's non-negotiable.

---

### 4. Parallel matmul: use all cores

**The problem:** Matrix-vector multiply is ~95% of inference time. SmolLM has 225 matmuls per forward pass (7 per layer × 32 layers + 1 lm_head). Each matmul computes thousands of independent dot products, but a single-threaded loop runs them one at a time.

**The fix:** Each output row is independent, so we parallelize across rows with rayon. No shared writes, no synchronization.

```rust
// Single-threaded: one row at a time
for i in 0..out_rows {
    out[i] = dot(weight_row[i], x);
}

// Parallel: each core takes a chunk of rows
out.par_iter_mut().enumerate().for_each(|(i, o)| {
    *o = dot(weight_row[i], x);
});
```

**Impact on Pi Zero 2W (SmolLM 360M Q4_K_M):**

| Config | tok/s |
|--------|-------|
| 1 thread | 0.7 |
| 4 threads | 2.6 |

3.7x speedup from 4 cores. This works because with quantized weights and NEON SIMD, the workload is compute-bound, not memory-bound. The cores have enough work to stay busy.

On memory-bound workloads (like f32 weights), multi-threading helps less. The 4 Cortex-A53 cores share a single L2 cache and a single bus to DRAM. As [Chips and Cheese](https://chipsandcheese.com/p/arms-cortex-a53-tiny-but-important) puts it: "we're really limited by the memory setup rather than the cores."

---

### 5. Weight quantization: from f32 to 4-bit

Quantization does two things:

1. Makes the model fit in RAM
2. Reduces memory bandwidth per token

**The problem:** f32 weights mean 4 bytes per parameter. SmolLM 360M in f32 is ~1.4 GB, which doesn't fit in the Pi's 512 MB RAM. Even bf16 (2 bytes, ~720 MB) doesn't fit. Every forward pass pages weights from the SD card.

**The fix:** Quantize weights to 4 bits (or less). The GGUF format (I'll write a separate post on the GGUF binary format) stores weights in blocks where each value is a 4-bit nibble plus per-block scale factors.

```
Q4_K block: 256 weights in 144 bytes (~4.5 bits/weight)

bytes [0..2]    d      : FP16 super-block scale
bytes [2..4]    dmin   : FP16 super-block minimum
bytes [4..16]   scales : 12 bytes, packed per-group scale+min
bytes [16..144] qs     : 128 bytes, 256 weights as 4-bit nibbles

Dequantize: weight = d * scale * nibble - dmin * min
```

Each byte stores two weights. Q4_K_M uses mixed precision, promoting sensitive layers to higher precision: Q5_0 for most attention and FFN tensors (~75%, see [section 7](#7-neon-simd-vectorized-inner-loops)), and Q6_K or Q8_0 for the most sensitive ones.

**Impact on Pi Zero 2W (SmolLM 360M):**

| Format | Model size | Fits in RAM? | tok/s |
|--------|-----------|-------------|-------|
| f32 | ~1.4 GB | No | 0.015 |
| bf16 | ~720 MB | No | 0.03 |
| Q4_K_M | ~210 MB | Yes | **2.6** |

The jump from bf16 to GGUF isn't just about fewer bits. It's about fitting in RAM. The bf16 vs f32 comparison (exactly 2x, half the bytes) proves the SD card is the bottleneck. Once quantized weights fit in memory, inference shifts from I/O-bound to compute-bound, and all the other optimizations (SIMD, parallelism) can actually help.

On GPUs, where inference is more clearly memory-bound, quantization plays an even bigger role since the compute cost of dequantization is negligible compared to the memory bandwidth savings.

---

### 6. Fused dequantize + dot product: no temp buffers

**The problem:** The naive approach to quantized matmul is two-pass: first dequantize an entire weight row into a temporary buffer, then compute the dot product against the input vector. This means one heap allocation per row, one pass to decode all weights, and a second pass to multiply-accumulate.

```
Naive (two-pass):
  1. allocate temp buffer
  2. dequantize entire row → temp buffer     ← pass 1: read quantized, write f32
  3. dot(temp buffer, input)                 ← pass 2: read f32, multiply-accumulate
  4. free temp buffer

Fused (single-pass):
  for each block in row:
    decode block → multiply with input → accumulate
  no allocation, no temp buffer
```

**The fix:** Fuse dequantization and accumulation into a single pass. Each quantized block is decoded and immediately multiplied with the input. No temporary buffer, no second pass. This eliminates the heap allocation and halves the memory traffic, since decoded weights are never written to memory and read back.

This matters because matmul is ~95% of inference time. Eliminating the allocation and the extra memory pass reduces both latency and cache pressure. The same principle applies on GPUs: dequantizing separately means writing decoded weights back to VRAM and then reading them again for the dot product. Fusion eliminates that round-trip entirely.

---

### 7. NEON SIMD: vectorized inner loops

**The problem:** The Cortex-A53 can process 4 single-precision floats per cycle with NEON (128-bit SIMD), running at 1 GHz. The scalar matmul inner loop, one multiply-add per iteration, uses only a fraction of that throughput.

**The fix:** ARM NEON processes 4 floats simultaneously using 128-bit registers:

```rust
// Scalar: 1 multiply-add per iteration
for i in 0..n {
    sum += a[i] * b[i];
}

// NEON: 4 multiply-adds per iteration
let mut acc = vdupq_n_f32(0.0);         // [0, 0, 0, 0]
while i + 4 <= n {
    let va = vld1q_f32(a.ptr(i));       // load 4 floats
    let vb = vld1q_f32(b.ptr(i));       // load 4 floats
    acc = vmlaq_f32(acc, va, vb);       // acc += a * b (4-wide)
    i += 4;
}
let sum = vaddvq_f32(acc);              // sum all 4 lanes
```

For quantized weights, the NEON path is more involved. Q4_K stores two 4-bit weights per byte. The inner loop loads 8 bytes (16 weights), extracts nibbles with bitwise ops, widens through `u8 → u16 → u32 → f32`, then multiply-accumulates, all without materializing a temporary buffer.

The NEON kernels use the fused approach from [section 6](#6-fused-dequantize-dot-product-no-temp-buffers): dequantize and accumulate in a single pass, no allocation, no temp buffer, each value used immediately.

For SmolLM 360M Q4_K_M, 75% of weight tensors are Q5_0 format. Before adding a dedicated NEON [vec_dot_q5_0](https://github.com/AyoubMDL/rust_llm/blob/2d9048474bd3a6836465060b4ae8312841f642c7/src/components/simd.rs#L275), those went through the slow fallback (allocate, dequantize, scalar dot). The dedicated kernel was the single biggest speedup of the project.

**Impact on Pi Zero 2W (SmolLM 360M Q4_K_M):**

| Config | tok/s |
|--------|-------|
| Scalar fallback (alloc + dequant + scalar dot) | 0.5 |
| NEON fused vec_dot | **2.6** |

---

## Conclusion

Every optimization in this post targets the same bottleneck from a different angle: reduce bytes moved per token. Quantization shrinks the weights. mmap avoids copying them. Fused kernels avoid writing intermediate results. SIMD processes more values per cycle.

Here's the impact of each one in isolation (disabling it while keeping everything else on):

| Optimization disabled | tok/s | vs full (2.6) |
|----------------------|-------|---------------|
| No KV cache | 0.05 | 52x slower |
| No SIMD (scalar fallback) | 0.5 | 5.2x slower |
| No parallelism (1 thread) | 0.7 | 3.7x slower |
| **All optimizations on** | **2.6** | |

And the format progression that got the model into RAM in the first place:

| Format | Fits in RAM? | tok/s |
|--------|-------------|-------|
| f32 (1.4 GB) | No | 0.015 |
| bf16 (720 MB) | No | 0.03 |
| Q4_K_M (210 MB) | Yes | **2.6** |

That's the end. There's a lot here (KV caching, quantization, SIMD, fused kernels) and each one is a rabbit hole on its own. The post compresses weeks of work into a few minutes of reading, but the techniques themselves don't compress the same way. If anything caught your eye, my suggestion is to pick one, ignore the rest, and go deep: read how [llama.cpp](https://github.com/ggerganov/llama.cpp) implements it, profile it on your own hardware, try to break it and see what changes. That's where the real understanding comes from.

---

## References

- [rust_llm](https://github.com/AyoubMDL/rust_llm) - the inference engine used throughout this post
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - the canonical reference for GGUF and CPU LLM inference
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) - Sebastian Raschka, walkthrough of LLM architectures from the ground up
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - Ainslie et al., 2023
- [SmolLM 360M Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct) - HuggingFace, the model used in this post
