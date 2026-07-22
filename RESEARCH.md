# colibrì fork — research findings & work log

A record of the work done on this fork (`noobdev-ph/colibri`) — AMD GPU support,
generation-quality guards, performance profiling, and the strategic research that
followed. Paused pending upstream traction and performance; this document is the
resume point.

Hardware for all measurements: **AMD Ryzen 7 5700G** (8c/16t, Zen 3, AVX2-only —
no AVX-512), **AMD Radeon RX 9070 XT** (gfx1201 / RDNA4, 16 GB, wave32),
64 GB DDR4-3200, Crucial P510 NVMe. Model: GLM-5.2 744B MoE, int4 container,
int8 MTP head.

---

## 1. Contributions shipped upstream

Both PRs **merged into upstream** (colibrì **v1.1.0**) — the AMD HIP/ROCm backend
is now part of the engine, not a fork-only patch:

- **[#339] AMD GPU support via HIP/ROCm** — the existing CUDA backend compiles
  unchanged for AMD through a single mapping header (`backend_gpu_compat.h`),
  the same one-shim pattern `compat.h` uses for Windows. `make HIP=1`,
  `make hip-test`. Includes a **WMMA compile-gate** (`COLI_GPU_HAS_WMMA`):
  the tensor-core kernels are `__CUDA_ARCH__ >= 700` guarded, but gfx GPUs
  report `compute_major = 12`, so without a compile-time gate the runtime
  dispatch would select empty kernel bodies and return garbage silently.
  First AMD GPU to run the engine.
- **[#338] GPU backend failure-path hardening** — vendor-neutral. Fixes a
  sticky `cudaGetLastError` bug (a failed allocation poisoned the next healthy
  launch's error check), hardens the cached-tensor upload contract, and adds a
  `COLI_GPU_FAIL_AFTER=N` fault-injection hook that gates all 19 GPU compute
  entry points so the engine's CPU fallback / host-rematerialization can be
  tested end-to-end without real hardware faults.

Docs: `GPU_BACKENDS.md` (environments, build matrix, runtime knobs, validation).

### Adapting to v1.0.0 — zero code changes needed
When upstream tagged v1.0.0, the AMD code required **no adjustment**: all 33
CUDA runtime symbols v1 uses were already mapped; the WMMA dispatch still uses
`__CUDA_ARCH__ >= 700` + `compute_major` so the gate still applies; the
`refactor/split-colibri` (#391) kept `glm.c` monolithic. Merges were mechanical
one-line unions (`.PHONY`, `TEST_BINS`, a CI job).

---

## 2. Generation-quality guards (the "churn" work) — experimental

**Problem:** int4 greedy decoding degenerates on long single-artifact
generations (e.g. asking the model to write a complete HTML game in one shot).
Two distinct failure modes were observed and characterized:

1. **Tight token loops** — a fixed token sequence repeats at a fixed period
   (observed: `let spawnRateCurrentDelta = 0;` ×23, periods 8–15).
2. **Shuffled redundant declarations** — the *same lines* recur reordered and
   interleaved, so no fixed token-period ever holds (observed: **343 constant
   declarations when ~10 were needed**, and the model never reached the game
   logic before the token budget ran out). This slips under a period detector.

**What was built** (`churn.h`, `tests/test_churn.c`, env-gated, zero effect unset):
- `MIN_TOKENS=N` — mask stop tokens until N tokens emitted.
- `UNTIL=<str>` — mask stop tokens until the output contains a marker
  (hard semantics: no marker → run to `--ngen`).
- `NOLOOP=1` — periodic-loop guard. **v0** banned the continuation token —
  which *failed*: the model escaped the ban into "I apologize, let me restart"
  and re-typed the whole file (a meta-loop). **v1** replaced the ban with a
  **KV rewind**: on a detected loop, roll the KV cache + history back to before
  the loop (regenerate logits with one `step()` — the engine already relies on
  the "KV beyond current is stale" invariant) and reroll with escalating
  temperature. v1 killed the meta-loop (0 restarts vs 4+ under v0).

**Status / open gap:** v1 fixes the tight-loop failure but **not** the shuffled
redundant-declaration failure (it stays under the period-detection threshold).
The next step is a **line-level novelty detector** (rolling set of recent
line-hashes; trip when novelty drops below ~30%) feeding the same rewind
machinery. This is the finish line for the churn track and a genuinely novel
capability upstream hasn't addressed. The five recorded failure spirals are the
calibration fixtures. Branch: `feat/churn-sampling` (v0) and `local/churn-hip`
(v1 integration).

**Practical note for int4 code-gen:** the model one-shots *setup* (HTML/CSS)
well but pads/degenerates before completing complex logic. "Decompose then
assemble" (spec → per-function blank-page prompts → mechanical assembly) plays
to the model's proven strength (blank-page generation) over its weakness
(continuation), and is the likely right frame if the churn guards prove
insufficient.

---

## 3. Performance profiling (Phase 0) — the most important finding

Goal: before building rocWMMA matrix-core kernels, measure **where the expert
matmul time actually goes** on this machine. The engine profiles prefill and
decode separately and reports a CPU-vs-GPU expert-matmul split and achieved
bandwidth (`PROF=1 COLI_CUDA_PROFILE=1`).

**Representative code-gen workload (107-token prompt, 150 tokens generated,
GPU tier of 620 experts / 11.7 GB resident):**

| metric | prefill (large-M) | decode (M=1) |
|---|---|---|
| expert-matmul total | 43.7 s | **214.0 s** |
| …on the CPU | 42.7 s | **211.3 s (98.7%)** |
| …on the GPU | 1.0 s | 2.66 s (**1.2%**) |
| CPU matmul throughput | 4.25 GB/s | 4.10 GB/s |
| expert hit-rate | — | 49% (GPU pin **10.5%** + CPU LRU 38.7%) |
| decode attention | — | 83.4 s (grows with context) |

### Conclusion 1 — rocWMMA is the wrong lever for this machine
The GPU does **1.2%** of the expert matmul; the CPU does **98.7%**. The cause:
only **10.5% of routing lands on a VRAM-resident expert** (16 GB holds ~620 of
21,504 experts). rocWMMA accelerates GPU matmul — already tiny and already fast.
Its absolute ceiling here is ~1%. **The original Phase 1 (port rocWMMA) was
abandoned on this evidence.** (Measure before building: the same discipline that
earlier found the "mirror" VRAM-tier mode was performance-neutral on a
disk-bound machine.)

### Conclusion 2 — the real bottleneck is CPU expert matmul at ~4.1 GB/s
Decode is bound by the CPU computing ~90% of experts (LRU-cached or streamed) at
4.1 GB/s on the AVX2 int4 kernel. This is **compute-bound on the CPU**, well
below the ~50 GB/s memory bandwidth — the 5700G (Zen 3, AVX2-only) is at its
software ceiling. The AVX-512-VNNI (`dpbusd`) paths already in the code are
**dead on this CPU**; they light up only on Zen 4+/Intel.

### Conclusion 3 — a hardware-specific inversion worth prototyping ("Lever B")
The engine's founding rule is "streaming stays CPU-side; copying experts to the
GPU per-use only trades the disk bottleneck for a PCIe bottleneck." But on this
machine that assumption **inverts**:

- CPU matmul: **4.1 GB/s** (measured)
- PCIe Gen3 x16: **13.4 GB/s** (measured)

Copying a 19 MB LRU-resident expert to the GPU (~1.4 ms) + GPU compute is
roughly **3× faster** than the CPU computing it in place (~4.6 ms). So on a
slow-CPU / faster-PCIe box, **letting the GPU compute the RAM-cached (LRU)
experts — not just the pinned ones — attacks the real 211 s bottleneck.** This
is a scheduling/tiering change in the dispatch logic (not a new kernel), it is
specific to a hardware ratio no other profiled machine has, and it is the
recommended next prototype. First step is a micro-benchmark confirming the ~3×
per-expert before building. Caveats: raises shared-bus PCIe traffic; needs a
per-expert path-selection heuristic.

---

## 4. Where a fork can uniquely improve colibrì (strategic research)

The engine is already deeply optimized (AVX-512 VNNI, ARM i8mm, io_uring, MLA
weight absorption, page-cache eviction control, a router-lookahead prefetcher,
NUMA awareness). There are no overlooked easy wins. Ranked by leverage *only
this fork holds* (AMD hardware + the generation-quality detour):

1. **GPU-compute the LRU experts (Lever B, §3)** — attacks the measured
   bottleneck; hardware-specific; tiering not kernels. **New #1 after Phase 0.**
2. **Generation-quality guards finished** (line-novelty detector, §2) — novel,
   pure C, CI-testable.
3. **GPU-safe speculation (issue #163)** — MTP drafting collapses under GPU
   float numerics. A numerics-matched **integer (int8×int4) WMMA** kernel would
   match the CPU IDOT path and restore speculation. Note: RDNA4 has native
   integer WMMA, and its wave32 maps cleanly onto the existing NVIDIA
   32-lane-warp kernels — so *if* a GPU kernel is ever built, the integer
   variant (not fp16) is the one worth building, because it also fixes #163.
4. **Mixed-precision quantization (int3 / per-layer bit budget)** — README flags
   int2/int3 quality as unmeasured; the engine already carries int2/4/8 +
   grouped scales. A smaller on-disk model = less to stream = faster cold decode.
   Gated on running the eval harness.
5. **Attention tiling** — decode attention is 19% and *grows* with context
   (measured 83 s above). Flash-style tiling of score·softmax·value, or better
   DSA sparse-key selection. Touches validated MLA math (higher risk).

---

## 5. Rust / Go port — verdict

Assessed by whether a port fixes a class of problem colibrì *has* without
breaking what makes it work. The decisive evidence is the **bug history** —
colibrì's real failures were never bad math, they were memory-ownership and
concurrency: an OOM from a slack miscalculation, a Windows `0xC0000374` heap
corruption from `free()` on aligned memory, an mlock that wired 363 GB instead
of 231 GB (a host pointer not nulled after GPU upload), the sticky
`cudaGetLastError`. Exactly the class Rust's borrow checker catches and Go's GC
obscures.

- **Go — reject.** Its GC cannot model a 34 GB pinned set plus an evictable
  streaming cache with `mlock`/`FADV_DONTNEED` control. No portable SIMD — the
  VNNI/i8mm kernels become assembly or cgo. It would regress the exact things
  that make the engine work.
- **Rust — yes, but only as a targeted subsystem, never a wholesale rewrite.**
  Reimplement one bounded, bug-prone unit — the **expert streaming cache +
  prefetch scheduler** (the `uring.h` + tier/pin/LRU logic, where the ownership
  bugs live) — behind the existing C ABI, exactly how `backend_cuda` already
  plugs in. Prove it bit-identical and no slower; then the compiler *guarantees*
  the invariants that have bitten us. Wholesale port of 6,657 token-exact lines:
  no (risk + against the single-file, zero-dependency spirit).

---

## 6. Benchmark record (this machine)

| config | tok/s | expert hit | notes |
|---|---|---|---|
| CPU only, cold | 0.16 | ~30% | no pin, blank usage history |
| CPU only, warm | 0.22 | 38% | learned pin + `--topp 0.7` + int8 MTP |
| HIP mirror mode | 0.22 | 33% | GPU tier mirrors RAM pin — neutral (disk-bound) |
| HIP `CUDA_RELEASE_HOST` | **0.33** | 61% | VRAM extends the pin; best sustained result |
| profiling code-gen (§3) | 0.24 | 49% | domain-mismatched pin (10.5% GPU hit) |

Hardware facts established: the 5700G APU caps all PCIe at **Gen3** (the Gen5
NVMe negotiates Gen3 x4 ≈ 3.5 GB/s; the GPU runs Gen3 x16 ≈ 13.4 GB/s). The CPU
is at its AVX2 ceiling. The single highest-value hardware upgrade is a Zen 4 CPU
(unlocks the already-written AVX-512-VNNI path + more cores + Gen4 disk).

---

## 7. Resume points

- **Merge status:** PRs #338 and #339 **merged** into upstream v1.1.0. The #509
  TEMP/ROCm crash was fixed upstream via `COLI_TEMP` (the root-cause fix we
  endorsed); our defensive backend scrub was superseded and dropped from the fork.
  Full-model fmt=4 gs64 validation on the RX 9070 XT: coherent + executable output,
  GPU-computed grouped experts, token-identical to CPU (posted to #339).
- **Highest-value next build:** Lever B micro-benchmark (§3) — confirm the ~3×
  CPU-vs-PCIe-GPU per-expert, then prototype GPU compute for LRU experts.
- **Finish the churn track:** line-novelty detector (§2) → then the acceptance
  test (a fresh game spec, generated with zero human lines).
- **Cheap parallel win:** run the eval harness for a mixed-int3 quality sweep
  (§4.4) — our hardware can now afford runs upstream could not; even a null
  result is a contribution the project asked for.

Model (~363 GB int4) removed locally to reclaim disk; re-download via
`coli convert` or the Hugging Face int4 container when work resumes.
