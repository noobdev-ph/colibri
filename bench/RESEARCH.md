# colibrì fork — research findings & work log

A record of the work done on this fork (`noobdev-ph/colibri`) — AMD GPU support,
generation-quality guards, performance profiling, the Vulkan-backend evaluation,
and the strategic research that followed. This document is the resume point.

**Current best config on this machine:** `COLI_VULKAN=1 DRAFT=0` at 0.40 tok/s
(§7) — with the two knobs that matter most being *disabling speculation* (+65%)
and *enabling Resizable BAR in firmware* (+118%, and it fails silently without).

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
| **Vulkan, `DRAFT=0`** | **0.40** | 55% | §7 — fastest config measured on this box |
| HIP 12 GB, `DRAFT=0` | 0.33 | 60% | §7 — same binary, speculation off |
| Vulkan, MTP on | 0.24 | 40% | §7 — MTP costs ~40% when I/O-bound |
| HIP 12 GB, MTP on | 0.22 | 47% | §7 — same penalty, both backends |

**`DRAFT=0` is the single biggest knob on this machine** (§7): disabling
speculation is worth +65% on Vulkan and +53% on HIP. Every pre-§7 row above ran
with MTP enabled on the CPU/Vulkan paths, so they understate what this hardware
can do.

Hardware facts established: the 5700G APU caps all PCIe at **Gen3** (the Gen5
NVMe negotiates Gen3 x4 ≈ 3.5 GB/s; the GPU runs Gen3 x16 ≈ 13.4 GB/s). The CPU
is at its AVX2 ceiling. The single highest-value hardware upgrade is a Zen 4 CPU
(unlocks the already-written AVX-512-VNNI path + more cores + Gen4 disk).

---

## 7. Vulkan backend evaluation (PR #418) — 130 controlled runs

Upstream issue: **JustVugg/colibri#523**. Tested at `550df5c` (head of #418) on
the RX 9070 XT (gfx1201, RADV, Mesa 26.1.3, Vulkan 1.4.348). v1.1.1 was checked
and deliberately *not* merged into the test branch — the `g_draft` guard is
byte-identical, `npin+=prefix_est` unchanged, `backend_cuda.cu` has no diff, and
`backend_vulkan.c` exists only on the PR branch, so rebasing would have meant
benchmarking something that isn't the PR.

### 7.1 Resizable BAR is mandatory, and its absence fails silently

`pick_memtype()` requires `HOST_VISIBLE`, and the expert tier wants
`HOST_VISIBLE|DEVICE_LOCAL`. With ReBAR **off** that type exists only in a 256 MB
window, so the tier silently falls back to system RAM *while reporting success*:

| | ReBAR off (256 MB BAR) | ReBAR on (16 GB BAR) |
|---|---|---|
| tok/s | 0.11 | 0.24 |
| GPU VRAM in use | **82 MB** | **6,724 MB** |

The log said `320 hot experts resident (6.04 GB VRAM)` while the card held 82 MB
and every access crossed PCIe — slower than the pure CPU path, with nothing in
the output indicating the degraded mode. Diagnosed only via sysfs. Enabling
ReBAR in firmware is the fix; upstream was asked for an init-time warning that
compares the chosen memory type against the heap size.

**Operational note for this machine:** `llama-server` is an enabled user service
and returns after every reboot holding ~15 GB of VRAM. `systemctl --user stop
llama-server` before any GPU work.

### 7.2 The confound — and the methodology lesson

The first 50-run battery concluded HIP was 32–35% faster than Vulkan. **That was
wrong**, and the cause was ours: `g_draft` resolves as

```c
if(g_draft<0){ g_draft = (m.has_mtp && (!g_cuda_enabled || cuda_mtp)) ? 3 : 0; }
```

so setting `COLI_CUDA=1` on the HIP arms disabled MTP there, while the Vulkan
arms drafted at 2.00 tokens/forward. The two arms were not running the same
decode configuration. The engine printed `MTP ACTIVE (draft=3)` vs `draft=0` in
every log collected; it was seen, misattributed, and not followed up.

A second error compounded it: `experts loaded/token` was used as evidence that
the Vulkan tier wasn't on the critical path. It is a **router-side counter**
(`m->ereq += Ke`, `colibri.c:3069`) incremented before any tier is consulted, so
it cannot show that. Per *position* the arms were always near-identical
(338.5 vs 330.1).

> **Lesson worth keeping:** the battery was rigorous about everything it knew to
> control — interleaved arms, frozen `.coli_usage`, fresh process per run,
> medians over 10 runs, no-overlap checks — and that rigour made a wrong answer
> look authoritative. Careful methodology downstream of an unverified assumption
> amplifies error rather than catching it. **Verify what a counter counts before
> drawing a conclusion from it, and assert that both arms of an A/B are in the
> same mode before trusting the delta.**

Fix: set `DRAFT` **explicitly** on every arm. The guard only runs under
`if(g_draft<0)`, so an explicit value bypasses it on both backends.

### 7.3 Corrected results — Vulkan wins (80 runs, `DRAFT=0` both)

| arm | n | tok/s | hit% | vk bucket |
|---|---|---|---|---|
| **Vulkan 320 (6.04 GB)** | 10 | **0.3958** | 54.7% | 24.3% |
| HIP 6 GB | 10 | 0.3200 | 57.3% | — |
| **Vulkan 640 (12.08 GB)** | 10 | **0.3975** | 54.7% | 33.0% |
| HIP 12 GB | 10 | 0.3342 | 60.0% | — |

**Vulkan +23.7% at ~6 GB, +18.9% at ~12 GB, no overlap between arms.** Vulkan
wins with a *lower* hit rate, so it is winning on compute, not caching. HIP
reproduced its earlier numbers to within 0.4%, confirming no rig drift.

### 7.4 MTP is an I/O amplifier — the most transferable finding

| | MTP off | MTP on | cost |
|---|---|---|---|
| Vulkan 640 | 0.3975 | 0.2387 | **−39.9%** |
| HIP 12 GB | 0.3342 | 0.2176 | **−34.9%** |

Both backends lose roughly equally, so this is a property of speculation on
storage-bound hardware, not of any backend. With `g_draft=3` each forward
evaluates 4 positions; at acceptance *a* you emit `1+3a` tokens, so

```
positions per emitted token = 4 / (1 + 3a)
```

At our measured 33% acceptance that is **2.0**, and `ereq/token` goes
330.1 → 674.5 — a 2.04× match to the model. **Only at 100% acceptance is MTP
I/O-neutral.** Speculation trades fewer forward passes (saving dense compute) for
more positions (costing expert streaming); on this box dense compute is nearly
free and expert streaming is the bottleneck, so it pays 2× the scarce resource to
save on the abundant one. Hit rate also drops ~14pp, because rejected drafts load
experts the real token stream never wanted and evict ones it did.

Acceptance itself is healthy — **33%, stable across all 20 Vulkan runs**, and the
`[MTP]` auto-pause at `colibri.c:4910` never fired. So the `#163` CUDA guard's
stated rationale (acceptance collapse under a GPU tier) does not hold here, yet
the guard's *effect* would still help. The useful predicate is likely **"is
expert I/O the bottleneck"**, not "is a GPU tier active" — which would cover
Vulkan without a special case.

### 7.5 Tiny CPU cache — RAM was never masking residency

`cap=16` (per-layer LRU cut 20× from 320), `DRAFT=0`:

| arm | tok/s | vs `cap=320` |
|---|---|---|
| Vulkan 640 | 0.3741 | −5.9% |
| HIP 12 GB | 0.3429 | +2.6% |

Vulkan +9.2%, no overlap after excluding one outlier. Shrinking the LRU barely
moved either backend, because the RAM pin tier was already auto-capped to 10
experts (`cap lowered 320->10, projected peak 50.7 GB`) — the VRAM tiers were
doing essentially all the work. A prediction that the gap would *widen* was
wrong; it narrowed.

### 7.6 Data-quality practices that earned their keep

- Compute throughput from `decode_s`, not the printed `tok/s` — the latter rounds
  to 2 decimals and turned a true +2.35% into an apparent +4.2%.
- Freeze `.coli_usage` to one snapshot restored before every run, or the learning
  cache drifts and silently changes what each arm holds.
- Interleave arms; never run them in blocks.
- Check for outliers *and* for temporal structure. `HIP12mtp` was bimodal (5 runs
  ~0.175, 5 ~0.22, clean step at run 12) — but acceptance, `fw/tok`, `ereq` and
  hit rate were identical across all ten, proving it environmental (page-cache
  warming of MTP's doubled working set), not algorithmic.
- Scripts: `full-battery.sh`, `gpu-battery.sh`, `vk-battery.sh`, `hip-default.sh`,
  `vk-analyze.py` (local, untracked). Bash trap hit twice: a helper's loop
  variable must be `local` or it clobbers the caller's run counter.

### 7.7 Outcome

Filed as #523 with the correction posted prominently, the issue body bannered and
the title amended. Upstream verdict: the Vulkan backend is **correct on RDNA4 on a
real fmt=4/gs=64 container with no ROCm installed** (130/130 runs clean output),
and on this hardware it is the **fastest configuration measured** — 0.40 tok/s,
against 0.33 for HIP and 0.22 for the previous CPU best.

> **Footnote — results that live only in #523.** The issue thread continued past
> this writeup and carries one measurement not written up above: a 20-run
> `COLI_MMAP` A/B run 2026-07-24, answering @xxxajk's question of whether the
> expert LRU uses MMIO. It does not, and enabling it costs throughput — the
> default `pread`+slab path beats `COLI_MMAP=1` by **+25.4%** (paired median,
> wins 9 of 10 pairs, sign test p = 0.021), with an *identical* hit rate in both
> arms, so the cost is the fault path plus memory pressure rather than any
> caching difference. Raw data: `mmap-battery/results.csv`. See
> [#523](https://github.com/JustVugg/colibri/issues/523) for the full exchange,
> including JustVugg's 2026-07-29 note that PR #418 has since merged to `dev`.

---

## 8. Resume points

- **Merge status:** PRs #338 and #339 **merged** into upstream v1.1.0. The #509
  TEMP/ROCm crash was fixed upstream via `COLI_TEMP` (the root-cause fix we
  endorsed); our defensive backend scrub was superseded and dropped from the fork.
  Full-model fmt=4 gs64 validation on the RX 9070 XT: coherent + executable output,
  GPU-computed grouped experts, token-identical to CPU (posted to #339).
- **Vulkan (§7):** #418 still open upstream. Our datapoint is filed (#523). If it
  merges, `COLI_VULKAN=1 DRAFT=0` becomes the recommended config on this machine
  (0.40 tok/s, +20% over HIP) and `coli-play.sh` should gain a `vk` mode. Note
  `COLI_VK_SHADERS` must be the **full path to `qmatmul.spv`**, not a directory.
- **Worth proposing upstream (§7.4):** gate MTP on whether expert I/O dominates
  rather than on `g_cuda_enabled`. Acceptance is fine (33%) but speculation costs
  ~35–40% on *both* backends here; a storage-bound predicate would cover Vulkan
  without a special case and would help CPU-only users on slow disks too.
- **Highest-value next build:** Lever B micro-benchmark (§3) — confirm the ~3×
  CPU-vs-PCIe-GPU per-expert, then prototype GPU compute for LRU experts.
- **Finish the churn track:** line-novelty detector (§2) → then the acceptance
  test (a fresh game spec, generated with zero human lines).
- **Cheap parallel win:** run the eval harness for a mixed-int3 quality sweep
  (§4.4) — our hardware can now afford runs upstream could not; even a null
  result is a contribution the project asked for.

Model (~363 GB int4) removed locally to reclaim disk; re-download via
`coli convert` or the Hugging Face int4 container when work resumes.
