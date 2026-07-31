# bench — GPU backend research and benchmark harnesses

Research artifacts carried over from the `noobdev-ph/colibri` fork
(`~/Projects/glm5-2`). They are the measurement side of upstream issue
[#523](https://github.com/JustVugg/colibri/issues/523) and PRs #338 / #339.

`RESEARCH.md` is the writeup; §7 covers the Vulkan-vs-HIP evaluation
(130 controlled runs), §8 has the resume points.

## Layout

| path | what |
|---|---|
| `RESEARCH.md` | full findings, methodology, resume points |
| `full-battery.sh` | 3-phase VK/HIP A/B: DRAFT=0, DRAFT=3, tiny-LRU (130 runs) |
| `gpu-battery.sh`, `gpu-battery-d0.sh` | VK vs HIP at matched VRAM budgets |
| `vk-battery.sh` | Vulkan expert-tier size sweep (320 vs 640) |
| `mmap-battery.sh` | `COLI_MMAP=1` vs default pread+slab (20 runs) |
| `hip-default.sh` | HIP baseline at upstream defaults |
| `vk-analyze.py` | aggregates `vk-battery/results.csv` into a report |
| `*-battery/` | raw per-run logs, `results.csv`, frozen `.coli_usage` snapshots |
| `vk-*.txt`, `gs64-*.txt` | one-off session logs (ReBAR, warm-tier, coherence, gs64 codegen) |

## Running them here

Written for the fork's root; two lines were changed per script so they work
from `bench/` in this repo. **The recorded results came from the unpatched
versions** — only path resolution differs, no measurement logic was touched:

- `REPO=` now resolves to the repo root one level up, overridable via `COLI_REPO`
- `M=` (model dir) is overridable via `COLI_MODEL_DIR`
- `OUT=` writes results beside the scripts rather than at the repo root

The model is reached through a `model_gs64` symlink at the repo root pointing
into `~/Projects/glm5-2/` (git-excluded via `.git/info/exclude`). Build first:

```sh
make            # CPU engine
make VK=1       # + Vulkan backend and SPIR-V shaders (this repo tracks dev)
./bench/full-battery.sh
```

## Gotchas that cost real time

- **`COLI_VK_SHADERS` must be the full path to `qmatmul.spv`**, not a directory.
  (Upstream `809a847` later made the binary find shaders next to itself.)
- **Resizable BAR is mandatory and fails silently without it** — the tier lands
  in system RAM while still logging `320 hot experts resident (6.04 GB VRAM)`
  with the card at 82 MB. Worth 0.11 → 0.24 tok/s.
- **Set `DRAFT` explicitly on every arm.** `COLI_CUDA=1` trips the `g_draft`
  guard, so HIP arms silently run without speculation while Vulkan arms draft —
  this confound inverted the first 50-run battery's conclusion.
- **Freeze `.coli_usage`** to one snapshot restored before every run, or the
  learning cache drifts and changes what each arm holds.
- **Interleave arms**, never run them in blocks. Compute throughput from
  `decode_s`, not the printed `tok/s` (it rounds to 2 decimals).

## Headline results (RX 9070 XT, Ryzen 7 5700G, Gen3-capped NVMe)

- Vulkan is the fastest config measured: **0.40 tok/s**, +18.9% over HIP at
  ~12 GB and +23.7% at ~6 GB, no overlap between arms.
- MTP is an I/O amplifier: positions per emitted token = `4/(1+3a)`; at 33%
  acceptance that is 2.0. Costs ~35–40% on **both** backends, so `DRAFT=0` is
  the biggest single knob on storage-bound hardware.
- Default pread+slab beats `COLI_MMAP=1` by **+25.4%** (paired median, 9/10
  pairs, sign test p=0.021) with identical hit rate — fault-path and memory
  pressure, not caching. *This one is in issue #523 but not yet in RESEARCH.md.*
