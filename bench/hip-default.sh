#!/bin/bash
# hip-default.sh — control arm: HIP at 12 GB with CUDA_RELEASE_HOST left at its DEFAULT.
# On a single-GPU box that default is 0, so the VRAM tier only mirrors already-pinned
# experts instead of being additive. Hypothesis: this reproduces a ROCm baseline slow
# enough to explain the "Vulkan ~35% faster than ROCm" figure in c/Makefile:260.
# Same methodology as gpu-battery.sh so the numbers drop straight into that table.

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"; BIN="$REPO/c/colibri"; OUT="$(cd "$(dirname "$0")" && pwd)/gpu-battery"
SNAP="$OUT/.coli_usage.frozen"; USAGE="$M/.coli_usage"
CSV="$OUT/results.csv"
PROMPT='[gMASK]<sop><|user|>Explain in two sentences why the sky is blue.<|assistant|><think></think>'

log() { printf '\033[38;5;37m[hip-default] %s\033[0m\n' "$*"; }
systemctl --user stop llama-server 2>/dev/null; pkill -f firefox 2>/dev/null; sleep 5

vram_mb() { rocm-smi --showmeminfo vram 2>/dev/null | grep -A2 'GPU\[0\]' | grep -oP 'Used Memory \(B\): \K[0-9]+' | head -1 | awk '{print int($1/1048576)}'; }
wait_ram() { local i a; for i in $(seq 1 30); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 48 ] && return 0; sleep 2; done; }

for r in $(seq 1 10); do
    idx=$((100+r)); f="$OUT/run${idx}_HIP12def.txt"
    cp "$SNAP" "$USAGE"; wait_ram
    ( p=0; while :; do v=$(vram_mb); [ -n "$v" ] && [ "$v" -gt "$p" ] && p=$v; echo "$p" > "$OUT/.peak.$idx"; sleep 4; done ) &
    s=$!
    # NOTE: no CUDA_RELEASE_HOST -> takes the single-GPU default of 0
    env HIP_VISIBLE_DEVICES=0 COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=12 SNAP="$M" \
        PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" 320 > "$f" 2>&1
    kill "$s" 2>/dev/null; wait "$s" 2>/dev/null
    peak=$(cat "$OUT/.peak.$idx" 2>/dev/null || echo 0); rm -f "$OUT/.peak.$idx"

    line=$(grep -a "decode .* tok/s" "$f" | tail -1)
    toks=$(echo "$line" | grep -oP '\(\K[0-9.]+(?= tok/s\))' | tail -1)
    dec=$(echo "$line"  | grep -oP 'decode [0-9]+ tokens in \K[0-9.]+' | tail -1)
    pre=$(echo "$line"  | grep -oP 'prefill [0-9]+ tokens in \K[0-9.]+' | tail -1)
    hit=$(echo "$line"  | grep -oP 'expert hit rate \K[0-9.]+' | tail -1)
    pin=$(echo "$line"  | grep -oP 'pin \K[0-9.]+' | tail -1)
    lru=$(echo "$line"  | grep -oP 'lru \K[0-9.]+' | tail -1)
    rss=$(echo "$line"  | grep -oP 'RSS \K[0-9.]+' | tail -1)
    mls=$(grep -a 'loaded in' "$f" | grep -oP 'loaded in \K[0-9.]+' | tail -1)
    tier=$(grep -a '\[CUDA\] hot expert tier:' "$f" | tail -1)
    res=$(echo "$tier" | grep -oP 'tier: \K[0-9]+' | tail -1)
    tgb=$(echo "$tier" | grep -oP 'VRAM \K[0-9.]+' | tail -1)
    calls=$(grep -a 'calls served from VRAM' "$f" | grep -oP '\| \K[0-9]+(?= calls served)' | tail -1)

    echo "$idx,HIP12def,hip,12,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},NA,$peak,${res:-NA},${tgb:-NA},NA,${mls:-NA},${pre:-NA},${dec:-NA},${rss:-NA},${calls:-NA}" >> "$CSV"
    log "run $r -> ${toks:-FAIL} tok/s | hit ${hit:-?}% | resident ${res:-?} (${tgb:-?} GB) | VRAM ${peak} MB"
done
log "HIPDEF_COMPLETE"
