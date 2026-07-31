#!/bin/bash
# gpu-battery.sh — VRAM-matched Vulkan vs HIP/ROCm A/B on v1.1.0 (commit 550df5c).
# ONE binary built with HIP=1 VK=1, so the only difference between arms is env.
#
#   VK6   COLI_VK_EXPERTS=320   -> 6.04 GB VRAM
#   HIP6  CUDA_EXPERT_GB=6      -> 5.99 GB VRAM
#   VK12  COLI_VK_EXPERTS=640   -> 12.08 GB VRAM
#   HIP12 CUDA_EXPERT_GB=12     -> ~12 GB VRAM
#
# 10 runs per arm, interleaved so cache warmth / thermal drift hit all arms equally.
# .coli_usage frozen to a fixed snapshot before every run => identical hot-expert
# ranking everywhere. Fresh process per run.
#
# NOTE on hit-rate buckets: Vulkan reports its VRAM tier as a separate `vk` bucket;
# HIP with CUDA_RELEASE_HOST=1 folds its VRAM prefix into `pin` (npin+=prefix_est).
# So TOTAL hit rate and tok/s are comparable across backends; the split is not.

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"
SH="$REPO/c/shaders/qmatmul.spv"
BIN="$REPO/c/colibri"
OUT="$(cd "$(dirname "$0")" && pwd)/gpu-battery"
REPS=10
PROMPT='[gMASK]<sop><|user|>Explain in two sentences why the sky is blue.<|assistant|><think></think>'

mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "run,arm,backend,budget,tok_s,hit_total,pin,lru,vk,vram_peak_mb,experts_resident,tier_gb,tier_load_s,model_load_s,prefill_s,decode_s,rss_gb,gpu_calls" > "$CSV"

log() { printf '\033[38;5;37m[gpu-battery] %s\033[0m\n' "$*"; }

log "preflight"
systemctl --user stop llama-server 2>/dev/null
pkill -f firefox 2>/dev/null
sleep 5
log "  llama-server: $(systemctl --user is-active llama-server 2>/dev/null || echo inactive) | firefox: $(pgrep -c firefox 2>/dev/null || echo 0)"

USAGE="$M/.coli_usage"
SNAP="$OUT/.coli_usage.frozen"
if [ ! -f "$SNAP" ]; then
    cp "$REPO/vk-battery/.coli_usage.frozen" "$SNAP" 2>/dev/null || cp "$USAGE" "$SNAP"
fi
log "  frozen .coli_usage: $(stat -c%s "$SNAP") bytes"

vram_mb() { rocm-smi --showmeminfo vram 2>/dev/null | grep -A2 'GPU\[0\]' | grep -oP 'Used Memory \(B\): \K[0-9]+' | head -1 | awk '{print int($1/1048576)}'; }

wait_ram() {
    local i a                       # MUST be local: `i` is the caller's run counter
    for i in $(seq 1 30); do
        a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "$a" -ge 48 ] && return 0
        sleep 2
    done
}

one_run() {
    local idx=$1 arm=$2 backend=$3 budget=$4
    local f="$OUT/run$(printf '%02d' "$idx")_${arm}.txt"
    cp "$SNAP" "$USAGE"
    wait_ram

    ( local_peak=0
      while :; do v=$(vram_mb); [ -n "$v" ] && [ "$v" -gt "$local_peak" ] && local_peak=$v
                  echo "$local_peak" > "$OUT/.peak.$idx"; sleep 4; done ) &
    local sampler=$!

    if [ "$backend" = "vk" ]; then
        env COLI_VULKAN=1 COLI_VK_EXPERTS="$budget" COLI_VK_SHADERS="$SH" SNAP="$M" \
            PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" 320 > "$f" 2>&1
    else
        env HIP_VISIBLE_DEVICES=0 COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB="$budget" \
            CUDA_RELEASE_HOST=1 SNAP="$M" \
            PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" 320 > "$f" 2>&1
    fi
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    local peak; peak=$(cat "$OUT/.peak.$idx" 2>/dev/null || echo 0); rm -f "$OUT/.peak.$idx"

    local line toks dec pre hit pin lru vk rss res tgb tls mls calls
    line=$(grep -a "decode .* tok/s" "$f" | tail -1)
    toks=$(echo "$line" | grep -oP '\(\K[0-9.]+(?= tok/s\))' | tail -1)
    dec=$(echo "$line"  | grep -oP 'decode [0-9]+ tokens in \K[0-9.]+' | tail -1)
    pre=$(echo "$line"  | grep -oP 'prefill [0-9]+ tokens in \K[0-9.]+' | tail -1)
    hit=$(echo "$line"  | grep -oP 'expert hit rate \K[0-9.]+' | tail -1)
    pin=$(echo "$line"  | grep -oP 'pin \K[0-9.]+' | tail -1)
    lru=$(echo "$line"  | grep -oP 'lru \K[0-9.]+' | tail -1)
    vk=$(echo "$line"   | grep -oP 'vk \K[0-9.]+' | tail -1)          # absent on HIP
    rss=$(echo "$line"  | grep -oP 'RSS \K[0-9.]+' | tail -1)
    mls=$(grep -a 'loaded in' "$f" | grep -oP 'loaded in \K[0-9.]+' | tail -1)

    if [ "$backend" = "vk" ]; then
        local tier; tier=$(grep -a '\[VK\] expert tier:' "$f" | tail -1)
        res=$(echo "$tier" | grep -oP '\K[0-9]+(?= hot experts)' | tail -1)
        tgb=$(echo "$tier" | grep -oP '\(\K[0-9.]+(?= GB VRAM)' | tail -1)
        tls=$(echo "$tier" | grep -oP 'GB VRAM, \K[0-9.]+' | tail -1)
        calls=""
    else
        local tier; tier=$(grep -a '\[CUDA\] hot expert tier:' "$f" | tail -1)
        res=$(echo "$tier" | grep -oP 'tier: \K[0-9]+' | tail -1)
        tgb=$(echo "$tier" | grep -oP 'VRAM \K[0-9.]+' | tail -1)
        tls=""
        calls=$(grep -a 'calls served from VRAM' "$f" | grep -oP '\| \K[0-9]+(?= calls served)' | tail -1)
    fi

    echo "$idx,$arm,$backend,$budget,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},${vk:-NA},$peak,${res:-NA},${tgb:-NA},${tls:-NA},${mls:-NA},${pre:-NA},${dec:-NA},${rss:-NA},${calls:-NA}" >> "$CSV"
    log "run $idx [$arm] -> ${toks:-FAIL} tok/s | hit ${hit:-?}% | VRAM ${peak} MB | resident ${res:-?} (${tgb:-?} GB)"
}

log "starting $((REPS*4)) runs: VK6, HIP6, VK12, HIP12 interleaved"
i=0
for r in $(seq 1 "$REPS"); do
    i=$((i+1)); one_run "$i" VK6   vk  320
    i=$((i+1)); one_run "$i" HIP6  hip 6
    i=$((i+1)); one_run "$i" VK12  vk  640
    i=$((i+1)); one_run "$i" HIP12 hip 12
done

log "BATTERY_COMPLETE -> $CSV"
