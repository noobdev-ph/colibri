#!/bin/bash
# gpu-battery-d0.sh — RE-RUN of the VRAM-matched Vulkan vs HIP comparison with
# speculation FORCED OFF on both arms (DRAFT=0).
#
# Why: the first battery was confounded. g_draft resolves as
#   g_draft = (m.has_mtp && (!g_cuda_enabled || cuda_mtp)) ? 3 : 0;   [#ifdef COLI_CUDA]
# so the HIP arms (COLI_CUDA=1) ran with drafting disabled while the Vulkan arms
# ran with MTP at 2.00 tokens/forward. Setting DRAFT=0 explicitly pins both arms
# to one position per forward, isolating the backend difference.
#
# HIP arms should reproduce their previous numbers (they were already g_draft=0
# via the guard) — that doubles as a drift check on the whole rig.

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"; SH="$REPO/c/shaders/qmatmul.spv"; BIN="$REPO/c/colibri"
OUT="$(cd "$(dirname "$0")" && pwd)/gpu-battery-d0"; REPS=10
PROMPT='[gMASK]<sop><|user|>Explain in two sentences why the sky is blue.<|assistant|><think></think>'

mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "run,arm,backend,budget,tok_s,hit_total,pin,lru,vk,vram_peak_mb,experts_resident,tier_gb,tier_load_s,model_load_s,prefill_s,decode_s,rss_gb,gpu_calls,fw_per_tok,mtp_acc" > "$CSV"

log() { printf '\033[38;5;37m[d0-battery] %s\033[0m\n' "$*"; }

log "preflight"
systemctl --user stop llama-server 2>/dev/null
pkill -f firefox 2>/dev/null
sleep 5

SNAP="$OUT/.coli_usage.frozen"; USAGE="$M/.coli_usage"
[ -f "$SNAP" ] || cp "$REPO/gpu-battery/.coli_usage.frozen" "$SNAP"
log "  frozen .coli_usage: $(stat -c%s "$SNAP") bytes (same snapshot as the first battery)"

vram_mb() { rocm-smi --showmeminfo vram 2>/dev/null | grep -A2 'GPU\[0\]' | grep -oP 'Used Memory \(B\): \K[0-9]+' | head -1 | awk '{print int($1/1048576)}'; }
wait_ram() { local i a; for i in $(seq 1 30); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 48 ] && return 0; sleep 2; done; }

one_run() {
    local idx=$1 arm=$2 backend=$3 budget=$4
    local f="$OUT/run$(printf '%02d' "$idx")_${arm}.txt"
    cp "$SNAP" "$USAGE"; wait_ram

    ( p=0; while :; do v=$(vram_mb); [ -n "$v" ] && [ "$v" -gt "$p" ] && p=$v; echo "$p" > "$OUT/.peak.$idx"; sleep 4; done ) &
    local sampler=$!

    if [ "$backend" = "vk" ]; then
        env COLI_VULKAN=1 COLI_VK_EXPERTS="$budget" COLI_VK_SHADERS="$SH" SNAP="$M" \
            DRAFT=0 PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" 320 > "$f" 2>&1
    else
        env HIP_VISIBLE_DEVICES=0 COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB="$budget" \
            CUDA_RELEASE_HOST=1 SNAP="$M" \
            DRAFT=0 PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" 320 > "$f" 2>&1
    fi
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    local peak; peak=$(cat "$OUT/.peak.$idx" 2>/dev/null || echo 0); rm -f "$OUT/.peak.$idx"

    local line toks dec pre hit pin lru vk rss res tgb tls mls calls fw acc
    line=$(grep -a "decode .* tok/s" "$f" | tail -1)
    toks=$(echo "$line" | grep -oP '\(\K[0-9.]+(?= tok/s\))' | tail -1)
    dec=$(echo "$line"  | grep -oP 'decode [0-9]+ tokens in \K[0-9.]+' | tail -1)
    pre=$(echo "$line"  | grep -oP 'prefill [0-9]+ tokens in \K[0-9.]+' | tail -1)
    hit=$(echo "$line"  | grep -oP 'expert hit rate \K[0-9.]+' | tail -1)
    pin=$(echo "$line"  | grep -oP 'pin \K[0-9.]+' | tail -1)
    lru=$(echo "$line"  | grep -oP 'lru \K[0-9.]+' | tail -1)
    vk=$(echo "$line"   | grep -oP 'vk \K[0-9.]+' | tail -1)
    rss=$(echo "$line"  | grep -oP 'RSS \K[0-9.]+' | tail -1)
    mls=$(grep -a 'loaded in' "$f" | grep -oP 'loaded in \K[0-9.]+' | tail -1)
    fw=$(grep -a 'speculation:' "$f" | grep -oP 'speculation: \K[0-9.]+' | tail -1)
    acc=$(grep -a 'MTP acceptance' "$f" | grep -oP 'MTP acceptance \K[0-9]+' | tail -1)

    if [ "$backend" = "vk" ]; then
        local tier; tier=$(grep -a '\[VK\] expert tier:' "$f" | tail -1)
        res=$(echo "$tier" | grep -oP '\K[0-9]+(?= hot experts)' | tail -1)
        tgb=$(echo "$tier" | grep -oP '\(\K[0-9.]+(?= GB VRAM)' | tail -1)
        tls=$(echo "$tier" | grep -oP 'GB VRAM, \K[0-9.]+' | tail -1); calls=""
    else
        local tier; tier=$(grep -a '\[CUDA\] hot expert tier:' "$f" | tail -1)
        res=$(echo "$tier" | grep -oP 'tier: \K[0-9]+' | tail -1)
        tgb=$(echo "$tier" | grep -oP 'VRAM \K[0-9.]+' | tail -1); tls=""
        calls=$(grep -a 'calls served from VRAM' "$f" | grep -oP '\| \K[0-9]+(?= calls served)' | tail -1)
    fi

    echo "$idx,$arm,$backend,$budget,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},${vk:-NA},$peak,${res:-NA},${tgb:-NA},${tls:-NA},${mls:-NA},${pre:-NA},${dec:-NA},${rss:-NA},${calls:-NA},${fw:-NA},${acc:-NA}" >> "$CSV"
    log "run $idx [$arm] -> ${toks:-FAIL} tok/s | hit ${hit:-?}% | fw/tok ${fw:-?} | VRAM ${peak} MB"
}

log "starting $((REPS*4)) runs with DRAFT=0 on BOTH backends"
i=0
for r in $(seq 1 "$REPS"); do
    i=$((i+1)); one_run "$i" VK6   vk  320
    i=$((i+1)); one_run "$i" HIP6  hip 6
    i=$((i+1)); one_run "$i" VK12  vk  640
    i=$((i+1)); one_run "$i" HIP12 hip 12
done
log "D0_BATTERY_COMPLETE -> $CSV"
