#!/bin/bash
# full-battery.sh — all three phases requested on issue #523, run unattended.
# Tested at 550df5c (= head of PR #418). v1.1.1 verified not to touch the
# comparison surface (g_draft guard byte-identical, npin+=prefix_est unchanged,
# backend_cuda.cu no diff, backend_vulkan.c PR-only).
#
#   Phase 1  clean backend comparison   DRAFT=0 both, cap=320   4 arms x 10 = 40
#   Phase 2  what is MTP worth?         DRAFT=3 both, cap=320   2 arms x 10 = 20
#   Phase 3  tiny CPU cache             DRAFT=0 both, cap=16    2 arms x 10 = 20
#
# DRAFT is set EXPLICITLY on every arm: the CUDA guard at colibri.c only runs
# under `if(g_draft<0)`, so an explicit value bypasses it and both backends get
# the same speculation setting. That asymmetry is what confounded the first run.
#
# Controls (unchanged): arms interleaved within each phase, .coli_usage frozen to
# one snapshot restored before every run, fresh process per run, machine quiesced.

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"; SH="$REPO/c/shaders/qmatmul.spv"; BIN="$REPO/c/colibri"
OUT="$(cd "$(dirname "$0")" && pwd)/full-battery"; REPS=10
PROMPT='[gMASK]<sop><|user|>Explain in two sentences why the sky is blue.<|assistant|><think></think>'

mkdir -p "$OUT"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "phase,run,arm,backend,budget,draft,cap,tok_s,hit_total,pin,lru,vk,vram_peak_mb,experts_resident,tier_gb,model_load_s,prefill_s,decode_s,rss_gb,fw_per_tok,mtp_acc,ereq_tok" > "$CSV"

log() { printf '\033[38;5;37m[full] %s\033[0m\n' "$*"; }

log "preflight"
systemctl --user stop llama-server 2>/dev/null
pkill -f firefox 2>/dev/null
sleep 5
log "  llama-server: $(systemctl --user is-active llama-server 2>/dev/null || echo inactive) | firefox: $(pgrep -c firefox 2>/dev/null || echo 0)"

SNAP="$OUT/.coli_usage.frozen"; USAGE="$M/.coli_usage"
[ -f "$SNAP" ] || cp "$REPO/gpu-battery/.coli_usage.frozen" "$SNAP"
log "  frozen .coli_usage: $(stat -c%s "$SNAP") bytes (same snapshot as all prior batteries)"

vram_mb() { rocm-smi --showmeminfo vram 2>/dev/null | grep -A2 'GPU\[0\]' | grep -oP 'Used Memory \(B\): \K[0-9]+' | head -1 | awk '{print int($1/1048576)}'; }
wait_ram() { local i a; for i in $(seq 1 40); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 48 ] && return 0; sleep 2; done; }

one_run() {
    local phase=$1 idx=$2 arm=$3 backend=$4 budget=$5 draft=$6 cap=$7
    local f="$OUT/p${phase}_run$(printf '%02d' "$idx")_${arm}.txt"
    cp "$SNAP" "$USAGE"; wait_ram

    ( p=0; while :; do v=$(vram_mb); [ -n "$v" ] && [ "$v" -gt "$p" ] && p=$v; echo "$p" > "$OUT/.peak.$phase.$idx"; sleep 4; done ) &
    local sampler=$!

    if [ "$backend" = "vk" ]; then
        env COLI_VULKAN=1 COLI_VK_EXPERTS="$budget" COLI_VK_SHADERS="$SH" SNAP="$M" \
            DRAFT="$draft" PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" "$cap" > "$f" 2>&1
    else
        env HIP_VISIBLE_DEVICES=0 COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB="$budget" \
            CUDA_RELEASE_HOST=1 SNAP="$M" \
            DRAFT="$draft" PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" "$cap" > "$f" 2>&1
    fi
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    local peak; peak=$(cat "$OUT/.peak.$phase.$idx" 2>/dev/null || echo 0); rm -f "$OUT/.peak.$phase.$idx"

    local line toks dec pre hit pin lru vk rss res tgb mls fw acc ereq
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
    ereq=$(grep -a 'experts loaded/token' "$f" | grep -oP 'experts loaded/token: \K[0-9.]+' | tail -1)

    if [ "$backend" = "vk" ]; then
        local tier; tier=$(grep -a '\[VK\] expert tier:' "$f" | tail -1)
        res=$(echo "$tier" | grep -oP '\K[0-9]+(?= hot experts)' | tail -1)
        tgb=$(echo "$tier" | grep -oP '\(\K[0-9.]+(?= GB VRAM)' | tail -1)
    else
        local tier; tier=$(grep -a '\[CUDA\] hot expert tier:' "$f" | tail -1)
        res=$(echo "$tier" | grep -oP 'tier: \K[0-9]+' | tail -1)
        tgb=$(echo "$tier" | grep -oP 'VRAM \K[0-9.]+' | tail -1)
    fi

    echo "$phase,$idx,$arm,$backend,$budget,$draft,$cap,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},${vk:-NA},$peak,${res:-NA},${tgb:-NA},${mls:-NA},${pre:-NA},${dec:-NA},${rss:-NA},${fw:-NA},${acc:-NA},${ereq:-NA}" >> "$CSV"
    log "P$phase run $idx [$arm] -> ${toks:-FAIL} tok/s | hit ${hit:-?}% | fw/tok ${fw:-?} | acc ${acc:-?}% | VRAM ${peak} MB"
}

# ---------------- Phase 1: clean backend comparison, speculation OFF both ------
log "PHASE 1 — DRAFT=0 both backends, cap=320, 4 VRAM-matched arms x $REPS"
i=0
for r in $(seq 1 "$REPS"); do
    i=$((i+1)); one_run 1 "$i" VK6   vk  320 0 320
    i=$((i+1)); one_run 1 "$i" HIP6  hip 6   0 320
    i=$((i+1)); one_run 1 "$i" VK12  vk  640 0 320
    i=$((i+1)); one_run 1 "$i" HIP12 hip 12  0 320
done
log "PHASE1_COMPLETE"

# ---------------- Phase 2: what is MTP worth? speculation ON both -------------
log "PHASE 2 — DRAFT=3 both backends, cap=320, ~12 GB arms x $REPS"
i=0
for r in $(seq 1 "$REPS"); do
    i=$((i+1)); one_run 2 "$i" VK12mtp  vk  640 3 320
    i=$((i+1)); one_run 2 "$i" HIP12mtp hip 12  3 320
done
log "PHASE2_COMPLETE"

# ---------------- Phase 3: tiny CPU cache so RAM cannot mask disk -------------
log "PHASE 3 — DRAFT=0 both, cap=16 (tiny LRU), ~12 GB arms x $REPS"
i=0
for r in $(seq 1 "$REPS"); do
    i=$((i+1)); one_run 3 "$i" VK12tiny  vk  640 0 16
    i=$((i+1)); one_run 3 "$i" HIP12tiny hip 12  0 16
done
log "PHASE3_COMPLETE"

log "FULL_BATTERY_COMPLETE -> $CSV"
