#!/bin/bash
# vk-battery.sh — interleaved A/B battery for the Vulkan expert tier.
#   A = COLI_VK_EXPERTS=320 (PR default, our n=1 baseline)
#   B = COLI_VK_EXPERTS=640 (doubled)
# 10 runs each, interleaved A,B,A,B... so page-cache warmth and thermal drift
# hit both arms equally. Fresh process per run (steve-m's methodology).
#
# .coli_usage is frozen: the same snapshot is restored before every run, so the
# "top-N of history" expert selection is identical across all 20 runs.

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"
SH="$REPO/c/shaders/qmatmul.spv"
BIN="$REPO/c/colibri"
OUT="$(cd "$(dirname "$0")" && pwd)/vk-battery"
REPS=10
PROMPT='[gMASK]<sop><|user|>Explain in two sentences why the sky is blue.<|assistant|><think></think>'

mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "run,cfg,vk_experts,tok_s,hit_total,pin,lru,vk,vram_peak_mb,experts_resident,tier_gb,tier_load_s,model_load_s,prefill_s,decode_s,rss_gb" > "$CSV"

log() { printf '\033[38;5;37m[battery] %s\033[0m\n' "$*"; }

# ---- preflight -------------------------------------------------------------
log "preflight"
systemctl --user stop llama-server 2>/dev/null
pkill -f firefox 2>/dev/null
sleep 5
log "  llama-server: $(systemctl --user is-active llama-server 2>/dev/null || echo inactive) | firefox procs: $(pgrep -c firefox 2>/dev/null || echo 0)"

# freeze the learning cache
USAGE="$M/.coli_usage"
SNAP="$OUT/.coli_usage.frozen"
[ -f "$USAGE" ] && cp "$USAGE" "$SNAP" && log "  froze .coli_usage ($(stat -c%s "$SNAP") bytes)"

vram_mb() { rocm-smi --showmeminfo vram 2>/dev/null | grep -A2 'GPU\[0\]' | grep -oP 'Used Memory \(B\): \K[0-9]+' | head -1 | awk '{print int($1/1048576)}'; }

wait_ram() {  # engine sizes its RAM tier from MemAvailable at startup
    local i a                                  # MUST be local: `i` is the caller's run counter
    for i in $(seq 1 30); do
        a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "$a" -ge 48 ] && return 0
        sleep 2
    done
}

# ---- one run ---------------------------------------------------------------
one_run() {
    idx=$1; cfg=$2; nexp=$3
    f="$OUT/run$(printf '%02d' "$idx")_${cfg}.txt"
    [ -f "$SNAP" ] && cp "$SNAP" "$USAGE"       # identical expert history every run
    wait_ram

    # VRAM sampler for this run
    ( peak=0
      while :; do v=$(vram_mb); [ -n "$v" ] && [ "$v" -gt "$peak" ] && peak=$v
                  echo "$peak" > "$OUT/.peak.$idx"; sleep 4; done ) &
    sampler=$!

    start=$(date +%s)
    env COLI_VULKAN=1 COLI_VK_EXPERTS="$nexp" COLI_VK_SHADERS="$SH" SNAP="$M" \
        PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 \
        "$BIN" 320 > "$f" 2>&1
    end=$(date +%s)
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    peak=$(cat "$OUT/.peak.$idx" 2>/dev/null || echo 0); rm -f "$OUT/.peak.$idx"

    # ---- parse ----
    line=$(grep -a "decode .* tok/s" "$f" | tail -1)
    toks=$(echo "$line"   | grep -oP '\(\K[0-9.]+(?= tok/s\))' | tail -1)
    dec=$(echo "$line"    | grep -oP 'decode [0-9]+ tokens in \K[0-9.]+' | tail -1)
    pre=$(echo "$line"    | grep -oP 'prefill [0-9]+ tokens in \K[0-9.]+' | tail -1)
    hit=$(echo "$line"    | grep -oP 'expert hit rate \K[0-9.]+' | tail -1)
    pin=$(echo "$line"    | grep -oP 'pin \K[0-9.]+' | tail -1)
    lru=$(echo "$line"    | grep -oP 'lru \K[0-9.]+' | tail -1)
    vk=$(echo "$line"     | grep -oP 'vk \K[0-9.]+' | tail -1)
    rss=$(echo "$line"    | grep -oP 'RSS \K[0-9.]+' | tail -1)
    tier=$(grep -a '\[VK\] expert tier:' "$f" | tail -1)
    res=$(echo "$tier"    | grep -oP '\K[0-9]+(?= hot experts)' | tail -1)
    tgb=$(echo "$tier"    | grep -oP '\(\K[0-9.]+(?= GB VRAM)' | tail -1)
    tls=$(echo "$tier"    | grep -oP 'GB VRAM, \K[0-9.]+' | tail -1)
    mls=$(grep -a 'loaded in' "$f" | grep -oP 'loaded in \K[0-9.]+' | tail -1)

    echo "$idx,$cfg,$nexp,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},${vk:-NA},$peak,${res:-NA},${tgb:-NA},${tls:-NA},${mls:-NA},${pre:-NA},${dec:-NA},${rss:-NA}" >> "$CSV"
    log "run $idx [$cfg/$nexp] -> ${toks:-FAIL} tok/s | hit ${hit:-?}% (vk ${vk:-?}%) | VRAM peak ${peak} MB | resident ${res:-?} (${tgb:-?} GB) | wall $((end-start))s"
}

# ---- interleaved battery ---------------------------------------------------
log "starting $((REPS*2)) runs (10x A=320, 10x B=640), interleaved"
i=0
for r in $(seq 1 "$REPS"); do
    i=$((i+1)); one_run "$i" A 320
    i=$((i+1)); one_run "$i" B 640
done

log "BATTERY_COMPLETE  -> $CSV"
