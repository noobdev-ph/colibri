#!/bin/bash
# mmap-battery.sh — answers @xxxajk on issue #523: "is the LRU using MMIO or not?"
#
# colibri's DEFAULT is NOT mmio: experts are pulled with one coalescing pread into
# an owned slab (colibri.c:159). COLI_MMAP=1 is the opt-in that makes experts VIEWS
# inside an mmap of the safetensors ("niente pread, niente slab, niente copia: la
# page cache del kernel E' la cache" - colibri.c:1352), which is the llama.cpp-style
# approach they report is slower.
#
# So this measures the cost of turning mmio ON, on the fastest config we have.
#   A = default (pread + slab)   B = COLI_MMAP=1 (mmap views, kernel page cache)
# Both: Vulkan 640 experts, DRAFT=0, 10 runs, interleaved, .coli_usage frozen.

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"; SH="$REPO/c/shaders/qmatmul.spv"; BIN="$REPO/c/colibri"
OUT="$(cd "$(dirname "$0")" && pwd)/mmap-battery"; REPS=10
PROMPT='[gMASK]<sop><|user|>Explain in two sentences why the sky is blue.<|assistant|><think></think>'

mkdir -p "$OUT"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "run,arm,mmap,tok_s,hit_total,pin,lru,vk,vram_peak_mb,experts_resident,tier_gb,tier_load_s,model_load_s,prefill_s,decode_s,rss_gb,ereq_tok" > "$CSV"

log() { printf '\033[38;5;37m[mmap] %s\033[0m\n' "$*"; }
systemctl --user stop llama-server 2>/dev/null
pkill -f firefox 2>/dev/null
sleep 5

SNAP="$OUT/.coli_usage.frozen"; USAGE="$M/.coli_usage"
[ -f "$SNAP" ] || cp "$REPO/full-battery/.coli_usage.frozen" "$SNAP"
log "frozen .coli_usage: $(stat -c%s "$SNAP") bytes"

vram_mb() { rocm-smi --showmeminfo vram 2>/dev/null | grep -A2 'GPU\[0\]' | grep -oP 'Used Memory \(B\): \K[0-9]+' | head -1 | awk '{print int($1/1048576)}'; }
wait_ram() { local i a; for i in $(seq 1 40); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 48 ] && return 0; sleep 2; done; }

one_run() {
    local idx=$1 arm=$2 mm=$3
    local f="$OUT/run$(printf '%02d' "$idx")_${arm}.txt"
    cp "$SNAP" "$USAGE"; wait_ram

    ( p=0; while :; do v=$(vram_mb); [ -n "$v" ] && [ "$v" -gt "$p" ] && p=$v; echo "$p" > "$OUT/.peak.$idx"; sleep 4; done ) &
    local sampler=$!

    if [ "$mm" = "1" ]; then
        env COLI_VULKAN=1 COLI_VK_EXPERTS=640 COLI_VK_SHADERS="$SH" SNAP="$M" \
            COLI_MMAP=1 DRAFT=0 PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" 320 > "$f" 2>&1
    else
        env COLI_VULKAN=1 COLI_VK_EXPERTS=640 COLI_VK_SHADERS="$SH" SNAP="$M" \
            DRAFT=0 PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 "$BIN" 320 > "$f" 2>&1
    fi
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    local peak; peak=$(cat "$OUT/.peak.$idx" 2>/dev/null || echo 0); rm -f "$OUT/.peak.$idx"

    local line toks dec pre hit pin lru vk rss res tgb tls mls ereq
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
    ereq=$(grep -a 'experts loaded/token' "$f" | grep -oP 'experts loaded/token: \K[0-9.]+' | tail -1)
    local tier; tier=$(grep -a '\[VK\] expert tier:' "$f" | tail -1)
    res=$(echo "$tier" | grep -oP '\K[0-9]+(?= hot experts)' | tail -1)
    tgb=$(echo "$tier" | grep -oP '\(\K[0-9.]+(?= GB VRAM)' | tail -1)
    tls=$(echo "$tier" | grep -oP 'GB VRAM, \K[0-9.]+' | tail -1)

    echo "$idx,$arm,$mm,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},${vk:-NA},$peak,${res:-NA},${tgb:-NA},${tls:-NA},${mls:-NA},${pre:-NA},${dec:-NA},${rss:-NA},${ereq:-NA}" >> "$CSV"
    log "run $idx [$arm] -> ${toks:-FAIL} tok/s | hit ${hit:-?}% | load ${mls:-?}s | tier ${tls:-?}s | RSS ${rss:-?} GB"
}

log "COLI_MMAP A/B: default (pread+slab) vs COLI_MMAP=1 (mmap views), $((REPS*2)) runs"
i=0
for r in $(seq 1 "$REPS"); do
    i=$((i+1)); one_run "$i" slab 0
    i=$((i+1)); one_run "$i" mmap 1
done
log "MMAP_BATTERY_COMPLETE -> $CSV"
