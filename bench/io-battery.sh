#!/bin/bash
# io-battery.sh — screen the LOSSLESS expert-I/O knobs on a disk-bound host.
#
# Our profile is overwhelmingly storage-bound (expert-disk ~149s service /
# ~57s wait vs expert-matmul ~19s per run), yet every prior battery ran with
# the whole I/O subsystem at its defaults: PIPE=0, URING=0, DIRECT=0, PILOT=0,
# PILOT_WORKERS=1. Upstream #441 names PILOT_WORKERS=1 as an NVMe queue-depth-1
# bug; PIPE/URING/DIRECT are documented byte-identical and PILOT_REAL as
# value-preserving, so none of these arms can change output quality.
#
# Arms (all: Vulkan tier 320, DRAFT=0, frozen .coli_usage, fresh process):
#   base    engine defaults               -> the 0.3524 tok/s reference
#   pipe    PIPE=1 PIPE_WORKERS=8         -> overlap disk with matmul
#   uring   URING=1 PIPE_WORKERS=8        -> io_uring batched reads (implies PIPE)
#   uringd  URING=1 DIRECT=1              -> + O_DIRECT (note: btrfs zstd may defeat it)
#   pilot   PIPE=1 PILOT=1 PILOT_REAL=1 PILOT_WORKERS=8   -> the #441 fix
#   all     uring + direct + pilot
#
# Usage: [REPS=3] [ARMS="base pipe uring"] ./io-battery.sh

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"
SH="$REPO/c/shaders/qmatmul.spv"
BIN="$REPO/c/colibri"
OUT="${COLI_OUT:-$(cd "$(dirname "$0")" && pwd)/io-battery}"
REPS="${REPS:-3}"
ARMS="${ARMS:-base pipe uring uringd pilot all}"
NEXP="${NEXP:-320}"
PROMPT='[gMASK]<sop><|user|>Explain in two sentences why the sky is blue.<|assistant|><think></think>'

mkdir -p "$OUT"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "run,arm,vk_experts,tok_s,hit_total,pin,lru,vk,vram_peak_mb,experts_resident,tier_gb,tier_load_s,model_load_s,prefill_s,decode_s,rss_gb,edisk_service_s,edisk_wait_s,ematmul_s" > "$CSV"

log() { printf '\033[38;5;37m[io] %s\033[0m\n' "$*"; }

# ---- arm -> env --------------------------------------------------------------
arm_env() {
    case "$1" in
        base)   echo "" ;;
        pipe)   echo "PIPE=1 PIPE_WORKERS=8" ;;
        uring)  echo "URING=1 PIPE_WORKERS=8" ;;
        uringd) echo "URING=1 DIRECT=1 PIPE_WORKERS=8" ;;
        pilot)  echo "PIPE=1 PIPE_WORKERS=8 PILOT=1 PILOT_REAL=1 PILOT_WORKERS=8" ;;
        all)    echo "URING=1 DIRECT=1 PIPE_WORKERS=8 PILOT=1 PILOT_REAL=1 PILOT_WORKERS=8" ;;
        # --- round 2: isolate O_DIRECT, and hunt the matmul-contention headroom ---
        direct)  echo "DIRECT=1" ;;                                  # O_DIRECT alone, no PIPE/URING
        pipedir) echo "PIPE=1 DIRECT=1 PIPE_WORKERS=8" ;;            # pthread loaders + O_DIRECT
        uringd2) echo "URING=1 DIRECT=1 PIPE_WORKERS=2" ;;
        uringd4) echo "URING=1 DIRECT=1 PIPE_WORKERS=4" ;;
        uringd16) echo "URING=1 DIRECT=1 PIPE_WORKERS=16" ;;
        *)      echo "" ;;
    esac
}

# ---- preflight ---------------------------------------------------------------
log "preflight"
log "  llama-server: $(systemctl --user is-active llama-server 2>/dev/null || echo inactive)"
USAGE="$M/.coli_usage"
SNAP="$OUT/.coli_usage.frozen"
if [ ! -f "$SNAP" ]; then
    [ -f "$USAGE" ] && cp "$USAGE" "$SNAP" && log "  froze .coli_usage ($(stat -c%s "$SNAP") bytes)"
else
    log "  reusing frozen .coli_usage ($(stat -c%s "$SNAP") bytes)"
fi

vram_mb() { rocm-smi --showmeminfo vram 2>/dev/null | grep -A2 'GPU\[0\]' | grep -oP 'Used Memory \(B\): \K[0-9]+' | head -1 | awk '{print int($1/1048576)}'; }

wait_ram() {
    local i a
    for i in $(seq 1 30); do
        a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "$a" -ge 48 ] && return 0
        sleep 2
    done
}

# ---- one run -----------------------------------------------------------------
one_run() {
    idx=$1; arm=$2
    f="$OUT/run$(printf '%02d' "$idx")_${arm}.txt"
    [ -f "$SNAP" ] && cp "$SNAP" "$USAGE"
    wait_ram

    ( peak=0
      while :; do v=$(vram_mb); [ -n "$v" ] && [ "$v" -gt "$peak" ] && peak=$v
                  echo "$peak" > "$OUT/.peak.$idx"; sleep 4; done ) &
    sampler=$!

    start=$(date +%s)
    # shellcheck disable=SC2046
    env $(arm_env "$arm") COLI_VULKAN=1 COLI_VK_EXPERTS="$NEXP" COLI_VK_SHADERS="$SH" SNAP="$M" \
        PROMPT="$PROMPT" NGEN=40 TOPP=0.7 TEMP=0 DRAFT=0 \
        "$BIN" 320 > "$f" 2>&1
    end=$(date +%s)
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    peak=$(cat "$OUT/.peak.$idx" 2>/dev/null || echo 0); rm -f "$OUT/.peak.$idx"

    line=$(grep -a "decode .* tok/s" "$f" | tail -1)
    toks=$(echo "$line" | grep -oP '\(\K[0-9.]+(?= tok/s\))' | tail -1)
    dec=$(echo "$line"  | grep -oP 'decode [0-9]+ tokens in \K[0-9.]+' | tail -1)
    pre=$(echo "$line"  | grep -oP 'prefill [0-9]+ tokens in \K[0-9.]+' | tail -1)
    hit=$(echo "$line"  | grep -oP 'expert hit rate \K[0-9.]+' | tail -1)
    pin=$(echo "$line"  | grep -oP 'pin \K[0-9.]+' | tail -1)
    lru=$(echo "$line"  | grep -oP 'lru \K[0-9.]+' | tail -1)
    vk=$(echo "$line"   | grep -oP 'vk \K[0-9.]+' | tail -1)
    rss=$(echo "$line"  | grep -oP 'RSS \K[0-9.]+' | tail -1)
    tier=$(grep -a '\[VK\] expert tier:' "$f" | tail -1)
    res=$(echo "$tier"  | grep -oP '\K[0-9]+(?= hot experts)' | tail -1)
    tgb=$(echo "$tier"  | grep -oP '\(\K[0-9.]+(?= GB VRAM)' | tail -1)
    tls=$(echo "$tier"  | grep -oP 'GB VRAM, \K[0-9.]+' | tail -1)
    mls=$(grep -a 'loaded in' "$f" | grep -oP 'loaded in \K[0-9.]+' | tail -1)
    # decode-phase PROFILE line (the second one; prefill prints first)
    prof=$(grep -a '^PROFILE:' "$f" | tail -1)
    esvc=$(echo "$prof" | grep -oP 'expert-disk \K[0-9.]+' | tail -1)
    ewait=$(echo "$prof"| grep -oP 'expert-disk [0-9.]+s service / \K[0-9.]+' | tail -1)
    emm=$(echo "$prof"  | grep -oP 'expert-matmul \K[0-9.]+' | tail -1)

    echo "$idx,$arm,$NEXP,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},${vk:-NA},$peak,${res:-NA},${tgb:-NA},${tls:-NA},${mls:-NA},${pre:-NA},${dec:-NA},${rss:-NA},${esvc:-NA},${ewait:-NA},${emm:-NA}" >> "$CSV"
    log "run $idx [$arm] -> ${toks:-FAIL} tok/s | hit ${hit:-?}% | disk ${esvc:-?}s/${ewait:-?}s wait | matmul ${emm:-?}s | wall $((end-start))s"
}

# ---- interleaved battery -----------------------------------------------------
set -- $ARMS
log "arms: $ARMS | reps: $REPS | $(( $# * REPS )) runs, interleaved"
i=0
for r in $(seq 1 "$REPS"); do
    for a in $ARMS; do
        i=$((i+1)); one_run "$i" "$a"
    done
done

log "IO_BATTERY_COMPLETE -> $CSV"
