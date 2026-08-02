#!/bin/bash
# learn-battery.sh — cold start on code, then watch the expert cache learn.
#
# Every previous battery FROZE .coli_usage (restored one snapshot before each run)
# to hold placement constant. This one does the opposite on purpose: it deletes the
# history entirely, then lets it accumulate across runs, so the pin/VK tier are
# rebuilt from code routing alone.
#
# Why zeroing matters: one run of this prompt contributes ~93k expert selections
# (283 positions x ~330 experts/token). Against the existing 481k prose-dominated
# history the code signal stays buried; from zero it dominates within a few runs.
#
# Prediction under test: run 1 is the worst case (no history -> no useful pin, VK
# tier ranked on nothing). By run ~5-10 the top-320 should be code-derived and the
# vk bucket should climb back off the 6.2% floor measured with a prose cache,
# recovering some of the -22.7% code penalty.
#
# Config is fixed at the winning one: URING=1 DIRECT=1, VK tier 320, DRAFT=0.
# Usage: [RUNS=30] ./learn-battery.sh

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"
SH="$REPO/c/shaders/qmatmul.spv"
BIN="$REPO/c/colibri"
OUT="${COLI_OUT:-$(cd "$(dirname "$0")" && pwd)/learn-battery}"
RUNS="${RUNS:-30}"
NEXP="${NEXP:-320}"
NGEN="${NGEN:-40}"

read -r -d '' CODEBODY <<'EOF'
Here is the expert-streaming hot path from an inference engine:

static int expert_load(ExpertSlot *s, int layer, int eid) {
    off_t off = shard_offset(layer, eid);
    size_t n   = expert_bytes(layer);
    ssize_t r  = pread(s->fd, s->slab, n, off);
    if (r < 0) return -errno;
    if ((size_t)r != n) return -EIO;
    s->layer = layer; s->eid = eid;
    return 0;
}

It issues one blocking pread per expert, so the NVMe sees queue depth 1, and each
expert is ~21 MB. Rewrite it to batch up to 8 outstanding reads with io_uring,
preserving the exact function signature and error semantics (negative errno on
failure, short read is -EIO). Handle partial completions and requeue them. Then
explain what alignment constraints O_DIRECT imposes on s->slab, the offset, and
the length, and how you would coalesce two experts that are adjacent in the same
shard into a single request.
EOF
PROMPT="[gMASK]<sop><|user|>${CODEBODY}<|assistant|><think></think>"

mkdir -p "$OUT"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "run,tok_s,hit_total,pin,lru,vk,usage_selections,usage_bytes,pin_ram_experts,experts_resident,prompt_tok,prefill_s,decode_s,rss_gb,edisk_wait_s,read_gb,wall_s" > "$CSV"

log() { printf '\033[38;5;37m[learn] %s\033[0m\n' "$*"; }

USAGE="$M/.coli_usage"

# ---- COLD START: wipe the learned history -----------------------------------
if [ "${SKIP_WIPE:-0}" != "1" ]; then
    if [ -f "$USAGE" ]; then
        cp "$USAGE" "$OUT/.coli_usage.before-wipe" 2>/dev/null
        rm -f "$USAGE"
        log "COLD START: removed .coli_usage (backed up to $OUT/.coli_usage.before-wipe)"
    else
        log "COLD START: no .coli_usage present"
    fi
fi
log "  llama-server: $(systemctl --user is-active llama-server 2>/dev/null || echo inactive)"

wait_ram() {
    local i a
    for i in $(seq 1 30); do
        a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "$a" -ge 48 ] && return 0
        sleep 2
    done
}

one_run() {
    idx=$1
    f="$OUT/run$(printf '%02d' "$idx").txt"
    wait_ram
    # NOTE: deliberately NOT restoring a snapshot -- the history must accumulate.

    start=$(date +%s)
    env URING=1 DIRECT=1 PIPE_WORKERS=8 COLI_VULKAN=1 COLI_VK_EXPERTS="$NEXP" \
        COLI_VK_SHADERS="$SH" SNAP="$M" \
        PROMPT="$PROMPT" NGEN="$NGEN" TOPP=0.7 TEMP=0 DRAFT=0 \
        "$BIN" 320 > "$f" 2>&1 &
    cpid=$!
    ( while kill -0 "$cpid" 2>/dev/null; do
        v=$(awk '/^read_bytes:/{print $2}' "/proc/$cpid/io" 2>/dev/null)
        [ -n "$v" ] && echo "$v" > "$OUT/.io.$idx"
        sleep 1
      done ) &
    ios=$!
    wait "$cpid"
    kill "$ios" 2>/dev/null; wait "$ios" 2>/dev/null
    end=$(date +%s)
    rb=$(cat "$OUT/.io.$idx" 2>/dev/null); rm -f "$OUT/.io.$idx"; [ -n "$rb" ] || rb=0

    line=$(grep -a "decode .* tok/s" "$f" | tail -1)
    toks=$(echo "$line" | grep -oP '\(\K[0-9.]+(?= tok/s\))' | tail -1)
    dec=$(echo "$line"  | grep -oP 'decode [0-9]+ tokens in \K[0-9.]+' | tail -1)
    pre=$(echo "$line"  | grep -oP 'prefill [0-9]+ tokens in \K[0-9.]+' | tail -1)
    ptok=$(echo "$line" | grep -oP 'prefill \K[0-9]+' | tail -1)
    hit=$(echo "$line"  | grep -oP 'expert hit rate \K[0-9.]+' | tail -1)
    pin=$(echo "$line"  | grep -oP 'pin \K[0-9.]+' | tail -1)
    lru=$(echo "$line"  | grep -oP 'lru \K[0-9.]+' | tail -1)
    vk=$(echo "$line"   | grep -oP 'vk \K[0-9.]+' | tail -1)
    rss=$(echo "$line"  | grep -oP 'RSS \K[0-9.]+' | tail -1)
    res=$(grep -a '\[VK\] expert tier:' "$f" | grep -oP '\K[0-9]+(?= hot experts)' | tail -1)
    # history as the engine saw it AT STARTUP of this run
    sel=$(grep -a '\[USAGE\] expert history:' "$f" | grep -oP 'history: \K[0-9]+' | tail -1)
    pram=$(grep -a '\[PIN\] placement:' "$f" | grep -oP '\+ \K[0-9]+(?= RAM expert)' | tail -1)
    wt=$(grep -a '^PROFILE:' "$f" | tail -1 | grep -oP 'expert-disk [0-9.]+s service / \K[0-9.]+')
    ubytes=$(stat -c%s "$USAGE" 2>/dev/null || echo 0)
    rgb=$(echo "scale=2; $rb/1073741824" | bc -l 2>/dev/null || echo 0)

    echo "$idx,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},${vk:-NA},${sel:-0},${ubytes},${pram:-0},${res:-NA},${ptok:-NA},${pre:-NA},${dec:-NA},${rss:-NA},${wt:-NA},${rgb},$((end-start))" >> "$CSV"
    log "run $idx -> ${toks:-FAIL} tok/s | hit ${hit:-?}% (vk ${vk:-?}% pin ${pin:-?}% lru ${lru:-?}%) | hist ${sel:-0} sel | pinRAM ${pram:-0} | read ${rgb}GB | wall $((end-start))s"
}

log "cold-start learning curve: $RUNS runs of the code prompt, history accumulating"
for i in $(seq 1 "$RUNS"); do one_run "$i"; done
log "LEARN_BATTERY_COMPLETE -> $CSV"
