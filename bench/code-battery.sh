#!/bin/bash
# code-battery.sh — does a CODING prompt behave differently from prose?
#
# Everything measured so far used one 17-token prose prompt ("why the sky is
# blue"). Decode cost here is dominated by which experts must be fetched from
# disk, and the MoE router picks experts from the hidden state -- so a different
# domain can land on a different expert set than the .coli_usage-derived pin/tier
# holds. Two questions:
#
#   1. routing locality -- does hit rate (and therefore tok/s) move on code?
#   2. does DRAFT=0 still win?  MTP costs 4/(1+3a) positions per emitted token;
#      at the prose acceptance a=0.33 that is 2.0x the I/O and DRAFT=0 wins.
#      Code is more predictable, so if a rises the arithmetic can invert.
#
# Arms (all URING=1 DIRECT=1, VK tier 320, frozen .coli_usage, fresh process):
#   proseD0  prose prompt, DRAFT=0   <- control, ties back to the whole corpus
#   codeD0   code  prompt, DRAFT=0
#   codeD3   code  prompt, DRAFT=3   <- the acceptance question
#
# Usage: [REPS=3] [ARMS="proseD0 codeD0 codeD3"] ./code-battery.sh

REPO="${COLI_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
M="${COLI_MODEL_DIR:-$REPO/model_gs64}"
SH="$REPO/c/shaders/qmatmul.spv"
BIN="$REPO/c/colibri"
OUT="${COLI_OUT:-$(cd "$(dirname "$0")" && pwd)/code-battery}"
REPS="${REPS:-3}"
ARMS="${ARMS:-proseD0 codeD0 codeD3}"
NEXP="${NEXP:-320}"
NGEN="${NGEN:-40}"

PROSE='[gMASK]<sop><|user|>Explain in two sentences why the sky is blue.<|assistant|><think></think>'

# A genuinely complex, context-bearing coding request: real code in, a refactor
# with constraints, plus a reasoning sub-question. ~180 tokens of prompt.
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
CODE="[gMASK]<sop><|user|>${CODEBODY}<|assistant|><think></think>"

mkdir -p "$OUT"
CSV="$OUT/results.csv"
[ -f "$CSV" ] || echo "run,arm,prompt,draft,tok_s,hit_total,pin,lru,vk,experts_resident,model_load_s,prompt_tok,prefill_s,decode_s,rss_gb,edisk_wait_s,mtp_acc,fw_per_tok,read_gb" > "$CSV"

log() { printf '\033[38;5;37m[code] %s\033[0m\n' "$*"; }

arm_prompt() { case "$1" in prose*) printf '%s' "$PROSE";; code*) printf '%s' "$CODE";; esac; }
arm_draft()  { case "$1" in *D3) echo 3;; *) echo 0;; esac; }
arm_kind()   { case "$1" in prose*) echo prose;; *) echo code;; esac; }

log "preflight"
log "  llama-server: $(systemctl --user is-active llama-server 2>/dev/null || echo inactive)"
USAGE="$M/.coli_usage"
SNAP="$OUT/.coli_usage.frozen"
if [ ! -f "$SNAP" ]; then
    [ -f "$USAGE" ] && cp "$USAGE" "$SNAP" && log "  froze .coli_usage ($(stat -c%s "$SNAP") bytes)"
else
    log "  reusing frozen .coli_usage ($(stat -c%s "$SNAP") bytes)"
fi

wait_ram() {
    local i a
    for i in $(seq 1 30); do
        a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "$a" -ge 48 ] && return 0
        sleep 2
    done
}

one_run() {
    idx=$1; arm=$2
    f="$OUT/run$(printf '%02d' "$idx")_${arm}.txt"
    [ -f "$SNAP" ] && cp "$SNAP" "$USAGE"
    wait_ram
    d=$(arm_draft "$arm")

    start=$(date +%s)
    env URING=1 DIRECT=1 PIPE_WORKERS=8 COLI_VULKAN=1 COLI_VK_EXPERTS="$NEXP" \
        COLI_VK_SHADERS="$SH" SNAP="$M" \
        PROMPT="$(arm_prompt "$arm")" NGEN="$NGEN" TOPP=0.7 TEMP=0 DRAFT="$d" \
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
    mls=$(grep -a 'loaded in' "$f" | grep -oP 'loaded in \K[0-9.]+' | tail -1)
    acc=$(grep -a 'MTP acceptance' "$f" | grep -oP 'MTP acceptance \K[0-9]+' | tail -1)
    fw=$(grep -aoE '[0-9.]+ tok/fw' "$f" | tail -1 | grep -oE '^[0-9.]+')
    wait_s=$(grep -a '^PROFILE:' "$f" | tail -1 | grep -oP 'expert-disk [0-9.]+s service / \K[0-9.]+')
    rgb=$(echo "scale=2; $rb/1073741824" | bc -l 2>/dev/null || echo 0)

    echo "$idx,$arm,$(arm_kind "$arm"),$d,${toks:-NA},${hit:-NA},${pin:-NA},${lru:-NA},${vk:-NA},${res:-NA},${mls:-NA},${ptok:-NA},${pre:-NA},${dec:-NA},${rss:-NA},${wait_s:-NA},${acc:-NA},${fw:-NA},${rgb}" >> "$CSV"
    log "run $idx [$arm] -> ${toks:-FAIL} tok/s | hit ${hit:-?}% (vk ${vk:-?}%) | prompt ${ptok:-?}tok | prefill ${pre:-?}s | acc ${acc:-0}% fw ${fw:-?} | read ${rgb}GB | wall $((end-start))s"
}

set -- $ARMS
log "arms: $ARMS | reps: $REPS | $(( $# * REPS )) runs, interleaved | NGEN=$NGEN"
i=0
for r in $(seq 1 "$REPS"); do
    for a in $ARMS; do i=$((i+1)); one_run "$i" "$a"; done
done
log "CODE_BATTERY_COMPLETE -> $CSV"
