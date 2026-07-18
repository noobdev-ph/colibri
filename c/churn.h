/* churn.h — "churn until finished" sampling guards. All opt-in via env, zero
 * effect when unset. Designed for long single-artifact generations (code files)
 * where the failure modes are early EOS and periodic repetition loops.
 *
 *   MIN_TOKENS=N     stop tokens are masked out of the logits until N tokens
 *                    have been emitted.
 *   UNTIL=<string>   stop tokens stay masked until the decoded output contains
 *                    <string> (HARD semantics: if the marker never appears the
 *                    generation runs to its --ngen budget). Case-sensitive.
 *   NOLOOP=1         periodic-repetition guard: when the tail of the emitted
 *                    token stream becomes p-periodic (p in [2,NOLOOP_MAXP])
 *                    covering at least max(3 periods, NOLOOP_MIN tokens), the
 *                    token that would continue the pattern is banned from the
 *                    next pick, forcing divergence ("v0 divergence ban").
 *                    Tunables: NOLOOP_MAXP (default 32), NOLOOP_MIN (24).
 *
 * Churn features force DRAFT=0 (plain decode): accepted draft tokens bypass
 * the pick step, so masking cannot see them. The caller enforces this.
 * Detector constants were calibrated on five recorded GLM-5.2 int4 failure
 * loops (periods 8-12 tokens: "let spawnRateCurrentDelta = 0;" et al.). */
#ifndef COLIBRI_CHURN_H
#define COLIBRI_CHURN_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CHURN_RING 256                   /* token-id ring for the loop detector */
#define CHURN_TEXT 192                   /* rolling decoded-text tail for UNTIL */

static long  g_churn_min = 0;
static char  g_churn_until[96] = "";
static int   g_churn_until_seen = 0;
static int   g_churn_noloop = 0;
static int   g_churn_maxp = 32, g_churn_minrun = 24;
static int   g_churn_ring[CHURN_RING];
static int   g_churn_rn = 0;             /* tokens currently in ring (capped) */
static long  g_churn_fed = 0;            /* total tokens fed */
static int   g_churn_ban = -1;           /* pending one-shot ban for next pick */
static long  g_churn_trips = 0;          /* detector trigger count */
static char  g_churn_tail[CHURN_TEXT + 1] = "";

static int churn_active(void){
    return g_churn_min > 0 || g_churn_until[0] || g_churn_noloop;
}

static void churn_env_init(void){
    const char *s;
    if((s = getenv("MIN_TOKENS"))) g_churn_min = atol(s);
    if((s = getenv("UNTIL"))){ strncpy(g_churn_until, s, sizeof(g_churn_until)-1); }
    if((s = getenv("NOLOOP"))) g_churn_noloop = atoi(s);
    if((s = getenv("NOLOOP_MAXP"))) g_churn_maxp = atoi(s);
    if((s = getenv("NOLOOP_MIN")))  g_churn_minrun = atoi(s);
    if(g_churn_maxp < 2) g_churn_maxp = 2;
    if(g_churn_maxp > CHURN_RING/3) g_churn_maxp = CHURN_RING/3;
    if(churn_active())
        fprintf(stderr, "[CHURN] active: min_tokens=%ld until=%s noloop=%d (DRAFT forced to 0)\n",
                g_churn_min, g_churn_until[0] ? g_churn_until : "-", g_churn_noloop);
}

/* 1 when stop tokens may be sampled. */
static int churn_gate_open(long n_emit){
    if(g_churn_min > 0 && n_emit < g_churn_min) return 0;
    if(g_churn_until[0] && !g_churn_until_seen) return 0;
    return 1;
}

/* Feed decoded text (any chunking); sets the UNTIL flag once the marker has
 * fully appeared. Keeps a rolling tail so markers split across chunks match. */
static void churn_text(const char *piece){
    if(!g_churn_until[0] || g_churn_until_seen || !piece || !piece[0]) return;
    size_t tl = strlen(g_churn_tail), pl = strlen(piece);
    if(pl >= CHURN_TEXT){ memcpy(g_churn_tail, piece + pl - CHURN_TEXT, CHURN_TEXT); g_churn_tail[CHURN_TEXT] = 0; }
    else {
        if(tl + pl > CHURN_TEXT){
            size_t keep = CHURN_TEXT - pl;
            memmove(g_churn_tail, g_churn_tail + tl - keep, keep);
            tl = keep;
        }
        memcpy(g_churn_tail + tl, piece, pl + 1);
    }
    if(strstr(g_churn_tail, g_churn_until)){
        g_churn_until_seen = 1;
        fprintf(stderr, "[CHURN] UNTIL marker seen: stop tokens re-enabled\n");
    }
}

/* Pure detector (unit-tested): does ids[0..n) end in a p-periodic run covering
 * at least max(3*p, minrun) tokens for some p in [2,maxp]? Returns the period,
 * or 0. The smallest qualifying period wins (a 2-token loop inside a 10-token
 * window must not be reported as period 10). */
static int churn_detect(const int *ids, int n, int maxp, int minrun){
    for(int p = 2; p <= maxp; p++){
        int need = 3 * p > minrun ? 3 * p : minrun;
        if(need > n) continue;
        int run = 0;                      /* matching suffix length under period p */
        while(run < n - p && ids[n - 1 - run] == ids[n - 1 - run - p]) run++;
        if(run + p >= need) return p;     /* run counts matches beyond the first period */
    }
    return 0;
}

/* Feed an emitted token id. May arm a one-shot ban for the next pick. */
static void churn_feed(int tok){
    if(!g_churn_noloop) return;
    if(g_churn_rn == CHURN_RING){ memmove(g_churn_ring, g_churn_ring + 1, (CHURN_RING - 1) * sizeof(int)); g_churn_rn--; }
    g_churn_ring[g_churn_rn++] = tok; g_churn_fed++;
    int p = churn_detect(g_churn_ring, g_churn_rn, g_churn_maxp, g_churn_minrun);
    if(p > 0 && g_churn_ban < 0){
        g_churn_ban = g_churn_ring[g_churn_rn - p];   /* the pattern's next expected token */
        g_churn_trips++;
        fprintf(stderr, "[CHURN] periodic loop (period %d) at token %ld: banning continuation token %d (trip %ld)\n",
                p, g_churn_fed, g_churn_ban, g_churn_trips);
        g_churn_rn = 0;                   /* reset window so one loop = one trip */
    }
}

/* Mask logits before picking: closed gate masks the stop set; a pending
 * NOLOOP ban masks its token (one-shot). Works for greedy and sampling. */
static void churn_mask(float *lo, int V, const int *stops, int nstops, long n_emit){
    if(!churn_active()) return;
    if(!churn_gate_open(n_emit))
        for(int i = 0; i < nstops; i++)
            if(stops[i] >= 0 && stops[i] < V) lo[stops[i]] = -1e30f;
    if(g_churn_ban >= 0){
        if(g_churn_ban < V) lo[g_churn_ban] = -1e30f;
        g_churn_ban = -1;
    }
}

#endif
