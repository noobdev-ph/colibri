/* Dependency-free tests for churn.h: the periodic-loop detector, the UNTIL
 * matcher (incl. markers split across chunks), and the EOS gate + mask.
 * The loop fixtures replicate the structure of five RECORDED failures from
 * GLM-5.2 int4 greedy generation (periods 8-12, e.g. the
 * "let spawnRateCurrentDelta = 0;" spiral of 2026-07-18). */
#include <assert.h>
#include "../churn.h"

static void fill_pattern(int *ids, int n, const int *pat, int p){
    for(int i = 0; i < n; i++) ids[i] = pat[i % p];
}

int main(void){
    int ids[256];

    /* --- detector: the recorded spirals (period ~10, many repeats) --- */
    const int spiral10[10] = {701, 3204, 88, 1502, 9, 15, 2201, 6, 44, 199};
    fill_pattern(ids, 60, spiral10, 10);
    assert(churn_detect(ids, 60, 32, 24) == 10);

    /* period 8 ("= 0);\nlet lastDeltaUpdate" class) */
    const int spiral8[8] = {5, 6, 7, 8, 9, 10, 11, 12};
    fill_pattern(ids, 40, spiral8, 8);
    assert(churn_detect(ids, 40, 32, 24) == 8);

    /* tight 2-token flip-flop must be found as period 2, not a multiple */
    const int flip[2] = {42, 43};
    fill_pattern(ids, 30, flip, 2);
    assert(churn_detect(ids, 30, 32, 24) == 2);

    /* healthy prefix, loop only at the tail: still detected */
    for(int i = 0; i < 100; i++) ids[i] = 1000 + i * 7;   /* non-repeating */
    fill_pattern(ids + 100, 40, spiral10, 10);
    assert(churn_detect(ids, 140, 32, 24) == 10);

    /* --- no false positives on healthy streams --- */
    for(int i = 0; i < 200; i++) ids[i] = (i * i * 2654435761u) % 9973;  /* pseudo-random */
    assert(churn_detect(ids, 200, 32, 24) == 0);
    /* short repetition below threshold (2 periods only) is NOT a loop:
     * legitimate code repeats small n-grams (e.g. "ctx.beginPath();") */
    fill_pattern(ids, 20, spiral10, 10);
    ids[0] = -1;                                   /* break any accidental 3rd period */
    assert(churn_detect(ids, 20, 32, 24) == 0);

    /* --- UNTIL matcher: whole, split, and never --- */
    strcpy(g_churn_until, "</html>"); g_churn_until_seen = 0; g_churn_tail[0] = 0;
    churn_text("<body>ok</body>");
    assert(!g_churn_until_seen);
    churn_text("</ht"); churn_text("ml>");         /* marker split across chunks */
    assert(g_churn_until_seen);

    /* --- gate: MIN_TOKENS and UNTIL combine (both must clear) --- */
    g_churn_min = 100; g_churn_until_seen = 0;
    assert(!churn_gate_open(50));                  /* below min, marker unseen */
    assert(!churn_gate_open(150));                 /* above min, marker unseen */
    g_churn_until_seen = 1;
    assert(!churn_gate_open(50));                  /* marker seen, below min */
    assert(churn_gate_open(150));                  /* both clear */
    g_churn_until[0] = 0; g_churn_min = 0;

    /* --- mask: closed gate suppresses stops; one-shot ban clears --- */
    float lo[16]; for(int i = 0; i < 16; i++) lo[i] = 1.0f;
    int stops[2] = {3, 7};
    g_churn_min = 10;                              /* activates churn, closes gate */
    churn_mask(lo, 16, stops, 2, 5);
    assert(lo[3] < -1e29f && lo[7] < -1e29f && lo[4] == 1.0f);
    g_churn_ban = 11;
    for(int i = 0; i < 16; i++) lo[i] = 1.0f;
    churn_mask(lo, 16, stops, 2, 50);              /* gate open now */
    assert(lo[3] == 1.0f && lo[11] < -1e29f && g_churn_ban == -1);
    g_churn_min = 0;

    /* --- feed end-to-end: spiral in, ban armed exactly once per trip --- */
    g_churn_noloop = 1; g_churn_rn = 0; g_churn_ban = -1; g_churn_trips = 0;
    for(int r = 0; r < 6; r++)
        for(int i = 0; i < 10 && g_churn_ban < 0; i++) churn_feed(spiral10[i]);
    assert(g_churn_trips == 1 && g_churn_ban == spiral10[0]);   /* next expected after 3 periods */

    printf("churn tests: ok\n");
    return 0;
}
