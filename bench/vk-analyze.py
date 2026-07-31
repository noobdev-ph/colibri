#!/usr/bin/env python3
"""Aggregate the interleaved A/B Vulkan battery into a report."""
import csv, statistics as st, sys, pathlib

CSV = pathlib.Path(__file__).parent / "vk-battery" / "results.csv"
NUM = ("tok_s","hit_total","pin","lru","vk","vram_peak_mb","experts_resident",
       "tier_gb","tier_load_s","model_load_s","prefill_s","decode_s","rss_gb")

rows = [r for r in csv.DictReader(CSV.open()) if r["tok_s"] not in ("NA","")]
arms = {}
for r in rows:
    for k in NUM:
        try: r[k] = float(r[k])
        except (ValueError, TypeError): r[k] = None
    arms.setdefault(r["cfg"], []).append(r)

def col(rs, k): return [r[k] for r in rs if r[k] is not None]
def med(rs, k):
    v = col(rs, k); return st.median(v) if v else float("nan")
def iqr(rs, k):
    v = sorted(col(rs, k))
    if len(v) < 4: return (min(v), max(v)) if v else (0, 0)
    q = st.quantiles(v, n=4); return (q[0], q[2])

names = {"A": "320 (PR default)", "B": "640 (doubled)"}
print(f"\n{'='*78}\nVulkan expert-tier A/B — interleaved, .coli_usage frozen, fresh process/run\n{'='*78}")
for cfg in sorted(arms):
    rs = arms[cfg]
    lo, hi = iqr(rs, "tok_s")
    print(f"\n  {cfg} = COLI_VK_EXPERTS={names.get(cfg,cfg)}   n={len(rs)}")
    print(f"    tok/s          median {med(rs,'tok_s'):.3f}   IQR {lo:.3f}–{hi:.3f}   "
          f"min {min(col(rs,'tok_s')):.3f} max {max(col(rs,'tok_s')):.3f}")
    print(f"    expert hit     {med(rs,'hit_total'):.1f}%  = pin {med(rs,'pin'):.1f}"
          f" + lru {med(rs,'lru'):.1f} + vk {med(rs,'vk'):.1f}")
    print(f"    VRAM peak      {med(rs,'vram_peak_mb'):.0f} MB   tier {med(rs,'tier_gb'):.2f} GB"
          f" ({med(rs,'experts_resident'):.0f} experts, upload {med(rs,'tier_load_s'):.1f}s)")
    print(f"    prefill/decode {med(rs,'prefill_s'):.1f}s / {med(rs,'decode_s'):.1f}s"
          f"   model load {med(rs,'model_load_s'):.1f}s   RSS {med(rs,'rss_gb'):.1f} GB")

if "A" in arms and "B" in arms:
    a, b = arms["A"], arms["B"]
    da = med(b,"tok_s") - med(a,"tok_s")
    pct = 100*da/med(a,"tok_s")
    print(f"\n{'-'*78}\n  DELTA (B 640 vs A 320)")
    print(f"    tok/s        {med(a,'tok_s'):.3f} -> {med(b,'tok_s'):.3f}   "
          f"{da:+.3f} ({pct:+.1f}%)")
    print(f"    hit rate     {med(a,'hit_total'):.1f}% -> {med(b,'hit_total'):.1f}%   "
          f"{med(b,'hit_total')-med(a,'hit_total'):+.1f}pp")
    print(f"      vk         {med(a,'vk'):.1f}% -> {med(b,'vk'):.1f}%   {med(b,'vk')-med(a,'vk'):+.1f}pp")
    print(f"      pin        {med(a,'pin'):.1f}% -> {med(b,'pin'):.1f}%   {med(b,'pin')-med(a,'pin'):+.1f}pp")
    print(f"      lru        {med(a,'lru'):.1f}% -> {med(b,'lru'):.1f}%   {med(b,'lru')-med(a,'lru'):+.1f}pp")
    print(f"    VRAM         {med(a,'vram_peak_mb'):.0f} -> {med(b,'vram_peak_mb'):.0f} MB "
          f"({med(b,'vram_peak_mb')/med(a,'vram_peak_mb'):.2f}x)")
    print(f"    tier upload  {med(a,'tier_load_s'):.1f}s -> {med(b,'tier_load_s'):.1f}s")

    # is the difference distinguishable from run-to-run noise?
    sa, sb = col(a,"tok_s"), col(b,"tok_s")
    spread = max(max(sa)-min(sa), max(sb)-min(sb))
    print(f"\n    within-arm spread {spread:.3f} tok/s vs between-arm delta {abs(da):.3f} tok/s")
    print(f"    -> {'DISTINGUISHABLE' if abs(da) > spread else 'WITHIN NOISE — treat as no change'}")
print()
