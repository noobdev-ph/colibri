#!/usr/bin/env python3
"""Final analysis: two independent clean batteries, per-arm medians + I/O accounting.

Throughput from decode_s (RESEARCH 7.6). Bandwidth from per-process read_bytes
over WALL time -- read_bytes covers model load + prefill + decode, so dividing by
decode_s alone (as the live log line does) overstates it.
"""
import csv, pathlib, statistics as st, sys

B = pathlib.Path("/home/emcielo/Projects/colibri-dev/bench/clean-2026-08-01")
NGEN = 40
DRIVE_QD1_21MB = 3417.0     # measured, prober.c, colibri's own access pattern
DRIVE_QD8_21MB = 3911.0

ARM_LABEL = {
    "base":   "defaults (buffered)",
    "uring":  "URING=1 (buffered)",
    "direct": "DIRECT=1",
    "uringd": "URING=1 DIRECT=1",
}


def load(p):
    if not p.exists():
        return []
    return [r for r in csv.DictReader(p.open()) if r.get("tok_s") not in ("NA", "", None)]


def num(r, k, d=None):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


def wall_s(r):
    """model load + prefill + decode ~ the window read_bytes accumulated over"""
    return sum(filter(None, (num(r, "model_load_s", 0), num(r, "prefill_s", 0), num(r, "decode_s", 0))))


def summarize(rows, title):
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
    if not arms:
        print(f"\n{title}: no data yet")
        return {}
    print(f"\n{'='*92}\n{title}   (n={len(rows)} runs)\n{'='*92}")
    print(f"{'arm':22s} {'n':>2s} {'tok/s':>7s} {'vs base':>8s} {'GB read':>8s} {'MB/s':>7s} {'%drive':>7s} {'wait':>7s}")
    out = {}
    base = None
    for a in ("base", "uring", "direct", "uringd"):
        if a not in arms:
            continue
        v = arms[a]
        tps = [NGEN / num(r, "decode_s") for r in v if num(r, "decode_s")]
        gb = [num(r, "read_gb") for r in v if num(r, "read_gb")]
        bw = [num(r, "read_gb") * 1024 / wall_s(r) for r in v if num(r, "read_gb") and wall_s(r)]
        wt = [num(r, "edisk_wait_s") for r in v if num(r, "edisk_wait_s")]
        m = st.median(tps)
        if a == "base":
            base = m
        d = f"{100*(m-base)/base:+7.1f}%" if base else "      —"
        mb = st.median(bw) if bw else float("nan")
        print(f"{ARM_LABEL.get(a,a):22s} {len(v):2d} {m:7.4f} {d} "
              f"{st.median(gb) if gb else float('nan'):8.1f} {mb:7.0f} "
              f"{100*mb/DRIVE_QD1_21MB:6.0f}% {st.median(wt) if wt else float('nan'):6.1f}s")
        out[a] = {"tok_s": m, "runs": tps, "gb": st.median(gb) if gb else None, "mb": mb}
    return out


def main():
    A = load(B / "batteryA/results.csv")
    Bb = load(B / "batteryB/results.csv")
    ra = summarize(A, "BATTERY A")
    rb = summarize(Bb, "BATTERY B  (independent replication, identical frozen snapshot)")

    if ra and rb:
        print(f"\n{'='*92}\nREPLICATION CHECK — do the two batteries agree?\n{'='*92}")
        print(f"{'arm':22s} {'A':>8s} {'B':>8s} {'delta':>8s}")
        for a in ("base", "uring", "direct", "uringd"):
            if a in ra and a in rb:
                x, y = ra[a]["tok_s"], rb[a]["tok_s"]
                print(f"{ARM_LABEL.get(a,a):22s} {x:8.4f} {y:8.4f} {100*(y-x)/x:+7.1f}%")

        print(f"\n{'='*92}\nPOOLED (n=10/arm)\n{'='*92}")
        pool = {}
        for a in ("base", "uring", "direct", "uringd"):
            if a in ra and a in rb:
                pool[a] = ra[a]["runs"] + rb[a]["runs"]
        if "base" in pool:
            bm = st.median(pool["base"])
            for a, v in pool.items():
                m = st.median(v)
                lo, hi = min(v), max(v)
                print(f"{ARM_LABEL.get(a,a):22s} n={len(v):2d} median {m:.4f}  "
                      f"range {lo:.3f}-{hi:.3f}  vs base {100*(m-bm)/bm:+6.1f}%")
            # separation check on the headline claim
            if "uringd" in pool:
                b_hi, u_lo = max(pool["base"]), min(pool["uringd"])
                print(f"\n  headline: best base run {b_hi:.3f} vs worst uringd run {u_lo:.3f} -> "
                      f"{'NO OVERLAP' if u_lo > b_hi else 'DISTRIBUTIONS OVERLAP'}")

    print(f"\n  drive capability under this exact access pattern (prober.c):")
    print(f"    QD1 @21.2MB {DRIVE_QD1_21MB:.0f} MB/s   QD8 @21.2MB {DRIVE_QD8_21MB:.0f} MB/s")


if __name__ == "__main__":
    main()
