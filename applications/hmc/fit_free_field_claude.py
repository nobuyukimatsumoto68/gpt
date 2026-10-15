#!/usr/bin/env python3
#
# LO free-field speed alignment: scan / fit eta's of coupling-layer kernels (T1, R) to minimize Var[log omega^2].
# Model: free_field_speed_claude.py (validated by test_free_field_speed_claude.py). numpy only.
# Optimizer: hand-written Nelder-Mead (J. A. Nelder, R. Mead, Comput. J. 7, 308 (1965)).
# Usage: python3 fit_free_field_claude.py [--L 8] [--c1 -0.331] [--niters 1,2,4] [--out file]
#
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from free_field_speed_claude import free_field_model


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


lines = []
model = None


def log(s):
    print(s, flush=True)
    lines.append(s)


def evaluate(spec):
    v, ratio, w2 = model.objective(spec)  # uses global model (set in main)
    return v, ratio


def nelder_mead(f, x0, step=0.02, tol=1e-4, max_eval=80):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    nev = len(pts)
    while nev < max_eval:
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr); nev += 1
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe); nev += 1
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc); nev += 1
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i]); nev += 1
    i = int(np.argmin(vals))
    return pts[i], vals[i], nev



def main():
    global L, c1, niters, out, model, lines
    L = int(arg("--L", "8"))
    c1 = float(arg("--c1", "-0.331"))
    niters = [int(x) for x in arg("--niters", "1,2,4").split(",")]
    out = arg("--out", None)
    model = free_field_model(L, c1)
    lines = []

    t0 = time.time()
    v0, r0 = evaluate([])
    log(f"# L = {L}, c1 = {c1}; theta = 0: Var[log w2] = {v0:.4f}, w2max/w2min = {r0:.2f}")

    for n in niters:
        # 1D scans: T1 only, R only
        for kind in ("T1", "R"):
            for eta in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
                v, r = evaluate([(kind, eta)] * n)
                log(f"scan n={n} {kind:2s} eta={eta:.3f}: Var[log w2] = {v:.4f}, ratio = {r:.2f}")
        # 2D fit: T1 + R, shared per kind
        f = lambda x: evaluate([("T1", x[0]), ("R", x[1])] * n)[0]
        x, fv, nev = nelder_mead(f, [0.05, 0.02])
        v, r = evaluate([("T1", x[0]), ("R", x[1])] * n)
        log(f"fit  n={n} T1+R: eta_T1 = {x[0]:.4f}, eta_R = {x[1]:.4f}: Var[log w2] = {v:.4f}, ratio = {r:.2f} ({nev} evals, {time.time() - t0:.0f} s)")

    if out is not None:
        open(out, "wt").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
