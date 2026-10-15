#!/usr/bin/env python3
#
# Evaluate the speed-alignment loss R = tr M^2 / (tr M)^2, M = A h_sym A^dagger, on ONE interacting configuration
# for several kernels (eta = 0, masked transverse n = 2, 4 at eta = 0.05, poly N = 8). Also checks h_sym symmetry.
# Reported: R, the relative variance N R - 1 (N = all link-algebra d.o.f.), tr M / N (mean omega^2), probe spread.
# Usage: python3 eval_ml_loss_claude.py --load cfg [--beta 2.95] [--nprobe 4] [--op hessian|laplace]
#   --op laplace replaces h by -Delta (full covariant adjoint Laplacian) in M = A h A^dagger
#
import gpt as g
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_loss_claude import hessian_fd, speed_loss, kernel_identity, kernel_transverse, kernel_poly, ip
from poly_kernel_claude import COEFFS_N8

load = g.default.get("--load", None)
beta = g.default.get_float("--beta", 2.95)
nprobe = g.default.get_int("--nprobe", 4)
U = g.load(load)
grid = U[0].grid
N = 8 * 4 * grid.gsites
S = g.qcd.gauge.action.iwasaki(beta)
op = g.default.get("--op", "hessian")
H = hessian_fd(S, 1e-5, symmetric=True)
if op == "laplace":
    from momentum_filter_claude import covariant_gaussian_filter as _cgf

    _lap = _cgf(U, 4.0, 1.0)  # only .laplace is used
    H = lambda U_, v: [g(-1.0 * _lap.laplace(x)) for x in v]
g.message(f"operator in M: {op}")
rng = g.random("eval_ml_loss_claude")

# symmetry check of h_sym
v, w = g.group.cartesian(U), g.group.cartesian(U)
rng.normal_element(v)
rng.normal_element(w)
a, b = ip(w, H(U, v)), ip(H(U, w), v)
g.message(f"h_sym symmetry: <w,hv> = {a:.8e}, <hw,v> = {b:.8e}, rel diff {abs(a - b) / abs(a):.2e}")


def probes(n, tag):
    r = g.random(tag)
    out = []
    for i in range(n):
        p = g.group.cartesian(U)
        r.normal_element(p)
        out.append(p)
    return out


pn, pd = probes(nprobe, "num"), probes(nprobe, "den")
family = g.default.get("--family", "masked")  # masked: eta=0, masked n=2,4, poly N=8 | poly: FA polynomials N=0,2,4,6,8
if family == "poly":
    from poly_coeffs_family_claude import COEFFS

    kernels = [(f"poly N={N}", kernel_poly(U, COEFFS[N]) if N > 0 else kernel_identity()) for N in [0, 2, 4, 6, 8]]
elif family == "polyfull":  # full improved Laplacian (longitudinal hops), refitted coefficients
    from poly_coeffs_full_claude import COEFFS as COEFFS_FULL

    kernels = [(f"polyfull N={N}", kernel_poly(U, COEFFS_FULL[N], longitudinal=True) if N > 0 else kernel_identity()) for N in [0, 2, 4, 6, 8]]
else:
    kernels = [
        ("eta=0", kernel_identity()),
        ("masked n=2 eta=0.05", kernel_transverse(U, 0.05, 2)),
        ("masked n=4 eta=0.05", kernel_transverse(U, 0.05, 4)),
        ("poly N=8", kernel_poly(U, COEFFS_N8)),
    ]
for name, k in kernels:
    t0 = time.time()
    L = speed_loss(U, k, H)
    R, trM, trM2 = L.estimate(pn, pd)
    per = [ip(p, L.M(p)) for p in pd]
    g.message(
        f"RESULT {name:22s}: R = {R:.6e}, N R - 1 = {N * R - 1:.4f}, mean omega^2 = trM/N = {trM / N:.4f} "
        f"(probe spread of trM {np.std(per) / abs(np.mean(per)):.2e}), {time.time() - t0:.1f} s"
    )

# ---- shell-resolved mean frequencies: omega_j^2 = <G_j xi, M G_j xi> / |G_j xi|^2 (covariant Gaussian filters) ----
from momentum_filter_claude import covariant_gaussian_filter, shell_window

Lx = grid.fdimensions[0]
windows = [(j, *shell_window(2 * np.pi * j / Lx, 2 * np.pi / Lx)) for j in sorted(set([1, Lx // 4, Lx // 2]))]
extra = [("lam0=8", 8.0, 2.0), ("lam0=12", 12.0, 2.0)]  # high-lambda windows (bulk / UV)
filters = [(f"j={j} lam0={l0:.2f}", covariant_gaussian_filter(U, l0, sl)) for j, l0, sl in windows] + [
    (n, covariant_gaussian_filter(U, l0, sl)) for n, l0, sl in extra
]
xs = probes(g.default.get_int("--nshell", 2), "shell")
Gx = {name: [G(x) for x in xs] for name, G in filters}
for name, k in kernels:
    L = speed_loss(U, k, H)
    w2 = []
    for fname, G in filters:
        vals = [ip(p, L.M(p)) / ip(p, p) for p in Gx[fname]]
        w2.append(np.mean(vals))
    w2 = np.array(w2)
    g.message(
        f"SHELL {name:22s}: " + ", ".join(f"{fn}: {x:.3f}" for (fn, _), x in zip(filters, w2))
        + f" | max/min {w2.max() / w2.min():.2f}, Var log {np.var(np.log(w2)):.4f}"
    )
