#!/usr/bin/env python3
#
# Train the FA-polynomial kernel coefficients on ONE interacting configuration (NM, 2026-10-15).
#   kernel: A = prod_j (1 + alpha_j T + beta_j T^2), T = improved covariant Laplacian (--longitudinal 1: full)
#   parameters: complex roots z_j (Re z, log Im z) of each quadratic factor -> every factor > 0 (K strictly positive)
#               (alpha, beta) = (-2 s / c0, 1 / c0),  s = T0 - Re z,  c0 = s^2 + (Im z)^2,  T0 = 10 (full) / 7.5 (transverse)
#   loss: Var_j log omega_j^2,  omega_j^2 = <G_j xi, M G_j xi> / |G_j xi|^2,  M = A h_sym A^dagger  (Iwasaki Hessian),
#         G_j covariant Gaussian windows in -Delta[U] (lambda0 list --windows), fixed probes xi (deterministic objective)
#   optimizer: Nelder-Mead (J. A. Nelder, R. Mead, Comput. J. 7, 308 (1965)); start: analytic FA fit roots
# Design: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, entries (18)-(20).
# Usage: python3 train_poly_claude.py --load cfg --N 8 [--longitudinal 1] [--beta 2.95] [--nprobe 4]
#          [--windows 1,2,4,8,12,16] [--sigma 1.0] [--maxeval 300] [--out base]
#
import gpt as g
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_loss_claude import hessian_fd, speed_loss, kernel_poly, ip
from momentum_filter_claude import covariant_gaussian_filter
from fit_free_field_claude import nelder_mead

load = g.default.get("--load", None)
N = g.default.get_int("--N", 8)
longitudinal = bool(g.default.get_int("--longitudinal", 1))
beta = g.default.get_float("--beta", 2.95)
nprobe = g.default.get_int("--nprobe", 4)
windows = [float(x) for x in g.default.get("--windows", "1,2,4,8,12,16").split(",")]
sigma = g.default.get_float("--sigma", 1.0)
maxeval = g.default.get_int("--maxeval", 300)
out = g.default.get("--out", f"train_poly_N{N}_claude")
T0 = 10.0 if longitudinal else 7.5
roots_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "run_rmhmc_claude",
    "fa_complex_full_claude_roots.dat" if longitudinal else "fa_complex_claude_roots.dat",
)

U = g.load(load)
S = g.qcd.gauge.action.iwasaki(beta)
H = hessian_fd(S, 1e-5, symmetric=True)

# initial roots from the analytic FA fit, recovered from the coefficient modules (no data-file dependency; runs on Aurora):
#   beta = 1/c0, alpha = -2 s/c0  ->  s = -alpha/(2 beta), Im z = sqrt(c0 - s^2), Re z = T0 - s
# d = np.loadtxt(roots_file)
# z0 = np.array(sorted([complex(r[1], r[2]) for r in d if r[0] == N and r[3] == 1 and r[2] > 0], key=lambda c: c.real))
if longitudinal:
    from poly_coeffs_full_claude import COEFFS as _C0
else:
    from poly_coeffs_family_claude import COEFFS as _C0
_z = []
for a_, b_ in _C0[N]:
    c0_ = 1.0 / b_
    s_ = -a_ / (2.0 * b_)
    _z.append(complex(T0 - s_, np.sqrt(c0_ - s_ * s_)))
z0 = np.array(sorted(_z, key=lambda c: c.real))
x0 = np.concatenate([z0.real, np.log(z0.imag)])
n = len(z0)


def coeffs_of(x):
    z = x[:n] + 1j * np.exp(x[n:])
    co = []
    for zz in z:
        s = T0 - zz.real
        c0 = s * s + zz.imag**2
        co.append((-2 * s / c0, 1 / c0))
    return co, z


# fixed filtered probes per window
rng = g.random("train_poly_claude")
filt = [covariant_gaussian_filter(U, l0, sigma) for l0 in windows]
GX = []
for G in filt:
    lst = []
    for i in range(nprobe):
        xi = g.group.cartesian(U)
        rng.normal_element(xi)
        lst.append(G(xi))
    GX.append(lst)
norms = [[ip(p, p) for p in lst] for lst in GX]


def omega2(coeffs):
    L = speed_loss(U, kernel_poly(U, coeffs, longitudinal=longitudinal), H)
    return np.array([np.mean([ip(p, L.M(p)) / nn for p, nn in zip(lst, nl)]) for lst, nl in zip(GX, norms)])


hist = []


def loss(x):
    co, z = coeffs_of(np.asarray(x))
    w2 = omega2(co)
    if np.any(w2 <= 0):
        return 1e3
    v = float(np.var(np.log(w2)))
    hist.append((v, w2.copy(), np.asarray(x).copy()))
    if len(hist) % 10 == 1:
        g.message(f"eval {len(hist)}: loss {v:.6f}, omega^2 = {np.array2string(w2, precision=3)}")
    return v


t0 = time.time()
l0 = loss(x0)
w0 = hist[-1][1]
g.message(f"start (analytic FA, N={N}, longitudinal={longitudinal}): loss {l0:.6f}, omega^2 = {np.array2string(w0, precision=3)}")
x, fv, nev = nelder_mead(loss, x0, step=0.3, tol=1e-7, max_eval=maxeval)
co, z = coeffs_of(x)
w2 = omega2(co)
g.message(f"trained: loss {fv:.6f} after {nev} evals ({time.time() - t0:.0f} s); omega^2 = {np.array2string(w2, precision=3)}")
g.message(f"start roots   {np.array2string(z0, precision=3)}")
g.message(f"trained roots {np.array2string(z, precision=3)}")
g.message(f"trained (alpha, beta) = {[(round(float(a), 6), round(float(b), 6)) for a, b in co]}")

# validation on FRESH probes (guards against fitting the probe noise of the fixed training probes)
rngv = g.random("train_poly_claude validation")
GXv = []
for G in filt:
    lst = []
    for i in range(nprobe):
        xi = g.group.cartesian(U)
        rngv.normal_element(xi)
        lst.append(G(xi))
    GXv.append(lst)
GX_train, norms_train = GX, norms
GX, norms = GXv, [[ip(p, p) for p in lst] for lst in GXv]
w_start_v = omega2(coeffs_of(x0)[0])
w_tr_v = omega2(co)
GX, norms = GX_train, norms_train
v_start, v_tr = float(np.var(np.log(w_start_v))), float(np.var(np.log(w_tr_v)))
g.message(f"VALIDATION (fresh probes): start loss {v_start:.6f} omega^2 {np.array2string(w_start_v, precision=3)}")
g.message(f"VALIDATION (fresh probes): trained loss {v_tr:.6f} omega^2 {np.array2string(w_tr_v, precision=3)}")
if g.rank() == 0:
    with open(out + "_coeffs.py", "wt") as f:
        f.write(f"# trained by train_poly_claude.py on {load}; N={N}, longitudinal={longitudinal}, windows={windows}, sigma={sigma}, nprobe={nprobe}\n")
        f.write(f"# loss Var log omega_j^2: start {l0:.6f} -> trained {fv:.6f} ({nev} evals)\n")
        f.write(f"# start roots {list(np.round(z0, 6))}\n# trained roots {list(np.round(z, 6))}\n")
        f.write(f"# validation (fresh probes): start {v_start:.6f} -> trained {v_tr:.6f}\n")
        f.write(f"COEFFS_TRAINED = {[(round(float(a), 6), round(float(b), 6)) for a, b in co]}\n")
    np.savetxt(out + "_history.dat", np.array([[i, h[0]] + list(h[1]) for i, h in enumerate(hist)]),
               header="eval loss omega2_per_window(" + ",".join(str(w) for w in windows) + ")")
