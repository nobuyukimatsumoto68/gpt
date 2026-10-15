#!/usr/bin/env python3
#
# Momentum decorrelation along single RMHMC trajectories (NM study, 2026-10-15).
# Design: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, "Study: momentum decorrelation along a trajectory".
#
# Per trajectory (draw P(0) = A^{-1} xi, implicit leapfrog to s_tot, Metropolis), after every `meas_every` MD steps:
#   C(t)          = <P(0), P(t)> / (|P(0)| |P(t)|)
#   C_j(t)        = <G_j[U(0)] P(0), G_j[U(t)] P(t)> / (|..| |..|),   G_j = g_j(-\Delta[U]) covariant Gaussian filter
# Shells: k0 = 2 pi j / L (on axis 0), j in --js; sigma = 2 pi / L; Chebyshev coefficients fixed per shell
# (lambda_0 = \hat k_0^2), applied with the Laplacian of the current U(t) (the filter holds the U list, which
# the leapfrog updates in place).
# NOT EXACT (no volume preservation for U-dependent kernels; arXiv:2412.19904).
#
# Usage:
#   python3 decorrelation_claude.py --eta 0.05 --niter 2 --stot 3.0 --nstep 60 --n 10 --load therm.cfg \
#       [--L 8,8,8,8] [--beta 2.95] [--action iwasaki] [--js 1,2,4] [--meas_every 1] [--tol 1e-10] \
#       [--seed s] [--out file.dat] [--save final.cfg]
# Output (--out): one line per (trajectory, measurement): traj acc dH k t C C_j1 C_j2 ...
#
import gpt as g
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transverse_smear_claude import transverse_kinetic
from implicit_leapfrog_claude import implicit_leapfrog
from momentum_filter_claude import covariant_gaussian_filter, shell_window, cosine_similarity

eta = g.default.get_float("--eta", 0.05)
niter = g.default.get_int("--niter", 2)
stot = g.default.get_float("--stot", 3.0)
nstep = g.default.get_int("--nstep", 60)
n = g.default.get_int("--n", 10)
beta = g.default.get_float("--beta", 2.95)
action = g.default.get("--action", "iwasaki")
L = [int(x) for x in g.default.get("--L", "8,8,8,8").split(",")]
js = [int(x) for x in g.default.get("--js", "1,2,4").split(",")]
meas_every = g.default.get_int("--meas_every", 1)
tol = g.default.get_float("--tol", 1e-10)
seed = g.default.get("--seed", "decorrelation_claude")
load = g.default.get("--load", None)
save = g.default.get("--save", None)
out = g.default.get("--out", None)

grid = g.grid(L, g.double)
rng = g.random(seed)
U = g.qcd.gauge.unit(grid)
rng.normal_element(U, scale=0.3)
if load is not None:
    U0 = g.load(load)
    for mu in range(len(U)):
        U[mu] @= U0[mu]
    g.message(f"loaded {load}, plaquette = {g.qcd.gauge.plaquette(U)}")
P = g.group.cartesian(U)

S = g.qcd.gauge.action.iwasaki(beta) if action == "iwasaki" else g.qcd.gauge.action.wilson(beta)
K = transverse_kinetic(eta, niter)
lf = implicit_leapfrog(K, S, tol=tol)
metro = g.algorithms.markov.metropolis(rng)

# shells on axis 0
sigma = 2.0 * np.pi / L[0]
filters = []
for j in js:
    lam0, sig_lam = shell_window(2.0 * np.pi * j / L[0], sigma)
    G = covariant_gaussian_filter(U, lam0, sig_lam)  # holds the U list -> uses \Delta[U(t)] when applied
    filters.append(G)
    g.message(f"shell j = {j}: lambda0 = {lam0:.4f}, sigma_lambda = {sig_lam:.4f}, Chebyshev degree {G.degree} (err {G.cheb_error:.1e})")

g.message(f"lattice {L}, {action} beta = {beta}, eta = {eta}, niter = {niter}, stot = {stot}, nstep = {nstep}, meas_every = {meas_every}")

fout = open(out, "wt") if (out is not None and g.rank() == 0) else None
if fout is not None:
    fout.write(f"# eta {eta} niter {niter} stot {stot} nstep {nstep} L {L} {action} beta {beta} js {js}\n")
    fout.write("# traj acc dH k t C " + " ".join(f"C_j{j}" for j in js) + "\n")

nacc = 0
for i in range(n):
    t0 = time.time()
    K.draw(U + P, rng)
    accrej = metro(U)
    h0 = lf.hamiltonian(U, P)

    P0 = g.copy(P)
    F0 = [G(P0) for G in filters]  # G_j[U(0)] P(0)
    rows = [(0, 0.0, 1.0, [1.0] * len(js))]

    def measure(k, U_, P_):
        # if k % meas_every != 0:
        if meas_every <= 0 or k % meas_every != 0:  # meas_every = 0: no measurements (nstep tuning)
            return
        c = cosine_similarity(P0, P_)
        cj = [cosine_similarity(f0, G(P_)) for f0, G in zip(F0, filters)]
        rows.append((k, k * stot / nstep, c, cj))

    lf.niter_log = []
    lf(U, P, stot, nstep, callback=measure)
    h1 = lf.hamiltonian(U, P)
    acc = accrej(h1, h0)
    nacc += int(acc)
    dH = h1 - h0

    for k, t, c, cj in rows:
        line = f"{i} {int(acc)} {dH:.6e} {k} {t:.6f} {c:.8f} " + " ".join(f"{x:.8f}" for x in cj)
        if fout is not None:
            fout.write(line + "\n")
    if fout is not None:
        fout.flush()

    nP = np.mean([x[0] for x in lf.niter_log])
    nU = np.mean([x[1] for x in lf.niter_log])
    last = rows[-1]
    g.message(
        f"TRAJ {i} plaq {g.qcd.gauge.plaquette(U):.8f} dH {dH:.6e} acc {int(acc)} rate {nacc / (i + 1):.3f} "
        f"C(stot) {last[2]:.4f} C_j(stot) {' '.join(f'{x:.4f}' for x in last[3])} (nP,nU) ({nP:.1f},{nU:.1f}) {time.time() - t0:.1f}s"
    )

if fout is not None:
    fout.close()
if save is not None:
    g.save(save, U, g.format.nersc())
    g.message(f"saved {save}")
