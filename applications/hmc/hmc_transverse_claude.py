#!/usr/bin/env python3
#
# RMHMC driver with the hand-coded covariant transverse smearing kernel and implicit leapfrog.
#   H = (1/2) <A(U) P, A(U) P> + S_gauge(U),   A = transverse_smear(eta, niter)
# NOT EXACT yet: the implicit leapfrog on the group is not volume preserving for a U-dependent kernel
# (N. H. Christ, L.-C. Jin, C. Lehner, E. Lundstrum, N. Matsumoto, Phys. Rev. D 112, 034507 (2025),
# arXiv:2412.19904); NM's fix is deferred. Use for acceptance / timing / dynamics studies only.
# Design: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, "Implementation log" entry (5).
#
# Usage: python3 hmc_transverse_claude.py --eta 0.1 --niter 2 --tau 0.5 --nstep 10 --n 20 \
#          [--beta 2.95] [--action iwasaki|wilson] [--L 4,4,4,8] [--seed s] [--tol 1e-10]
#          [--load file] [--save file]   (gauge config in NERSC format; --save writes the final config)
#
import gpt as g
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transverse_smear_claude import transverse_kinetic
from implicit_leapfrog_claude import implicit_leapfrog

eta = g.default.get_float("--eta", 0.1)
niter = g.default.get_int("--niter", 2)
tau = g.default.get_float("--tau", 0.5)
nstep = g.default.get_int("--nstep", 10)
n = g.default.get_int("--n", 20)
beta = g.default.get_float("--beta", 2.95)
action = g.default.get("--action", "iwasaki")
L = [int(x) for x in g.default.get("--L", "4,4,4,8").split(",")]
seed = g.default.get("--seed", "hmc_transverse_claude")
tol = g.default.get_float("--tol", 1e-10)
load = g.default.get("--load", None)
save = g.default.get("--save", None)

grid = g.grid(L, g.double)
rng = g.random(seed)

# start as in Lehner's rmhmc.py: unit links perturbed by normal_element(scale = 0.3)
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

g.message(f"lattice {L}, {action} beta = {beta}, eta = {eta}, niter = {niter}, tau = {tau}, nstep = {nstep}, tol = {tol}")
g.message("# traj  plaq  dH  exp(-dH)  acc  acc_rate  n_P_avg  n_U_avg  sec")

nacc = 0
for i in range(n):
    t0 = time.time()
    K.draw(U + P, rng)
    accrej = metro(U)
    h0 = lf.hamiltonian(U, P)
    lf.niter_log = []
    lf(U, P, tau, nstep)
    h1 = lf.hamiltonian(U, P)
    acc = accrej(h1, h0)
    nacc += int(acc)
    nP = np.mean([x[0] for x in lf.niter_log])
    nU = np.mean([x[1] for x in lf.niter_log])
    plaq = g.qcd.gauge.plaquette(U)
    g.message(
        f"TRAJ {i} {plaq:.8f} {h1 - h0:.6e} {np.exp(-(h1 - h0)):.6f} {int(acc)} {nacc / (i + 1):.3f} {nP:.1f} {nU:.1f} {time.time() - t0:.1f}"
    )

if save is not None:
    g.save(save, U, g.format.nersc())
    g.message(f"saved {save}")
