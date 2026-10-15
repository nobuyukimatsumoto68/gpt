#!/usr/bin/env python3
#
# Validation + A/B timing of poly_kernel.grad_A_fast (prefix/suffix products + bilinear merge) vs grad_A.
#   1. grad_A_fast(L, X) == grad_A(L, X) on random fields (relative difference)
#   2. same leapfrog steps (4 MD steps, H = H_kin + S + H_R, warm-start CG) with fast_grad False / True:
#      final (U, P), dH, wall time
# Usage: python3 test_grad_fast_claude.py [--L 4] [--nstep 4]
#
import gpt as g
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poly_kernel_claude import poly_kernel, poly_kinetic, poly_logdet, action_with_R, COEFFS_N8, poly_functional_base
from implicit_leapfrog_claude import implicit_leapfrog

L = g.default.get_int("--L", 4)
nstep = g.default.get_int("--nstep", 4)
grid = g.grid([L] * 4, g.double)
rng = g.random("test_grad_fast_claude")
rel = lambda X, Y: (sum(g.norm2(x - y) for x, y in zip(X, Y)) / sum(g.norm2(y) for y in Y)) ** 0.5

U = g.qcd.gauge.random(grid, rng)
Lf, Xf = g.group.cartesian(U), g.group.cartesian(U)
rng.normal_element(Lf)
rng.normal_element(Xf)
A = poly_kernel(U, COEFFS_N8)
t0 = time.time(); F0 = A.grad_A(Lf, Xf); t1 = time.time(); F1 = A.grad_A_fast(Lf, Xf); t2 = time.time()
g.message(f"1. grad_A_fast vs grad_A: rel diff {rel(F1, F0):.3e}; time {t1 - t0:.2f} s -> {t2 - t1:.2f} s (ratio {(t2 - t1) / (t1 - t0):.3f})")

U_init = g.qcd.gauge.unit(grid)
rng.normal_element(U_init, scale=0.3)
S = g.qcd.gauge.action.wilson(5.6)
res = {}
for fast in [False, True]:
    poly_functional_base.fast_grad = fast
    K, D = poly_kinetic(COEFFS_N8), poly_logdet(COEFFS_N8)
    U_ = g.copy(U_init)
    P_ = g.group.cartesian(U_)
    R_ = g.group.cartesian(U_)
    D.draw(U_ + R_, g.random("fixed R"))
    K.draw(U_ + P_, g.random("fixed momenta"))
    lf = implicit_leapfrog(K, action_with_R(S, D, R_), tol=1e-12, cache_action=True)
    h0 = lf.hamiltonian(U_, P_)
    t0 = time.time()
    lf(U_, P_, 0.2, nstep)
    dt = time.time() - t0
    res[fast] = (U_, P_, lf.hamiltonian(U_, P_) - h0, dt, list(lf.niter_log))
    g.message(f"2. fast_grad={fast}: dH {res[fast][2]:.10e}, wall {dt:.1f} s, (n_P, n_U) {lf.niter_log}")
poly_functional_base.fast_grad = True
g.message(f"2. final U diff {rel(res[True][0], res[False][0]):.3e}, P diff {rel(res[True][1], res[False][1]):.3e}, dH diff {abs(res[True][2] - res[False][2]):.3e}")
g.message(f"2. wall-time ratio fast/original = {res[True][3] / res[False][3]:.3f}")
