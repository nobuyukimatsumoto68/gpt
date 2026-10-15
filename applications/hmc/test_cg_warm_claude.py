#!/usr/bin/env python3
#
# A/B test of the CG warm start in poly_logdet (poly_kernel_claude.py): same leapfrog steps with
# warm_start = False / True; compare CG iteration counts, final (U, P) and dH.
# Usage: python3 test_cg_warm_claude.py [--L 4] [--nstep 4]
#
import gpt as g
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poly_kernel_claude import poly_kinetic, poly_logdet, action_with_R, COEFFS_N8
from implicit_leapfrog_claude import implicit_leapfrog

L = g.default.get_int("--L", 4)
nstep = g.default.get_int("--nstep", 4)
grid = g.grid([L] * 4, g.double)
rng = g.random("test_cg_warm_claude")
U_init = g.qcd.gauge.unit(grid)
rng.normal_element(U_init, scale=0.3)
S = g.qcd.gauge.action.wilson(5.6)

res = {}
for warm in [False, True]:
    K = poly_kinetic(COEFFS_N8)
    D = poly_logdet(COEFFS_N8)
    D.warm_start = warm
    U = g.copy(U_init)
    P = g.group.cartesian(U)
    R = g.group.cartesian(U)
    D.draw(U + R, g.random("fixed R"))
    K.draw(U + P, g.random("fixed momenta"))
    lf = implicit_leapfrog(K, action_with_R(S, D, R), tol=1e-12, cache_action=True)
    h0 = lf.hamiltonian(U, P)
    D.cg_log = []
    t0 = time.time()
    lf(U, P, 0.2, nstep)
    dt = time.time() - t0
    h1 = lf.hamiltonian(U, P)
    res[warm] = (U, P, h1 - h0, list(D.cg_log), dt)
    g.message(f"warm_start={warm}: CG iterations {D.cg_log}, total {sum(D.cg_log)}, dH {h1 - h0:.10e}, time {dt:.1f} s")

rel = lambda X, Y: (sum(g.norm2(x - y) for x, y in zip(X, Y)) / sum(g.norm2(y) for y in Y)) ** 0.5
g.message(f"final U difference {rel(res[True][0], res[False][0]):.3e}, P difference {rel(res[True][1], res[False][1]):.3e}, dH difference {abs(res[True][2] - res[False][2]):.3e}")
g.message(f"CG iteration ratio warm/cold = {sum(res[True][3]) / sum(res[False][3]):.3f}, time ratio = {res[True][4] / res[False][4]:.3f}")
