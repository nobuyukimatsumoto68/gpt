#!/usr/bin/env python3
#
# Cumulative A/B timing of the poly-kernel MD optimisations (same initial state, 4 MD steps, fixed-point tol 1e-10):
#   C0 baseline (fast grad_A + CG warm start)  ->  C1 + transport cache  ->  C2 + action-force reuse  ->  C3 + CG tol 1e-10
# Reports wall time, deviation of final (U, P, dH) from C0, and reversibility for C3.
# Usage: python3 test_poly_speed_claude.py [--L 4] [--nstep 4]
#
import gpt as g
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poly_kernel_claude import poly_kinetic, poly_logdet, action_with_R, COEFFS_N8, poly_functional_base
from implicit_leapfrog_claude import implicit_leapfrog

L = g.default.get_int("--L", 4)
nstep = g.default.get_int("--nstep", 4)
grid = g.grid([L] * 4, g.double)
rng = g.random("test_poly_speed_claude")
U_init = g.qcd.gauge.unit(grid)
rng.normal_element(U_init, scale=0.3)
S = g.qcd.gauge.action.wilson(5.6)
rel = lambda X, Y: (sum(g.norm2(x - y) for x, y in zip(X, Y)) / sum(g.norm2(y) for y in Y)) ** 0.5

configs = [
    ("C0 baseline", dict(cache=False, reuse=False, cg=1e-12)),
    ("C1 +transport cache", dict(cache=True, reuse=False, cg=1e-12)),
    ("C2 +force reuse", dict(cache=True, reuse=True, cg=1e-12)),
    ("C3 +CG tol 1e-10", dict(cache=True, reuse=True, cg=1e-10)),
]


def setup(c):
    poly_functional_base.transport_cache = c["cache"]
    poly_functional_base._shared["key"] = None
    poly_functional_base._shared["kernel"] = None
    K, D = poly_kinetic(COEFFS_N8, cg_tol=c["cg"]), poly_logdet(COEFFS_N8, cg_tol=c["cg"])
    U = g.copy(U_init)
    P = g.group.cartesian(U)
    R = g.group.cartesian(U)
    D.draw(U + R, g.random("fixed R"))
    K.draw(U + P, g.random("fixed momenta"))
    lf = implicit_leapfrog(K, action_with_R(S, D, R), tol=1e-10, cache_action=True, reuse_action=c["reuse"])
    return K, D, U, P, lf


res = {}
for name, c in configs:
    K, D, U, P, lf = setup(c)
    h0 = lf.hamiltonian(U, P)
    t0 = time.time()
    lf(U, P, 0.2, nstep)
    dt = time.time() - t0
    res[name] = (U, P, lf.hamiltonian(U, P) - h0, dt)
    base = res["C0 baseline"]
    g.message(
        f"{name:22s}: wall {dt:6.1f} s (ratio to C0 {dt / base[3]:.3f}); dH {res[name][2]:.10e}; "
        f"vs C0: U {rel(U, base[0]):.1e}, P {rel(P, base[1]):.1e}, dH {abs(res[name][2] - base[2]):.1e}; (n_P,n_U) {lf.niter_log}"
    )

# reversibility with C3
K, D, U, P, lf = setup(configs[-1][1])
U0, P0 = g.copy(U), g.copy(P)
lf(U, P, 0.2, nstep)
for p in P:
    p @= -p
lf(U, P, 0.2, nstep)
for p in P:
    p @= -p
g.message(f"C3 reversibility: U {rel(U, U0):.2e}, P {rel(P, P0):.2e}")
