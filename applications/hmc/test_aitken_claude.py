#!/usr/bin/env python3
#
# Validation of the Aitken-accelerated implicit leapfrog (implicit_leapfrog.step_aitken) against the
# plain fixed point (implicit_leapfrog.step), H = (1/2)<A(U)P, A(U)P> + S_Wilson(U).
#   1. same trajectory: plain vs Aitken final (U, P) agree to ~ solver tolerance
#   2. fixed-point iteration counts (n_P, n_U) per step: plain vs Aitken
#   3. reversibility with Aitken
# Usage: python3 test_aitken_claude.py
#
import gpt as g
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transverse_smear_claude import transverse_kinetic
from implicit_leapfrog_claude import implicit_leapfrog

grid = g.grid([4, 4, 4, 4], g.double)
rng = g.random("test_aitken_claude")
U_init = g.qcd.gauge.unit(grid)
rng.normal_element(U_init, scale=0.3)
S = g.qcd.gauge.action.wilson(5.6)
tau, nstep, tol = 0.5, 8, 1e-12

ok = True


def check(name, val, tol_):
    global ok
    passed = val < tol_
    ok = ok and passed
    g.message(f"[{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol_:.0e})")


def rel(X, Y):
    return (sum(g.norm2(x - y) for x, y in zip(X, Y)) / sum(g.norm2(y) for y in Y)) ** 0.5


for eta, niter in [(0.1, 2), (0.3, 1)]:
    g.message(f"=== eta = {eta}, niter = {niter}, nstep = {nstep}, tol = {tol} ===")
    K = transverse_kinetic(eta, niter)
    res = {}
    # for name, use_aitken in [("plain", False), ("aitken", True)]:
    for name, use_aitken, mode in [("plain", False, None), ("vector", True, "vector"), ("aitken", True, "componentwise")]:
        U = g.copy(U_init)
        P = g.group.cartesian(U)
        K.draw(U + P, g.random("fixed momenta"))
        lf = implicit_leapfrog(K, S, tol=tol, use_aitken=use_aitken, aitken_mode=mode)
        t0 = time.time()
        try:
            lf(U, P, tau, nstep)
        except RuntimeError as e:
            g.message(f"{name:6s}: FAILED to converge ({e})")
            res[name] = None
            continue
        dt = time.time() - t0
        nP = np.mean([x[0] for x in lf.niter_log])
        nU = np.mean([x[1] for x in lf.niter_log])
        g.message(f"{name:6s}: mean (n_P, n_U) = ({nP:.1f}, {nU:.1f}), time {dt:.1f} s")
        res[name] = (U, P, nP + nU, dt)

    # componentwise (CPN-style) Aitken: reported only (it stalls near tol in SU(3); see plan entry (6))
    if res["aitken"] is not None:
        g.message(f"componentwise: |dU|/|U| vs plain = {rel(res['aitken'][0], res['plain'][0]):.3e}, iteration ratio = {res['aitken'][2] / res['plain'][2]:.2f}")
    # 1. same trajectory (vector Aitken)
    check("plain vs vector Aitken |dU|/|U|", rel(res["vector"][0], res["plain"][0]), 1e-10)
    check("plain vs vector Aitken |dP|/|P|", rel(res["vector"][1], res["plain"][1]), 1e-10)
    # 2. fewer iterations
    g.message(f"vector Aitken / plain: iteration ratio = {res['vector'][2] / res['plain'][2]:.2f}, time ratio = {res['vector'][3] / res['plain'][3]:.2f}")

    # 3. reversibility with vector Aitken
    U, P = res["vector"][0], res["vector"][1]
    lf = implicit_leapfrog(K, S, tol=tol, use_aitken=True, aitken_mode="vector")
    for p in P:
        p @= -p
    lf(U, P, tau, nstep)
    for p in P:
        p @= -p
    U0 = g.copy(U_init)
    P0 = g.group.cartesian(U0)
    K.draw(U0 + P0, g.random("fixed momenta"))
    check("vector Aitken reversibility |dU|/|U|", rel(U, U0), 1e-10)
    check("vector Aitken reversibility |dP|/|P|", rel(P, P0), 1e-10)

g.message("ALL PASS" if ok else "SOME FAILED")
