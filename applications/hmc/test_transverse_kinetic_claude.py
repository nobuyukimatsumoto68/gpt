#!/usr/bin/env python3
#
# Validation of transverse_kinetic (H_kin = (1/2) <A(U) P, A(U) P>) in transverse_smear_claude.py.
#   1. draw:             P = A^{-1} xi  =>  H_kin(U, P) = (1/2) <xi, xi>
#   2. P-gradient:       velocity A^dagger A P vs finite differences (gpt assert_gradient_error)
#   3. U-gradient:       backprop force vs finite differences          (gpt assert_gradient_error)
#   4. gauge invariance: H_kin(U^V, V P V^dagger) = H_kin(U, P)
# Usage: python3 test_transverse_kinetic_claude.py
#
import gpt as g
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transverse_smear_claude import transverse_kinetic

grid = g.grid([4, 4, 4, 4], g.double)
rng = g.random("test_transverse_kinetic_claude")

U = g.qcd.gauge.random(grid, rng)
P = g.group.cartesian(U)
rng.element(P)

ok = True


def check(name, val, tol_):
    global ok
    passed = val < tol_
    ok = ok and passed
    g.message(f"[{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol_:.0e})")


for eta, niter in [(0.025, 2), (0.1, 3), (0.3, 1)]:
    g.message(f"=== eta = {eta}, niter = {niter} ===")
    K = transverse_kinetic(eta, niter)

    # 1. draw consistency
    v0 = K.draw(U + P, rng)
    v1 = K(U + P)
    check("draw: |H_kin - (1/2)<xi,xi>| / H_kin", abs(v1 - v0) / abs(v0), 1e-13)

    # 2. + 3. gradients (gpt's checker: 4th-order finite differences along exp(i eps W))
    for name, dfields in [("P-gradient (velocity A^dag A P)", P), ("U-gradient (force)", U)]:
        g.message(f"--- {name}")
        try:
            K.assert_gradient_error(rng, U + P, dfields, 1e-3, 1e-8)
            check(name, 0.0, 1.0)
        except AssertionError:
            check(name, 1.0, 1.0)

    # 4. gauge invariance
    V = g.lattice(U[0])
    rng.element(V)
    UV = [g(V * U[mu] * g.cshift(g.adj(V), mu, 1)) for mu in range(4)]
    PV = [g(V * x * g.adj(V)) for x in P]
    for x, y in zip(PV, P):
        x.otype = y.otype
    h0 = K(U + P)
    check("gauge invariance |H(U^V,P^V) - H(U,P)| / H", abs(K(UV + PV) - h0) / abs(h0), 1e-13)

g.message("ALL PASS" if ok else "SOME FAILED")
