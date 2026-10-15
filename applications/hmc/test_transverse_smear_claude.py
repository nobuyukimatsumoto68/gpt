#!/usr/bin/env python3
#
# Validation of transverse_smear_claude.py (kernel A only; no HMC yet).
#   1. flow composition: A^{-1} A P = P and A A^{-1} P = P   (plus |A P - P| to show A is nontrivial)
#   2. adjoint:          <Q, A P> = <A^\dagger Q, P>
#   3. gauge covariance: A_{U^V}(V P V^\dagger) = V (A_U P) V^\dagger
#   4. algebra:          A P stays in su(N)   (group defect)
# Usage: python3 test_transverse_smear_claude.py
#
import gpt as g
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transverse_smear_claude import transverse_smear

grid = g.grid([4, 4, 4, 4], g.double)
rng = g.random("test_transverse_smear_claude")

U = g.qcd.gauge.random(grid, rng)
P = g.group.cartesian(U)
Q = g.group.cartesian(U)
rng.element(P)
rng.element(Q)

nP = sum(g.norm2(x) for x in P)  # g.norm2(list) returns a list
tol = 1e-26
ok = True


def check(name, val, tol_):
    global ok
    passed = val < tol_
    ok = ok and passed
    g.message(f"[{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol_:.0e})")


for eta, niter in [(0.025, 2), (0.1, 3), (0.3, 1)]:
    g.message(f"=== eta = {eta}, niter = {niter} ===")
    A = transverse_smear(U, eta, niter)

    AP = A.forward(P)
    g.message(f"|A P - P|^2 / |P|^2 = {g.norm2(g(AP[0] - P[0]) ) / g.norm2(P[0]):.3e} (mu=0; nontrivial)")

    # 1. composition
    check("|A^-1 A P - P|^2/|P|^2", sum(g.norm2(x - y) for x, y in zip(A.inverse(AP), P)) / nP, tol)
    check("|A A^-1 P - P|^2/|P|^2", sum(g.norm2(x - y) for x, y in zip(A.forward(A.inverse(P)), P)) / nP, tol)

    # 2. adjoint
    lhs = sum(g.group.inner_product(x, y) for x, y in zip(Q, AP))
    rhs = sum(g.group.inner_product(x, y) for x, y in zip(A.adjoint(Q), P))
    check("|<Q,AP> - <A^dag Q,P>| / |<Q,AP>|", abs(lhs - rhs) / abs(lhs), 1e-13)

    # 3. gauge covariance
    V = g.lattice(U[0])
    rng.element(V)
    UV = [g(V * U[mu] * g.cshift(g.adj(V), mu, 1)) for mu in range(4)]
    PV = [g(V * x * g.adj(V)) for x in P]
    for x, y in zip(PV, P):
        x.otype = y.otype
    AV = transverse_smear(UV, eta, niter)
    lhs = AV.forward(PV)
    rhs = [g(V * x * g.adj(V)) for x in AP]
    check("|A_{U^V}(P^V) - (A_U P)^V|^2/|P|^2", sum(g.norm2(x - y) for x, y in zip(lhs, rhs)) / nP, tol)

    # 4. algebra
    check("group defect of A P", max(g.group.defect(x) for x in AP), 1e-12)

g.message("ALL PASS" if ok else "SOME FAILED")
