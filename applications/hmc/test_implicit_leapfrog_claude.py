#!/usr/bin/env python3
#
# Validation of implicit_leapfrog_claude.py with H = (1/2)<A(U)P, A(U)P> + S_Wilson(U).
#   1. reversibility: integrate tau, flip P, integrate tau, flip P -> back to the start (to ~ solver tol)
#   2. energy error scaling: |dH| ~ eps^2 for fixed tau (ratio ~ 4 when nstep doubles)
#   3. K = 1 limit (eta = 0): agrees with the plain explicit leapfrog (sanity of signs/conventions)
# Usage: python3 test_implicit_leapfrog_claude.py
#
import gpt as g
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transverse_smear_claude import transverse_kinetic
from implicit_leapfrog_claude import implicit_leapfrog

grid = g.grid([4, 4, 4, 4], g.double)
rng = g.random("test_implicit_leapfrog_claude")

U_init = g.qcd.gauge.unit(grid)
rng.normal_element(U_init, scale=0.3)
S = g.qcd.gauge.action.wilson(5.6)

eta, niter = 0.1, 2
K = transverse_kinetic(eta, niter)
tau = 0.5

ok = True


def check(name, val, tol_):
    global ok
    passed = val < tol_
    ok = ok and passed
    g.message(f"[{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol_:.0e})")


def fresh():
    U = g.copy(U_init)
    P = g.group.cartesian(U)
    K.draw(U + P, g.random("fixed momenta"))
    return U, P


# 1. reversibility
U, P = fresh()
U0, P0 = g.copy(U), g.copy(P)
lf = implicit_leapfrog(K, S, tol=1e-13)
lf(U, P, tau, 8)
for p in P:
    p @= -p
lf(U, P, tau, 8)
for p in P:
    p @= -p
dU = sum(g.norm2(x - y) for x, y in zip(U, U0)) / sum(g.norm2(x) for x in U0)
dP = sum(g.norm2(x - y) for x, y in zip(P, P0)) / sum(g.norm2(x) for x in P0)
g.message(f"fixed-point iterations per step (n_P, n_U): {lf.niter_log[:4]} ...")
check("reversibility |dU|/|U| (rel. norm, not squared)", dU**0.5, 1e-10)
check("reversibility |dP|/|P|", dP**0.5, 1e-10)

# 2. energy error scaling
dH = []
for nstep in [4, 8, 16]:
    U, P = fresh()
    lf = implicit_leapfrog(K, S, tol=1e-13)
    h0 = lf.hamiltonian(U, P)
    lf(U, P, tau, nstep)
    h1 = lf.hamiltonian(U, P)
    dH.append(h1 - h0)
    g.message(f"nstep = {nstep:3d}: H0 = {h0:.6f}, dH = {h1 - h0:.6e}")
for i in range(len(dH) - 1):
    r = abs(dH[i] / dH[i + 1])
    g.message(f"dH ratio nstep {4 * 2**i} / {8 * 2**i}: {r:.3f} (expect ~4)")
    check(f"eps^2 scaling |ratio - 4| ({4 * 2**i} -> {8 * 2**i})", abs(r - 4), 0.5)

# 3. K = 1 (eta = 0) agrees with the explicit leapfrog
K0 = transverse_kinetic(0.0, 1)
U, P = fresh()
Ue, Pe = g.copy(U), g.copy(P)
implicit_leapfrog(K0, S, tol=1e-14)(U, P, tau, 8)
eps = tau / 8
for i in range(8):
    F = S.gradient(Ue, Ue)
    for mu in range(4):
        Pe[mu] @= Pe[mu] - 0.5 * eps * F[mu]
        Ue[mu] @= g.group.compose(g(eps * Pe[mu]), Ue[mu])
    F = S.gradient(Ue, Ue)
    for mu in range(4):
        Pe[mu] @= Pe[mu] - 0.5 * eps * F[mu]
d = sum(g.norm2(x - y) for x, y in zip(U + P, Ue + Pe)) / sum(g.norm2(x) for x in Ue + Pe)
check("eta = 0 vs explicit leapfrog (rel.)", d**0.5, 1e-12)

g.message("ALL PASS" if ok else "SOME FAILED")
