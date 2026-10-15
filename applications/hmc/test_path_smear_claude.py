#!/usr/bin/env python3
#
# Validation of path_smear_claude.py (generic path layers: T1 plaquette, R rectangle, C chair, mixed flows).
#   0. T1 via the generic path layer == transverse_smear_claude.py (forward and force)
#   1. per kernel: A^{-1} A = A A^{-1} = 1, adjoint, gauge covariance, su(3) defect      (as plan entry (2))
#   2. per kernel: draw, P-gradient and U-gradient vs finite differences, gauge invariance (as plan entry (3))
#   3. mixed flow: implicit leapfrog reversibility and eps^2 scaling                       (as plan entry (4))
# Usage: python3 test_path_smear_claude.py [--L 8] [--skip_leapfrog]
#
import gpt as g
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_smear_claude import path_flow, path_kinetic, kernel_builder
from transverse_smear_claude import transverse_smear, transverse_kinetic
from implicit_leapfrog_claude import implicit_leapfrog

Lx = g.default.get_int("--L", 8)
skip_leapfrog = "--skip_leapfrog" in sys.argv
grid = g.grid([Lx] * 4, g.double)
rng = g.random("test_path_smear_claude")

U = g.qcd.gauge.random(grid, rng)
P = g.group.cartesian(U)
Q = g.group.cartesian(U)
rng.element(P)
rng.element(Q)
nP = sum(g.norm2(x) for x in P)

ok = True


def check(name, val, tol_):
    global ok
    passed = val < tol_
    ok = ok and passed
    g.message(f"[{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol_:.0e})")


def rel(X, Y):
    return (sum(g.norm2(x - y) for x, y in zip(X, Y)) / sum(g.norm2(y) for y in Y)) ** 0.5


builder = kernel_builder(grid)

# 0. T1 generic vs existing
g.message("=== 0. T1 generic vs transverse_smear_claude ===")
eta, niter = 0.1, 2
A_old = transverse_smear(U, eta, niter)
A_new = path_flow(U, [("T1", eta)] * niter, builder)
check("forward: generic vs existing", rel(A_new.forward(P), A_old.forward(P)), 1e-14)
check("force: generic vs existing", rel(A_new.force(P), A_old.force(P)), 1e-13)

specs = {
    "R": [("R", 0.05)],
    "C": [("C", 0.1)],
    "mixed T1+R+C x2": [("T1", 0.05), ("R", 0.03), ("C", 0.05)] * 2,
}

for name, spec in specs.items():
    g.message(f"=== kernel {name}: {spec} ===")
    A = path_flow(U, spec, builder)
    AP = A.forward(P)
    g.message(f"layers: {len(A.seq)}; |A P - P|^2/|P|^2 = {sum(g.norm2(x - y) for x, y in zip(AP, P)) / nP:.3e} (nontrivial)")

    # 1.
    check("A^-1 A P = P", rel(A.inverse(AP), P), 1e-13)
    check("A A^-1 P = P", rel(A.forward(A.inverse(P)), P), 1e-13)
    lhs = sum(g.group.inner_product(x, y) for x, y in zip(Q, AP))
    rhs = sum(g.group.inner_product(x, y) for x, y in zip(A.adjoint(Q), P))
    check("adjoint <Q,AP> = <A^dag Q,P>", abs(lhs - rhs) / abs(lhs), 1e-13)
    V = g.lattice(U[0])
    rng.element(V)
    UV = [g(V * U[mu] * g.cshift(g.adj(V), mu, 1)) for mu in range(4)]
    PV = [g(V * x * g.adj(V)) for x in P]
    for x, y in zip(PV, P):
        x.otype = y.otype
    AV = path_flow(UV, spec, builder)
    check("gauge covariance", rel(AV.forward(PV), [g(V * x * g.adj(V)) for x in AP]), 1e-13)
    check("su(3) defect of A P", max(g.group.defect(x) for x in AP), 1e-12)

    # 2.
    K = path_kinetic(spec)
    PP = g.copy(P)
    v0 = K.draw(U + PP, rng)
    check("draw consistency", abs(K(U + PP) - v0) / abs(v0), 1e-12)
    for gname, dfields in [("P-gradient", PP), ("U-gradient (force)", U)]:
        try:
            K.assert_gradient_error(rng, U + PP, dfields, 1e-3, 1e-8)
            check(gname + " vs finite differences", 0.0, 1.0)
        except AssertionError:
            check(gname + " vs finite differences", 1.0, 1.0)
    PPV = [g(V * x * g.adj(V)) for x in PP]
    for x, y in zip(PPV, PP):
        x.otype = y.otype
    h0 = K(U + PP)
    check("gauge invariance of H_kin", abs(K(UV + PPV) - h0) / abs(h0), 1e-12)

# 3. leapfrog with the mixed flow (smoother config, Wilson beta = 5.6)
if skip_leapfrog:
    g.message("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0)
g.message("=== 3. implicit leapfrog with mixed flow ===")
spec = specs["mixed T1+R+C x2"]
K = path_kinetic(spec)
S = g.qcd.gauge.action.wilson(5.6)
U_init = g.qcd.gauge.unit(grid)
rng.normal_element(U_init, scale=0.3)


def fresh():
    U_ = g.copy(U_init)
    P_ = g.group.cartesian(U_)
    K.draw(U_ + P_, g.random("fixed momenta"))
    return U_, P_


tau = 0.5
U_, P_ = fresh()
U0, P0 = g.copy(U_), g.copy(P_)
lf = implicit_leapfrog(K, S, tol=1e-13)
lf(U_, P_, tau, 4)
for p in P_:
    p @= -p
lf(U_, P_, tau, 4)
for p in P_:
    p @= -p
check("reversibility U", rel(U_, U0), 1e-10)
check("reversibility P", rel(P_, P0), 1e-10)
dH = []
for nstep in [4, 8, 16]:
    U_, P_ = fresh()
    lf = implicit_leapfrog(K, S, tol=1e-13)
    h0 = lf.hamiltonian(U_, P_)
    lf(U_, P_, tau, nstep)
    dH.append(lf.hamiltonian(U_, P_) - h0)
    g.message(f"nstep {nstep}: dH = {dH[-1]:.6e}, (n_P, n_U) = {lf.niter_log[0]}")
for i in range(2):
    r = abs(dH[i] / dH[i + 1])
    check(f"eps^2 scaling |ratio - 4| ({4 * 2**i} -> {8 * 2**i})", abs(r - 4), 0.5)

g.message("ALL PASS" if ok else "SOME FAILED")
