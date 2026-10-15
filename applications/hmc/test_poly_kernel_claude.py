#!/usr/bin/env python3
#
# Validation of poly_kernel_claude.py (unmasked N = 8 polynomial FA kernel, plan entry (15)).
#   1. free field (U = 1): A P == FFT^{-1}[ p(t_mu(k)) FFT P ],  t_mu(k) = sum_{nu != mu} (4/3 2cos k_nu - 1/12 2cos 2k_nu)
#   2. hermiticity <Q, A P> = <A Q, P>; gauge covariance; CG inverse A^{-1} A P = P; positivity <P, A P> > 0
#   3. poly_kinetic: draw consistency; P- and U-gradients vs finite differences
#   4. poly_logdet:  draw consistency; U-gradient vs finite differences
#   5. implicit leapfrog with H = H_kin + S + H_R (R fixed): reversibility, eps^2 scaling   (--skip_leapfrog to skip)
# Usage: python3 test_poly_kernel_claude.py [--L 8] [--skip_leapfrog] [--longitudinal]
#   --longitudinal: full improved Laplacian (incl. nu = mu hops) with poly_coeffs_full_claude.COEFFS[8]
#
import gpt as g
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poly_kernel_claude import poly_kernel, poly_kinetic, poly_logdet, action_with_R, COEFFS_N8, poly_functional_base
from implicit_leapfrog_claude import implicit_leapfrog

L = g.default.get_int("--L", 8)
skip_leapfrog = "--skip_leapfrog" in sys.argv
LONG = "--longitudinal" in sys.argv
if LONG:
    from poly_coeffs_full_claude import COEFFS as _CF

    COEFFS_N8 = _CF[8]
    poly_functional_base.longitudinal = True
ND = 4
grid = g.grid([L] * ND, g.double)
rng = g.random("test_poly_kernel_claude")
ok = True


def check(name, val, tol_):
    global ok
    passed = val < tol_
    ok = ok and passed
    g.message(f"[{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol_:.0e})")


def rel(X, Y):
    return (sum(g.norm2(x - y) for x, y in zip(X, Y)) / sum(g.norm2(y) for y in Y)) ** 0.5


# 1. free field
Uunit = g.qcd.gauge.unit(grid)
P1 = g.group.cartesian(Uunit)
gen = P1[0].otype.generators(np.complex128)[0].array
tr00 = np.trace(gen @ gen).real
coor = g.coordinates(P1[0])
idx = tuple(coor[:, d] for d in range(ND))
phi = np.random.default_rng(2).normal(size=(ND,) + (L,) * ND)
for mu in range(ND):
    P1[mu][coor] = (phi[mu][idx][:, None, None] * gen[None]).astype(np.complex128)
A0 = poly_kernel(Uunit, COEFFS_N8, longitudinal=LONG)
out = A0.forward(P1)
res = np.zeros_like(phi)
for mu in range(ND):
    v = out[mu][coor]
    res[mu][idx] = np.real(np.einsum("nij,ji->n", v, gen)) / tr00
k = 2 * np.pi * np.indices((L,) * ND) / L
ref = np.zeros_like(phi)
for mu in range(ND):
    t = sum((4.0 / 3.0) * 2 * np.cos(k[nu]) - (1.0 / 12.0) * 2 * np.cos(2 * k[nu]) for nu in range(ND) if (LONG or nu != mu))
    p = np.prod([1 + a * t + b * t * t for a, b in COEFFS_N8], axis=0)
    ref[mu] = np.real(np.fft.ifftn(p * np.fft.fftn(phi[mu])))
check("1. free field A P vs analytic symbol (FFT)", np.linalg.norm(res - ref) / np.linalg.norm(ref), 1e-12)

# 2. operator properties on a random gauge field
U = g.qcd.gauge.random(grid, rng)
P = g.group.cartesian(U)
Q = g.group.cartesian(U)
rng.normal_element(P)
rng.normal_element(Q)
A = poly_kernel(U, COEFFS_N8, longitudinal=LONG)
AP = A.forward(P)
a, b = poly_kernel.ip(Q, AP), poly_kernel.ip(A.forward(Q), P)
check("2. hermiticity", abs(a - b) / abs(a), 1e-13)
check("2. positivity <P, A P> > 0 (value -> pass if > 0)", 0.0 if poly_kernel.ip(P, AP) > 0 else 1.0, 0.5)
V = g.lattice(U[0])
rng.element(V)
UV = [g(V * U[mu] * g.cshift(g.adj(V), mu, 1)) for mu in range(4)]
PV = [g(V * x * g.adj(V)) for x in P]
for x, y in zip(PV, P):
    x.otype = y.otype
check("2. gauge covariance", rel(poly_kernel(UV, COEFFS_N8, longitudinal=LONG).forward(PV), [g(V * x * g.adj(V)) for x in AP]), 1e-13)
X = A.inverse(AP)
check("2. CG inverse A^-1 A P = P", rel(X, P), 1e-10)
g.message(f"CG iterations: {A.cg_iters}")

# 3. kinetic
K = poly_kinetic(COEFFS_N8)
v0 = K.draw(U + P, rng)
check("3. kinetic draw consistency", abs(K(U + P) - v0) / abs(v0), 1e-10)
for name, df in [("3. kinetic P-gradient", P), ("3. kinetic U-gradient (force)", U)]:
    try:
        K.assert_gradient_error(rng, U + P, df, 1e-3, 1e-8)
        check(name + " vs finite differences", 0.0, 1.0)
    except AssertionError:
        check(name + " vs finite differences", 1.0, 1.0)

# 4. log det term
D = poly_logdet(COEFFS_N8)
R = g.group.cartesian(U)
v0 = D.draw(U + R, rng)
check("4. logdet draw consistency", abs(D(U + R) - v0) / abs(v0), 1e-10)
try:
    D.assert_gradient_error(rng, U + R, U, 1e-3, 1e-7)
    check("4. logdet U-gradient vs finite differences", 0.0, 1.0)
except AssertionError:
    check("4. logdet U-gradient vs finite differences", 1.0, 1.0)

# 5. leapfrog with H = H_kin + S + H_R (R fixed)
if not skip_leapfrog:
    S = g.qcd.gauge.action.wilson(5.6)
    U_init = g.qcd.gauge.unit(grid)
    rng.normal_element(U_init, scale=0.3)
    R0 = g.group.cartesian(U_init)
    D.draw(U_init + R0, g.random("fixed R"))
    act = action_with_R(S, D, R0)

    def fresh():
        U_ = g.copy(U_init)
        P_ = g.group.cartesian(U_)
        K.draw(U_ + P_, g.random("fixed momenta"))
        return U_, P_

    tau = 0.2
    U_, P_ = fresh()
    U0, P0 = g.copy(U_), g.copy(P_)
    lf = implicit_leapfrog(K, act, tol=1e-12, cache_action=True)
    lf(U_, P_, tau, 4)
    for p in P_:
        p @= -p
    lf(U_, P_, tau, 4)
    for p in P_:
        p @= -p
    check("5. reversibility U", rel(U_, U0), 1e-9)
    check("5. reversibility P", rel(P_, P0), 1e-9)
    dH = []
    for nstep in [4, 8, 16]:
        U_, P_ = fresh()
        lf = implicit_leapfrog(K, act, tol=1e-12, cache_action=True)
        h0 = lf.hamiltonian(U_, P_)
        lf(U_, P_, tau, nstep)
        dH.append(lf.hamiltonian(U_, P_) - h0)
        g.message(f"nstep {nstep}: dH = {dH[-1]:.6e}, (n_P, n_U) = {lf.niter_log[0]}")
    for i in range(2):
        r = abs(dH[i] / dH[i + 1])
        check(f"5. eps^2 scaling |ratio - 4| ({4 * 2**i} -> {8 * 2**i})", abs(r - 4), 0.6)

g.message("ALL PASS" if ok else "SOME FAILED")
