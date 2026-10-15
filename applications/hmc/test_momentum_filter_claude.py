#!/usr/bin/env python3
#
# Validation of momentum_filter_claude.py (covariant Gaussian filter G = g(-\Delta[U])).
#   1. Chebyshev accuracy of g on [0, 16] (scalar)
#   2. free field (U = 1): G P == FFT^{-1} [ g(\hat k^2) FFT P ]   (exact reference)
#   3. gauge covariance:   G_{U^V}(V P V^dag) == V (G_U P) V^dag
#   4. self-adjointness:   <Q, G P> == <G Q, P>
#   5. cosine similarity sanity: cos(P, P) = 1, cos(P, -P) = -1
# Usage: python3 test_momentum_filter_claude.py
#
import gpt as g
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from momentum_filter_claude import covariant_gaussian_filter, shell_window, cosine_similarity, khat2

ok = True


def check(name, val, tol_):
    global ok
    passed = val < tol_
    ok = ok and passed
    g.message(f"[{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol_:.0e})")


def rel(X, Y):
    return (sum(g.norm2(x - y) for x, y in zip(X, Y)) / sum(g.norm2(y) for y in Y)) ** 0.5


for L in [4, 8]:
    grid = g.grid([L, L, L, L], g.double)
    rng = g.random(f"test_momentum_filter_claude_{L}")
    Uunit = g.qcd.gauge.unit(grid)
    P = g.group.cartesian(Uunit)
    Q = g.group.cartesian(Uunit)
    rng.normal_element(P)
    rng.normal_element(Q)
    sigma = 2.0 * np.pi / L

    # momentum-space |hat k|^2 field for the FFT reference
    one = g.complex(grid)
    coor = g.coordinates(one)
    kh2 = sum(4.0 * np.sin(np.pi * coor[:, nu] / L) ** 2 for nu in range(4))
    fft = g.fft()

    for j in sorted(set([1, L // 4, L // 2])):
        k0 = 2.0 * np.pi * j / L
        lam0, sig_lam = shell_window(k0, sigma)
        g.message(f"=== L = {L}, k0 = 2 pi {j}/{L}: lambda0 = {lam0:.4f}, sigma_lambda = {sig_lam:.4f} ===")

        # 1. Chebyshev
        G = covariant_gaussian_filter(Uunit, lam0, sig_lam)
        g.message(f"Chebyshev degree {G.degree}")
        check("Chebyshev max |g - g_M| on [0,16]", G.cheb_error, 1e-8)

        # 2. free field vs FFT
        w = g.complex(grid)
        w[coor] = np.exp(-((kh2 - lam0) ** 2) / (2.0 * sig_lam**2)).astype(np.complex128).reshape(-1, 1)
        ref = [g(g.inv(fft) * (w * g(fft * p))) for p in P]
        GP = G(P)
        check("free field: |G P - FFT reference| / |ref|", rel(GP, ref), 1e-7)
        frac = sum(g.norm2(x) for x in GP) / sum(g.norm2(x) for x in P)
        g.message(f"fraction of |P|^2 passing the filter: {frac:.4f}")

        # 3. gauge covariance (random gauge field U, random V)
        U = g.qcd.gauge.random(grid, rng)
        V = g.lattice(U[0])
        rng.element(V)
        UV = [g(V * U[mu] * g.cshift(g.adj(V), mu, 1)) for mu in range(4)]
        PV = [g(V * x * g.adj(V)) for x in P]
        for x, y in zip(PV, P):
            x.otype = y.otype
        GU = covariant_gaussian_filter(U, lam0, sig_lam)
        GUV = covariant_gaussian_filter(UV, lam0, sig_lam)
        lhs = GUV(PV)
        rhs = [g(V * x * g.adj(V)) for x in GU(P)]
        check("gauge covariance", rel(lhs, rhs), 1e-12)

        # 4. self-adjointness
        a = sum(g.group.inner_product(x, y) for x, y in zip(Q, GU(P)))
        b = sum(g.group.inner_product(x, y) for x, y in zip(GU(Q), P))
        check("self-adjointness |<Q,GP> - <GQ,P>| / |<Q,GP>|", abs(a - b) / abs(a), 1e-12)

    # 5. cosine sanity
    check("cos(P, P) - 1", abs(cosine_similarity(P, P) - 1.0), 1e-14)
    check("cos(P, -P) + 1", abs(cosine_similarity(P, [g(-x) for x in P]) + 1.0), 1e-14)

g.message("ALL PASS" if ok else "SOME FAILED")
