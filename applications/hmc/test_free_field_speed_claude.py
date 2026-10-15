#!/usr/bin/env python3
#
# Validation of free_field_speed_claude.py (LO free-field mode-speed model).
#   v1  : numpy real-space scalar layers == gpt path_flow at U = unit (single generator)
#   v1b : Fourier-orbit layers == numpy real-space layers (via FFT)
#   v2  : theta = 0 -> omega^2 = nonzero eigenvalues of h(k)
#   v3  : h(k) == numerical Hessian of gpt's Wilson / Iwasaki action (force linear response), up to normalization
# Usage: python3 test_free_field_speed_claude.py
#
import gpt as g
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from free_field_speed_claude import free_field_model, real_space_flow, ND
from path_smear_claude import path_flow, kernel_builder

L = 8
ok = True


def check(name, val, tol_):
    global ok
    passed = val < tol_
    ok = ok and passed
    g.message(f"[{'PASS' if passed else 'FAIL'}] {name}: {val:.3e} (tol {tol_:.0e})")


grid = g.grid([L] * ND, g.double)
rng = np.random.default_rng(1)
Uunit = g.qcd.gauge.unit(grid)
P = g.group.cartesian(Uunit)
gen = P[0].otype.generators(np.complex128)[0].array
tr00 = np.trace(gen @ gen).real
coor = g.coordinates(P[0])
idx = tuple(coor[:, d] for d in range(ND))


def to_gpt(phi):  # phi: (ND, L,L,L,L) real -> list of algebra lattices phi_mu * T_0
    Q = g.group.cartesian(Uunit)
    for mu in range(ND):
        Q[mu][coor] = (phi[mu][idx][:, None, None] * gen[None]).astype(np.complex128)
    return Q


def from_gpt(Q):
    phi = np.zeros((ND,) + (L,) * ND)
    for mu in range(ND):
        v = Q[mu][coor]
        phi[mu][idx] = np.real(np.einsum("nij,ji->n", v, gen)) / tr00
    return phi


spec = [("T1", 0.07), ("R", 0.04), ("C", 0.05)] * 2
phi = rng.normal(size=(ND,) + (L,) * ND)
rs = real_space_flow(L, spec)

# v1
A_gpt = path_flow(Uunit, spec, kernel_builder(grid))
d1 = from_gpt(A_gpt.forward(to_gpt(phi)))
d2 = rs.forward(phi)
check("v1 real-space numpy vs gpt path_flow at U=1", np.linalg.norm(d1 - d2) / np.linalg.norm(d2), 1e-13)
check("v1 numpy inverse", np.linalg.norm(rs.forward(d2, -1) - phi) / np.linalg.norm(phi), 1e-13)

# v1b
model = free_field_model(L)
ft_in = np.fft.fftn(phi, axes=range(1, ND + 1)) / L**ND
ft_out = np.fft.fftn(d2, axes=range(1, ND + 1)) / L**ND
err = 0.0
for n0 in model.reps:
    k = model.momenta(n0)
    nidx = tuple(np.rint(k[:, d] * L / (2 * np.pi)).astype(int) % L for d in range(ND))
    for mu in range(ND):
        A = model.A_orbit(spec, mu, k)
        err = max(err, np.abs(A @ ft_in[mu][nidx] - ft_out[mu][nidx]).max())
check("v1b Fourier-orbit vs real space (max abs)", err / np.abs(ft_out).max(), 1e-12)

# v2
w2 = model.omega2([])
ref = []
for n0 in model.reps:
    for kk in model.momenta(n0):
        ev = np.linalg.eigvalsh(model.hessian(kk))
        ref += list(ev)
ref = np.sort(np.array(ref))
ref = ref[ref > 1e-9 * ref.max()]
check("v2 theta=0: omega^2 == eig h(k) (count)", abs(len(ref) - len(w2)), 0.5)
check("v2 theta=0: omega^2 == eig h(k) (values)", np.abs(np.sort(w2) - ref).max(), 1e-10)
g.message(f"physical modes: {len(w2)} of {ND * L**ND}; omega^2 range at theta=0: [{w2.min():.4f}, {w2.max():.4f}]")

# v3: linear response of gpt's force at U = exp(i eps a T_0)
for name, c1, S in [("Wilson", 0.0, g.qcd.gauge.action.wilson(6.0)), ("Iwasaki", -0.331, g.qcd.gauge.action.iwasaki(6.0))]:
    eps = 1e-5
    a = rng.normal(size=(ND,) + (L,) * ND)
    Aalg = to_gpt(eps * a)
    U = [g(g.group.compose(x, u)) for x, u in zip(Aalg, Uunit)]
    F = from_gpt(S.gradient(U, U)) / eps
    m = free_field_model(L, c1)
    at = np.fft.fftn(a, axes=range(1, ND + 1))
    resp = {}
    for label, conj in [("h", False), ("h*", True)]:
        out = np.zeros_like(at)
        for n in np.ndindex(*(L,) * ND):
            kk = 2 * np.pi * np.array(n) / L
            h = m.hessian(kk)
            h = h.conj() if conj else h
            out[(slice(None),) + n] = h @ at[(slice(None),) + n]
        resp[label] = np.real(np.fft.ifftn(out, axes=range(1, ND + 1)))
    for label, r in resp.items():
        c = np.sum(F * r) / np.sum(r * r)
        res = np.linalg.norm(F - c * r) / np.linalg.norm(F)
        g.message(f"v3 {name}: F vs c * ({label} a): c = {c:.6f}, residual = {res:.3e}")
    best = min(np.linalg.norm(F - (np.sum(F * r) / np.sum(r * r)) * r) / np.linalg.norm(F) for r in resp.values())
    check(f"v3 {name}: analytic Hessian matches gpt force (best convention)", best, 1e-4)

g.message("ALL PASS" if ok else "SOME FAILED")
