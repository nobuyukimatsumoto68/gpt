#!/usr/bin/env python3
#
# Chunk 1 test (multilevel_fermion_impl_plan_claude.md): N_f = 2 plain Wilson pseudofermion (fermion_setup_claude.py).
#   (a) heat-bath draw: S_f after draw = |eta|^2 returned by draw
#   (b) force vs finite difference (gpt assert_gradient_error, 4th-order difference quotient), random and loaded U
#   (c) gauge invariance of S_f (random gauge transformation of U)
#   (d) force norms rms per link: |F_f| vs |F_g| (Iwasaki), several kappa (first look at the fermion force scale)
# Usage: python3 test_fermion_setup_claude.py [--load cfg] [--L 4] [--kappas 0.12,0.125,0.13]
#
import gpt as g
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fermion_setup_claude import wilson_nf2

g.default.set_verbose("cg", False)
load = g.default.get("--load", None)
L = g.default.get_int("--L", 4)
kappas = [float(x) for x in g.default.get("--kappas", "0.12,0.125,0.13").split(",")]
beta = g.default.get_float("--beta", 2.95)
rng = g.random("test_fermion_setup_claude")

if load is None:
    grid = g.grid([L] * 4, g.double)
    U = g.qcd.gauge.unit(grid)
    rng.normal_element(U, scale=0.3)
    tag = f"random {L}^4 (scale 0.3)"
else:
    U = g.load(load)
    tag = load
S_g = g.qcd.gauge.action.iwasaki(beta)
rms = lambda F: (sum(g.norm2(f) for f in F) / (len(F) * F[0].grid.gsites)) ** 0.5
Fg = S_g.gradient(U, U)
g.message(f"{tag}: |F_g| rms per link {rms(Fg):.4f}")

ok = True
for kappa in kappas:
    fa = wilson_nf2(U, kappa)
    s_draw = fa.draw(U, rng)
    s = fa(U)
    da = abs(s - s_draw) / abs(s)
    g.message(f"kappa={kappa}: (a) S_f = {s:.6f}, draw {s_draw:.6f}, rel diff {da:.1e}")
    ok &= da < 1e-8
    # (b) force vs finite difference
    try:
        fa.action.assert_gradient_error(rng, fa.fields(U), U, 1e-3, 1e-6)
        g.message(f"kappa={kappa}: (b) force vs finite difference OK")
    except AssertionError:
        g.message(f"kappa={kappa}: (b) force vs finite difference FAILED")
        ok = False
    # (c) gauge invariance: U_mu(x) -> V(x) U_mu(x) V(x+mu)^dagger, phi -> V phi (odd sites)
    V = g.mcolor(U[0].grid)
    rng.element(V)
    Ut = [g(V * U[mu] * g.cshift(g.adj(V), mu, 1)) for mu in range(4)]
    Vo = g.lattice(fa.M.F_grid_eo, V.otype)
    g.pick_checkerboard(g.odd, Vo, V)
    phi0 = g.copy(fa.phi)
    fa.phi @= Vo * phi0
    st = fa(Ut)
    fa.phi @= phi0
    dg = abs(st - s) / abs(s)
    g.message(f"kappa={kappa}: (c) gauge invariance rel diff {dg:.1e}")
    ok &= dg < 1e-9
    # (d) force norm
    Ff = fa.gradient(U, U)
    g.message(f"kappa={kappa}: (d) |F_f| rms per link {rms(Ff):.4f}   |F_f|/|F_g| = {rms(Ff) / rms(Fg):.3f}")

g.message("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
