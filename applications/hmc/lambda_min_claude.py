#!/usr/bin/env python3
#
# Lowest eigenvalues of the improved covariant Laplacian -Delta_imp[U] that the FA polynomial kernel uses
# (NM 2026-10-16: set the FA regulator per ensemble, m^2 = <lambda_min(-Delta_imp[U])>; plan entry (24)).
#   -Delta_imp = T0 - T,  T = poly_kernel_claude.improved_hops ((4/3) dist-1 - (1/12) dist-2, straight Wilson lines)
#   longitudinal = True : T0 = 10,  free symbol lambda = sum_nu [(4/3)(2 - 2 cos k_nu) - (1/12)(2 - 2 cos 2 k_nu)], in [0, 64/3]
#   longitudinal = False: T0 = 15/2, sum over nu != mu only, lambda in [0, 16]
# T acts on each P_mu separately (block diagonal in mu), so the spectrum is the union over mu of single-field problems.
# Method: implicitly restarted Lanczos (gpt g.algorithms.eigen.irl; C. Lehner, after T. Blum, T. Izubuchi, C. Jung,
#   R. Arthur) on a Chebyshev polynomial of -Delta_imp that maps [--cheb_low, lambda_max] to [-1, 1] (low modes
#   amplified; even order so they come out large positive); eigenvalues of -Delta_imp itself from Rayleigh quotients.
#   Chebyshev acceleration of Lanczos: e.g. Y. Saad, "Numerical Methods for Large Eigenvalue Problems" (2011), ch. 8.
# Note: single-vector Lanczos resolves one vector per exactly degenerate eigenspace (free field: one copy per distinct
#   lambda); on interacting configurations the degeneracies are lifted.
# su(3) restriction: iterate with  P L + lam_max (1 - P)  (P = traceless-hermitian projector); the trace part of a 3x3
#   field is transported trivially (free spectrum, exact zero mode) and would otherwise be picked up from rounding noise.
# Checks (Mac, 2026-10-16): free 8^4 full: 0.61438192 = closed form; thermalized 4^3x8 (Iwasaki 2.95): full 1.477118
#   (same for --cheb_low 2/order 30 and 4/50, residual 1e-29), transverse 0.834494.  --free 1 --longitudinal 0 breaks
#   down (Krylov exhaustion: only 35 distinct free values); not a problem on interacting configurations.
# Usage: python3 lambda_min_claude.py --load cfg [--longitudinal 1] [--nev 4] [--cheb_low 2.0] [--cheb_order 30]
#        python3 lambda_min_claude.py --free 1 --L 8 [--longitudinal 1]       (free-field check vs the closed form)
#   output: per mu the lowest --nev eigenvalues; "lambda_min <value>" = min over mu (grep-able)
#
import gpt as g
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poly_kernel_claude import improved_hops
from path_smear_claude import path_layer

load = g.default.get("--load", None)
free = bool(g.default.get_int("--free", 0))
L = g.default.get_int("--L", 8)
longitudinal = bool(g.default.get_int("--longitudinal", 1))
nev = g.default.get_int("--nev", 4)
cheb_low = g.default.get_float("--cheb_low", 2.0)
cheb_order = g.default.get_int("--cheb_order", 30)
resid = g.default.get_float("--resid", 1e-8)
T0 = 10.0 if longitudinal else 7.5
lam_max = 64.0 / 3.0 if longitudinal else 16.0
if cheb_order % 2:
    cheb_order += 1  # even: T_n(y) > 0 for y < -1

if free:
    grid = g.grid([L] * 4, g.double)
    U = [g.identity(g.mcolor(grid)) for mu in range(4)]
    tag = f"free {L}^4"
else:
    U = g.load(load)
    tag = load
grid = U[0].grid
ones = g.complex(grid)
ones[:] = 1.0


# closed-form free-field spectrum: distinct lambda values, ascending
def free_spectrum(dims, mu):
    ks = [2.0 * np.pi * np.arange(n) / n for n in dims]
    s = [(4.0 / 3.0) * (2 - 2 * np.cos(k)) - (1.0 / 12.0) * (2 - 2 * np.cos(2 * k)) for k in ks]
    lam = np.zeros(1)
    for nu in range(4):
        if nu == mu and not longitudinal:
            continue
        lam = np.add.outer(lam, s[nu]).ravel()
    return np.unique(np.round(lam, 10))


rng = g.random("lambda_min_claude")
c = g.algorithms.polynomial.chebyshev({"low": cheb_low, "high": lam_max * 1.02, "order": cheb_order})
irl = g.algorithms.eigen.irl(
    {
        "Nk": max(2 * nev, 16),
        "Nstop": nev,
        "Nm": max(2 * nev, 16) + 16,
        "resid": resid,
        "betastp": 0.0,
        "maxiter": 40,
        "Nminres": 0,
        "sort_eigenvalues": lambda x: sorted(x),
    }
)

all_ev = []
t_start = time.time()
# longitudinal: improved_hops(mu, True) hops along ALL nu for every mu -> T_mu identical, solve mu = 0 only
mus = [0] if longitudinal else range(4)
# for mu in range(4):
for mu in mus:
    layer = path_layer(U, mu, improved_hops(4, mu, longitudinal), ones, ones, 1.0, f"T_imp mu={mu}", cache=True)

    def lap(dst, src):
        dst @= T0 * src - layer.hop(src)  # -Delta_imp
        # keep the iteration in su(3): the trace (u(1)) part of a 3x3 matrix field is invariant under the adjoint
        # transport, so it sees the FREE Laplacian (exact zero mode); rounding noise there is amplified ~1e8 by the
        # Chebyshev filter and IRL would return it.  Use  P L + lam_max (1 - P),  P = traceless-hermitian projector:
        # the non-su(3) sector is lifted to the top of the spectrum (a bare P L would put it at 0, the most amplified).
        # dst @= g.qcd.gauge.project.traceless_hermitian(dst)
        Pdst = g.qcd.gauge.project.traceless_hermitian(dst)
        dst @= Pdst + lam_max * (src - g.qcd.gauge.project.traceless_hermitian(src))
        if free:
            # free check only: lift the U = 1 zero modes (constant algebra fields) to lam_max so the lowest
            # NONZERO lambda is resolved:  -Delta_imp + lam_max Pi_0,  Pi_0 v = (1/V) sum_x v(x)
            c0 = g.lattice(src)
            # c0[:] = g.sum(src) / grid.gsites
            c0[:] = g.sum(g.qcd.gauge.project.traceless_hermitian(src)) / grid.gsites  # su(3) part only (else 2 lam_max > Chebyshev high)
            dst += lam_max * c0

    op = g.matrix_operator(lap)
    start = g.group.cartesian(U[0])
    rng.normal_element(start)
    t0 = time.time()
    evec, _ = irl(c(op), start)
    ev, eps2 = g.algorithms.eigen.evals(op, evec, real=True)
    order = np.argsort(ev)
    ev = np.array(ev)[order]
    eps2 = np.array(eps2)[order]
    all_ev.append(ev)
    g.message(f"mu={mu}: lowest eigenvalues of -Delta_imp {np.array2string(ev, precision=6)}  "
              f"max |M v - l v|^2 {eps2.max():.1e}  ({time.time() - t0:.1f} s, {irl.napply} Chebyshev applications)")
    if free:
        ref = free_spectrum(grid.fdimensions, mu)
        g.message(f"mu={mu}: closed form (distinct) {np.array2string(ref[:nev], precision=6)}")

lmin = min(e[0] for e in all_ev)
evs = np.sort(np.concatenate(all_ev))
g.message(f"{tag}: longitudinal={longitudinal} cheb_low={cheb_low} order={cheb_order}: lowest over mu {np.array2string(evs[:nev], precision=6)}")
g.message(f"lambda_min {lmin:.8f}  ({time.time() - t_start:.1f} s total)")
if free:
    k = 2 * np.pi / L
    g.message(f"free check: lowest nonzero = (4/3)(2-2cos(2pi/L)) - (1/12)(2-2cos(4pi/L)) = "
              f"{(4 / 3) * (2 - 2 * np.cos(k)) - (1 / 12) * (2 - 2 * np.cos(2 * k)):.8f} (zero modes lifted by lam_max Pi_0)")
