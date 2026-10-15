#!/usr/bin/env python3
#
# Polynomial Fourier-acceleration kernel (free field, unmasked analytic model):
#   K(lambda) = p_N(lambda)^2 ~ c / (lambda + m^2),  lambda = -a^2 Delta_perp in [0, 16] (improved transverse Laplacian)
#   p_N(lambda) = prod_l (1 + eta_l T_imp),  T_imp = 15/2 - lambda  ->  root r_l = 15/2 + 1/eta_l
# (a) best real-rooted p_N (roots r_l > lambda_max, i.e. 0 < eta_l < 2/17), objective Var_lambda[log(p^2 (lambda + m^2))]
# (b) reference: unconstrained Chebyshev interpolant of (lambda + m^2)^{-1/2} (complex roots allowed)
# Design: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, "E3(a') target fixed". numpy only; writes data + a gnuplot script.
# Chebyshev interpolation: e.g. L. N. Trefethen, Approximation Theory and Approximation Practice (SIAM, 2013).
# Usage: python3 fa_polynomial_claude.py [--m2 0.586] [--out plot.png]; then gnuplot <out base>.gp
#
import numpy as np
from numpy.polynomial import chebyshev as cheb
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_free_field_claude import nelder_mead  # noqa: E402  (hand-written Nelder-Mead)


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


LMAX = 16.0
m2 = float(arg("--m2", str(4 * np.sin(np.pi / 8) ** 2)))
out = arg("--out", "fa_spectrum_claude.png")
lam = np.linspace(0.0, LMAX, 801)


def logdev(pvals):
    return np.log(pvals**2 * (lam + m2))


def fit_real_roots(N):
    # r_l = LMAX + exp(s_l)
    def f(s):
        r = LMAX + np.exp(np.asarray(s))
        p = np.prod([1.0 - lam / ri for ri in r], axis=0)
        return np.var(logdev(p))

    best = None
    for start in [np.log(np.geomspace(0.5, 50.0, N)), np.log(np.geomspace(2.0, 200.0, N)), np.full(N, np.log(5.0))]:
        x, fv, _ = nelder_mead(f, start, step=0.5, tol=1e-12, max_eval=4000)
        if best is None or fv < best[1]:
            best = (x, fv)
    r = np.sort(LMAX + np.exp(best[0]))
    return r, best[1]


def cheb_approx(N):
    f = lambda x: (0.5 * (x + 1.0) * LMAX + m2) ** -0.5
    c = cheb.chebinterpolate(f, N)
    roots_x = cheb.chebroots(c)
    roots = 0.5 * (roots_x + 1.0) * LMAX
    p = cheb.chebval(2.0 * lam / LMAX - 1.0, c)
    return roots, p


rows = []
curves = {}
for N in [2, 4, 6, 8]:
    r, v = fit_real_roots(N)
    p = np.prod([1.0 - lam / ri for ri in r], axis=0)
    dev = logdev(p)
    c = np.exp(np.mean(dev))
    K = p**2 / c  # normalized so that mean log(K (lambda + m2)) = 0
    eta = 1.0 / (r - 7.5)
    roots_c, pc = cheb_approx(N)
    Kc = pc**2 / np.exp(np.mean(logdev(pc))) if np.all(pc > 0) else None
    rows.append((N, r, eta, v, np.exp(dev.max() - dev.min()), roots_c, (np.var(logdev(pc)) if Kc is not None else np.nan)))
    curves[N] = (K, Kc)

target = 1.0 / (lam + m2)
print(f"# m^2 = {m2:.4f}, lambda in [0, {LMAX}]; target K = c/(lambda + m^2) spans factor {target.max() / target.min():.1f}")
for N, r, eta, v, spread, rc, vc in rows:
    print(f"N = {N}: real roots r_l = {np.array2string(r, precision=2)}")
    print(f"        eta_l = {np.array2string(eta, precision=4)}   Var[log K(lam+m2)] = {v:.4f}, max/min of K(lam+m2) = {spread:.2f}")
    print(f"        Chebyshev (unconstrained) roots = {np.array2string(rc, precision=2)}, Var = {vc:.4f}")

# ---------------- data + gnuplot ----------------
base = os.path.splitext(out)[0]
norm_t = target / np.exp(np.mean(np.log(target * (lam + m2))))
cols = [lam, norm_t]
hdr = "lambda target"
for N in [2, 4, 6, 8]:
    K, Kc = curves[N]
    cols += [K, Kc if Kc is not None else np.full_like(lam, np.nan)]
    hdr += f" K_real_N{N} K_cheb_N{N}"
np.savetxt(base + ".dat", np.array(cols).T, header=hdr)
with open(base + "_roots.dat", "wt") as f:
    f.write("# N re im kind(0=real-rooted fit, 1=Chebyshev)\n")
    for N, r, eta, v, spread, rc, vc in rows:
        for x in r:
            f.write(f"{N} {x:.6f} 0.0 0\n")
        for x in rc:
            f.write(f"{N} {x.real:.6f} {x.imag:.6f} 1\n")
gp_body = f"""set multiplot layout 1,2
set title "K(lambda) vs target c/(lambda+m^2), m^2 = {m2:.3f}"
set xlabel "lambda = -a^2 Delta_perp (free-field spectrum)"
set ylabel "K (normalized)"
set logscale y
set grid
set key top right
plot "{base}.dat" u 1:2 w l lw 4 lc rgb "black" t "target", \\
     "" u 1:3 w l lw 2 lc 1 t "N=2 real roots", "" u 1:4 w l dt 2 lc 1 t "N=2 Chebyshev", \\
     "" u 1:5 w l lw 2 lc 2 t "N=4 real roots", "" u 1:6 w l dt 2 lc 2 t "N=4 Chebyshev", \\
     "" u 1:7 w l lw 2 lc 3 t "N=6 real roots", "" u 1:8 w l dt 2 lc 3 t "N=6 Chebyshev", \\
     "" u 1:9 w l lw 2 lc 4 t "N=8 real roots", "" u 1:10 w l dt 2 lc 4 t "N=8 Chebyshev"
unset logscale y
set title "roots of p_N(lambda) (zeros of K); spectrum [0,16] shaded; target pole at -m^2"
set xlabel "Re lambda"
set ylabel "Im lambda"
set object 1 rect from 0,graph 0 to 16,graph 1 fc rgb "#eeeeee" fs solid behind
set arrow 1 from {-m2:.4f},graph 0 to {-m2:.4f},graph 1 nohead dt 3 lc rgb "black"
set key top left
plot for [N in "2 4 6 8"] "{base}_roots.dat" u ($1==N && $4==0 ? $2 : 1/0):3 w p pt 7 ps 1.4 lc (N/2) t "N=".N." real-rooted", \\
     for [N in "2 4 6 8"] "{base}_roots.dat" u ($1==N && $4==1 ? $2 : 1/0):3 w p pt 6 ps 1.4 lc (N/2) t "N=".N." Chebyshev"
unset multiplot
"""
# vector output (pdf) for NM, plus a png preview
gp = (f'set terminal pdfcairo size 9in,3.4in font ",9" noenhanced\nset output "{base}.pdf"\n' + gp_body
      + f'reset\nset terminal pngcairo size 1400,560 font ",11" noenhanced\nset output "{base}.png"\n' + gp_body)
open(base + ".gp", "wt").write(gp)
print(f"# wrote {base}.dat, {base}_roots.dat, {base}.gp  (run: gnuplot {base}.gp -> {base}.png)")
