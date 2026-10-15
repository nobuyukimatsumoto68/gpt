#!/usr/bin/env python3
#
# Fourier-acceleration kernel with unmasked layers (nonzero determinant allowed; NM 2026-10-15):
#   A = p_N(T) = prod_j q_j(T),  K = A^2 ~ c / (lambda + m^2),  lambda = -a^2 Delta_perp in [0, 16]
#   q_j(lambda) = (lambda - z_j)(lambda - conj z_j) / |z_j|^2,  Im z_j > 0  ->  q_j > 0 for all real lambda
#   (strictly positive definite K by construction; one quadratic factor = one unmasked layer 1 + alpha T + beta T^2)
# Fit: minimize Var_lambda[log(p^2 (lambda + m^2))] over the N/2 complex roots z_j (start: Chebyshev roots),
# compared with the best real-rooted (det-1-like / heat-kernel) polynomial of the same degree.
# Design: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, entries (13)-(14). numpy only; data + gnuplot script.
# Usage: python3 fa_complex_claude.py [--m2 0.586] [--lmax 16] [--t0 7.5] [--out plot.png]; then gnuplot <out base>.gp
#
import numpy as np
from numpy.polynomial import chebyshev as cheb
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_free_field_claude import nelder_mead  # hand-written Nelder-Mead (Nelder, Mead, Comput. J. 7, 308 (1965))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# LMAX = 16.0
LMAX = float(arg("--lmax", "16.0"))  # 16 (transverse improved, 3 directions) or 64/3 (full improved, 4 directions)
T0 = float(arg("--t0", "7.5"))  # T = T0 - lambda: 7.5 (transverse) or 10 (full)
m2 = float(arg("--m2", str(4 * np.sin(np.pi / 8) ** 2)))
out = arg("--out", "fa_complex_claude.png")
lam = np.linspace(0.0, LMAX, 801)
Ns = [2, 4, 6, 8]


def logdev(p):
    return np.log(p**2 * (lam + m2))


def p_real(r):
    return np.prod([1.0 - lam / ri for ri in r], axis=0)


def p_complex(z):
    return np.prod([np.abs(lam - zj) ** 2 / np.abs(zj) ** 2 for zj in z], axis=0)


def fit_real(N):
    f = lambda s: np.var(logdev(p_real(LMAX + np.exp(np.asarray(s)))))
    best = None
    for start in [np.log(np.geomspace(0.5, 50.0, N)), np.full(N, np.log(5.0))]:
        x, fv, _ = nelder_mead(f, start, step=0.5, tol=1e-12, max_eval=4000)
        if best is None or fv < best[1]:
            best = (x, fv)
    return np.sort(LMAX + np.exp(best[0])), best[1]


def cheb_roots(N):
    c = cheb.chebinterpolate(lambda x: (0.5 * (x + 1.0) * LMAX + m2) ** -0.5, N)
    r = 0.5 * (cheb.chebroots(c) + 1.0) * LMAX
    return np.sort_complex(r[r.imag > 0])


def fit_complex(N):
    z0 = cheb_roots(N)
    x0 = np.concatenate([z0.real, np.log(z0.imag)])  # parametrize Im z = exp(.) > 0
    n = len(z0)

    def unpack(x):
        return x[:n] + 1j * np.exp(x[n:])

    f = lambda x: np.var(logdev(p_complex(unpack(np.asarray(x)))))
    x, fv, _ = nelder_mead(f, x0, step=0.3, tol=1e-14, max_eval=20000)
    for _ in range(3):  # restarts
        x, fv, _ = nelder_mead(f, x, step=0.1, tol=1e-14, max_eval=20000)
    z = unpack(x)
    return np.sort_complex(z), fv, np.var(logdev(p_complex(z0)))


def norm_K(p):
    return p**2 / np.exp(np.mean(logdev(p)))


rows, curves = [], {}
for N in Ns:
    r, vr = fit_real(N)
    z, vc, vcheb = fit_complex(N)
    pr, pc = p_real(r), p_complex(z)
    curves[N] = (norm_K(pr), norm_K(pc))
    rows.append((N, r, vr, z, vc, vcheb, np.exp(np.ptp(logdev(pr))), np.exp(np.ptp(logdev(pc)))))

target = 1.0 / (lam + m2)
print(f"# m^2 = {m2:.4f}; lambda in [0, {LMAX}]; target spans a factor {target.max() / target.min():.1f}")
print("# N | real-rooted (det-1-like): Var, max/min | complex pairs (det != 1): Var, max/min, cond(A) | complex roots z_j")
for N, r, vr, z, vc, vcheb, sr, sc in rows:
    pc = p_complex(z)
    condA = pc.max() / pc.min()
    print(f"N={N}: real Var {vr:.4f}, max/min {sr:.2f} | complex Var {vc:.2e} (Chebyshev start {vcheb:.2e}), max/min {sc:.3f}, cond(A) {condA:.2f} | z = {np.array2string(z, precision=3)}")

# ---------------- data + gnuplot ----------------
base = os.path.splitext(out)[0]
cols, hdr = [lam, target / np.exp(np.mean(np.log(target * (lam + m2))))], "lambda target"
for N in Ns:
    Kr, Kc = curves[N]
    cols += [Kr, Kc, Kr * (lam + m2), Kc * (lam + m2)]
    hdr += f" Kreal_N{N} Kcplx_N{N} ratio_real_N{N} ratio_cplx_N{N}"
np.savetxt(base + ".dat", np.array(cols).T, header=hdr)
with open(base + "_roots.dat", "wt") as f:
    f.write("# N re im kind(0 = real-rooted, 1 = complex pairs)\n")
    for N, r, vr, z, vc, vcheb, sr, sc in rows:
        for x in r:
            f.write(f"{N} {x:.6f} 0.0 0\n")
        for x in z:
            f.write(f"{N} {x.real:.6f} {x.imag:.6f} 1\n{N} {x.real:.6f} {-x.imag:.6f} 1\n")

col = lambda N, k: 3 + 4 * Ns.index(N) + k  # k: 0 Kreal, 1 Kcplx, 2 ratio real, 3 ratio cplx
lines_K = ", \\\n     ".join(
    [f'"" u 1:{col(N, 1)} w l lw 2 lc {i + 1} t "N={N} complex pairs"' for i, N in enumerate(Ns)]
    + [f'"" u 1:{col(N, 0)} w l dt 2 lc {i + 1} t "N={N} real roots"' for i, N in enumerate(Ns)]
)
lines_R = ", \\\n     ".join(
    [f'"" u 1:{col(N, 3)} w l lw 2 lc {i + 1} t "N={N} complex pairs"' for i, N in enumerate(Ns)]
    + [f'"" u 1:{col(N, 2)} w l dt 2 lc {i + 1} t "N={N} real roots"' for i, N in enumerate(Ns)]
)
gp_body = f"""set multiplot layout 1,3
set grid
set title "K(lambda) vs target c/(lambda+m^2), m^2 = {m2:.3f}"
set xlabel "lambda = -a^2 Delta_perp (free-field spectrum)"
set ylabel "K (normalized)"
set logscale y
set key top right
plot "{base}.dat" u 1:2 w l lw 4 lc rgb "black" t "target", \\
     {lines_K}
set title "K(lambda) (lambda+m^2) / c  (flat = exact Fourier acceleration)"
set ylabel "K (lambda + m^2), normalized"
set yrange [0.2:5]
plot 1 w l lw 4 lc rgb "black" t "target", "{base}.dat" u 1:{col(Ns[0], 3)} w l lw 2 lc 1 notitle, \\
     {lines_R}
unset logscale y
unset yrange
set title "roots of p_N(lambda); spectrum [0,16] shaded; target pole at -m^2"
set xlabel "Re lambda"
set ylabel "Im lambda"
set object 1 rect from 0,graph 0 to 16,graph 1 fc rgb "#eeeeee" fs solid behind
set arrow 1 from {-m2:.4f},graph 0 to {-m2:.4f},graph 1 nohead dt 3 lc rgb "black"
set xrange [-2:40]
set key top right
plot for [N in "2 4 6 8"] "{base}_roots.dat" u ($1==N && $4==1 ? $2 : 1/0):3 w p pt 7 ps 1.5 lc (N/2) t "N=".N." complex pairs", \\
     for [N in "2 4 6 8"] "{base}_roots.dat" u ($1==N && $4==0 ? $2 : 1/0):3 w p pt 6 ps 1.5 lc (N/2) t "N=".N." real roots (beyond xrange for N>2)"
unset multiplot
"""
# vector output (pdf) for NM, plus a png preview
gp = (f'set terminal pdfcairo size 11in,3.4in font ",9" noenhanced\nset output "{base}.pdf"\n' + gp_body
      + f'reset\nset terminal pngcairo size 1800,560 font ",11" noenhanced\nset output "{base}.png"\n' + gp_body)
open(base + ".gp", "wt").write(gp)
print(f"# wrote {base}.dat, {base}_roots.dat, {base}.gp")

# (alpha_j, beta_j) of 1 + alpha T + beta T^2 for each complex pair, T = T0 - lambda
with open(base + "_coeffs.py", "wt") as f:
    f.write(f"# FA polynomial coefficients from fa_complex_claude.py: m2 = {m2}, lambda in [0, {LMAX}], T = {T0} - lambda\n")
    f.write("COEFFS = {\n    0: [],\n")
    for N, r, vr, z, vc, vcheb, sr, sc in rows:
        co = []
        for x in sorted(z, key=lambda c: c.real):
            sft = T0 - x.real
            c0 = sft * sft + x.imag**2
            co.append((float(round(-2 * sft / c0, 6)), float(round(1 / c0, 6))))
        f.write(f"    {N}: {co},\n")
    f.write("}\n")
print(f"# wrote {base}_coeffs.py")
