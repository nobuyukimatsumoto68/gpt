#!/usr/bin/env python3
#
# Plot momentum decorrelation C(t), C_j(t) along single trajectories (decorrelation_claude.py output) for several
# kernels, with first zero crossings marked. Writes a gnuplot script producing a vector PDF (+ PNG preview).
# Input format (per file): "# ..." header lines, then rows: traj acc dH k t C C_j1 C_j2 ...
# Usage: python3 plot_uturn_claude.py --dir <dir> --out <base>   (files fp_blk_{poly,e0,n2,n4}_claude.dat in <dir>)
#
import numpy as np
import sys, os


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


d = arg("--dir", "/Users/nobuyukimatsumoto/Documents/genesis/run_rmhmc_claude/aurora_fp")
base = arg("--out", os.path.join(d, "uturn_first_pass_claude"))
kernels = [("poly", "poly N=8 FA"), ("e0", "eta=0"), ("n2", "masked n=2"), ("n4", "masked n=4")]
shells = ["C (all modes)", "C_1 (lambda0=0.59)", "C_2 (lambda0=2.0)", "C_4 (lambda0=4.0)"]


def first_zero(t, c):
    for i in range(1, len(c)):
        if c[i - 1] > 0 >= c[i]:
            return t[i - 1] + (t[i] - t[i - 1]) * c[i - 1] / (c[i - 1] - c[i])
    return np.nan


zeros = {}
for key, label in kernels:
    a = np.loadtxt(os.path.join(d, f"fp_blk_{key}_claude.dat"), comments="#")
    a = a[a[:, 0] == a[0, 0]]  # first trajectory
    t = a[:, 4]
    np.savetxt(f"{base}_{key}.dat", np.column_stack([t] + [a[:, 5 + i] for i in range(4)]), header="t C C1 C2 C4")
    zeros[key] = [first_zero(t, a[:, 5 + i]) for i in range(4)]
with open(f"{base}_zeros.dat", "wt") as f:
    f.write("# shell_index " + " ".join(k for k, _ in kernels) + "\n")
    for i in range(4):
        f.write(f"{i} " + " ".join(f"{zeros[k][i]:.4f}" for k, _ in kernels) + "\n")
for key, label in kernels:
    print(f"{label:14s} first zeros (C, C1, C2, C4): " + ", ".join(f"{z:.3f}" for z in zeros[key]))

panels = []
for i, sh in enumerate(shells):
    lines = ", \\\n     ".join(f'"{base}_{k}.dat" u 1:{i + 2} w l lw 2 lc {j + 1} t "{lab}"' for j, (k, lab) in enumerate(kernels))
    panels.append(f'set title "{sh}"\nplot 0 w l lc rgb "#999999" notitle, \\\n     {lines}\n')
body = (
    'set multiplot layout 1,5\nset grid\nset xlabel "MD time t"\nset ylabel "cosine similarity"\nset yrange [-1:1]\nset xrange [0:3]\nset key bottom left\n'
    + "".join(panels)
    + 'set title "first zero crossing (U-turn proxy)"\nset xlabel ""\nset ylabel "t"\nset yrange [0:2]\nset xrange [-0.5:3.5]\n'
    + 'set xtics ("C" 0, "C_1" 1, "C_2" 2, "C_4" 3)\nset key top right\n'
    + 'plot ' + ", \\\n     ".join(f'"{base}_zeros.dat" u 1:{j + 2} w lp lw 2 pt 7 lc {j + 1} t "{lab}"' for j, (k, lab) in enumerate(kernels)) + "\n"
    + "unset multiplot\n"
)
gp = (
    f'set terminal pdfcairo size 15in,3.2in font ",9" noenhanced\nset output "{base}.pdf"\n' + body
    + f'reset\nset terminal pngcairo size 2200,480 font ",11" noenhanced\nset output "{base}.png"\n' + body
)
open(base + ".gp", "wt").write(gp)
print(f"# wrote {base}.gp (gnuplot {base}.gp -> {base}.pdf, .png)")

# ---- second layout (NM): one panel per kernel, ordered by smearing (no smearing first); curves = momentum filters ----
order = [("e0", "no smearing (eta=0)"), ("n2", "masked n=2, eta=0.05 (n*eta=0.1)"), ("n4", "masked n=4, eta=0.05 (n*eta=0.2)"), ("poly", "poly N=8 Fourier acceleration")]
fl = ["C (all modes)", "C_1  lambda0=0.59", "C_2  lambda0=2.0", "C_4  lambda0=4.0"]
pk = []
for key, title in order:
    zs = zeros[key]
    zt = ", ".join(f"{z:.2f}" for z in zs)
    lines = ", \\\n     ".join(f'"{base}_{key}.dat" u 1:{i + 2} w l lw 2 lc {i + 1} t "{fl[i]}"' for i in range(4))
    pk.append(f'set title "{title}\\nfirst zeros: {zt}"\nplot 0 w l lc rgb "#999999" notitle, \\\n     {lines}\n')
body2 = (
    'set multiplot layout 1,4\nset grid\nset xlabel "MD time t"\nset ylabel "cosine similarity"\nset yrange [-0.4:1]\nset xrange [0:3]\nset key top right\n'
    + "".join(pk) + "unset multiplot\n"
)
gp2 = (
    f'set terminal pdfcairo size 13in,3.4in font ",9" noenhanced\nset output "{base}_by_kernel.pdf"\n' + body2
    + f'reset\nset terminal pngcairo size 2000,520 font ",11" noenhanced\nset output "{base}_by_kernel.png"\n' + body2
)
open(base + "_by_kernel.gp", "wt").write(gp2)
print(f"# wrote {base}_by_kernel.gp")
