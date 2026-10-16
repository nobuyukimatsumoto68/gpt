#!/usr/bin/env python3
#
# Periodically replicate a gauge configuration onto a larger lattice: U'_mu(x) = U_mu(x mod L_small).
# Used to start a 16^4 ensemble from a thermalized 8^4 config (2^4 copies); the copies satisfy the
# small-lattice periodicity, which the subsequent re-thermalization must wash out.
# Plan: genesis/volume_N_scaling_impl_plan_claude.md ("Revised order", step 1).
#
# Usage: python3 replicate_cfg_claude.py --load small.cfg --L 16,16,16,16 --save big.cfg
#        (each entry of --L must be a multiple of the small lattice extent)
#
import gpt as g
import numpy as np

load = g.default.get("--load", None)
L = [int(x) for x in g.default.get("--L", "16,16,16,16").split(",")]
save = g.default.get("--save", None)
assert load is not None and save is not None, "need --load and --save"

U0 = g.load(load)
L0 = U0[0].grid.gdimensions
assert len(L) == len(L0) and all(l % l0 == 0 for l, l0 in zip(L, L0)), f"--L {L} is not a multiple of {L0}"

grid = g.grid(L, U0[0].grid.precision)
U = g.qcd.gauge.unit(grid)
x = g.coordinates(U[0])            # local coordinates on the big lattice
x0 = np.mod(x, np.array(L0)).astype(np.int32)  # corresponding site of the small lattice (cgpt wants int32)
for mu in range(len(U)):
    U[mu][x] = U0[mu][x0]

p0, p = g.qcd.gauge.plaquette(U0), g.qcd.gauge.plaquette(U)
g.message(f"replicated {load} {L0} -> {L}: plaquette {p0:.10f} -> {p:.10f} (diff {p - p0:.1e})")
assert abs(p - p0) < 1e-10, "plaquette changed: replication is wrong"
g.save(save, U, g.format.nersc())
g.message(f"saved {save}")
