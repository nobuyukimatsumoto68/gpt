#
# Generic gauge-covariant coupling layers for SU(N) link momenta, built from parallel-transport paths.
# Design and equations: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, "Extended smearing kernels".
# Sources: kernel idea: R. Abbott, ..., N. Matsumoto et al. (draft TrainableKernel-2.pdf), Eqs. (1)-(3);
#   coupling layers: L. Dinh et al., arXiv:1410.8516, arXiv:1605.08803;
#   RMHMC: M. Girolami, B. Calderhead, J. R. Stat. Soc. B 73 (2011) 123.
#
# Conventions (gpt): U_\nu(x) <- exp(i W_\nu(x)) U_\nu(x);  P_\mu(x) -> V(x) P_\mu(x) V^\dagger(x).
# A path from x is a list of steps [(\nu, s), ...], s = +1 / -1:
#   step (\nu, +1) at site z contributes U_\nu(z) and moves z -> z + \nu,
#   step (\nu, -1) at site z contributes U_\nu^\dagger(z - \nu) and moves z -> z - \nu.
# W(x) = V_1 V_2 ... V_m (transport from the end point y = x + d back to x).
# Hop term:  t(x) = W(x) P_{\mu*}(x + d) W^\dagger(x).
# Layer:     P_{\mu*} <- P_{\mu*} + \eta \Pi_w \sum_{hops} w_h t_h      (\Pi_w: write mask)
# Hop lists are closed under path reversal, so T is hermitian; every hop maps the write region into
# the read region (complement), so the layer never reads what it writes:
#   inverse = same layer with -\eta, adjoint = same layer with write / read masks swapped.
# Force of a layer (adjoint state Rt = \Pi_w R[\mu*], forward state X = P_{\mu*} before the layer):
#   V_k = U_l(z):          F_l(z) += i \eta w [M_k, At_k],    M_k = V_k B_k X(.+d) B_k^\dagger V_k^\dagger, At_k = A_k^\dagger Rt A_k
#   V_k = U_l(z)^\dagger:  F_l(z) += -i \eta w [M'_k, At'_k], M'_k = B_k X(.+d) B_k^\dagger, At'_k = V_k^\dagger A_k^\dagger Rt A_k V_k
# with A_k = V_1 ... V_{k-1}, B_k = V_{k+1} ... V_m, all as fields at x, shifted to the link site z.
#
import gpt as g
import numpy as np


def shift(f, disp):
    # shift(f, o)(x) = f(x + o)
    for mu, d in enumerate(disp):
        if d != 0:
            f = g.cshift(f, mu, d)
    return f


def unit_vec(nd, mu, s=1):
    return tuple(s if i == mu else 0 for i in range(nd))


def add_vec(a, b):
    return tuple(x + y for x, y in zip(a, b))


def neg_vec(a):
    return tuple(-x for x in a)


class path_layer:
    # hops: list of (path, weight); write / read: complex lattice masks (0/1), read = complement of write
    def __init__(self, U, mu_star, hops, write, read, eta, tag="", cache=False):
        self.U = U
        self.nd = len(U)
        self.mu = mu_star
        self.hops = hops
        self.write = write
        self.read = read
        self.eta = eta
        self.tag = tag
        # cache = True: memoize path factors / transports / partial products. ONLY valid while U is not modified
        # (the caller must create a new layer, or call clear_cache(), when U changes). Default False = original behaviour.
        self.cache = {} if cache else None

    def clear_cache(self):
        if self.cache is not None:
            self.cache = {}

    def factors(self, path):
        if self.cache is None:
            return self.factors_uncached(path)
        key = ("fac", tuple(path))
        if key not in self.cache:
            self.cache[key] = self.factors_uncached(path)
        return self.cache[key]

    # W = V_1 ... V_m (transport of the whole path), memoized when caching
    def transport(self, path):
        if self.cache is None:
            fac, d = self.factors(path)
            return self.product([f[3] for f in fac], self.U[0])
        key = ("W", tuple(path))
        if key not in self.cache:
            fac, d = self.factors(path)
            self.cache[key] = self.product([f[3] for f in fac], self.U[0])
        return self.cache[key]

    # (A_k, B_k) = (V_1 ... V_{k-1}, V_{k+1} ... V_m), memoized when caching
    def partials(self, path, k):
        key = ("AB", tuple(path), k)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        fac, d = self.factors(path)
        Ak = self.product([f[3] for f in fac[:k]], self.U[0])
        Bk = self.product([f[3] for f in fac[k + 1 :]], self.U[0])
        if self.cache is not None:
            self.cache[key] = (Ak, Bk)
        return Ak, Bk

    # factors V_k of a path, with the offset o_k of the site where step k starts, and end point d
    def factors_uncached(self, path):
        o = tuple(0 for i in range(self.nd))
        fac = []
        for nu, s in path:
            if s > 0:
                fac.append((nu, +1, o, shift(self.U[nu], o)))
                o = add_vec(o, unit_vec(self.nd, nu))
            else:
                o = add_vec(o, unit_vec(self.nd, nu, -1))
                fac.append((nu, -1, o, shift(g.adj(self.U[nu]), o)))  # link U_nu at x + o (after the step)
        return fac, o

    @staticmethod
    def product(mats, like):
        if len(mats) == 0:
            r = g.lattice(like)
            r[:] = 0
            r += g.identity(like)
            return r
        r = g.copy(mats[0])
        for m in mats[1:]:
            r = g(r * m)
        return r

    # (T P_{\mu*})(x) for a single component field Pm (full lattice; caller applies masks)
    def hop(self, Pm):
        res = g.lattice(Pm)
        res[:] = 0
        for path, w in self.hops:
            fac, d = self.factors(path)
            # W = self.product([f[3] for f in fac], self.U[0])
            W = self.transport(path)
            res += w * W * shift(Pm, d) * g.adj(W)
        return res

    def apply(self, P, sign=+1, adjoint=False):
        mask = self.read if adjoint else self.write
        P[self.mu] @= P[self.mu] + (sign * self.eta) * mask * self.hop(P[self.mu])

    # add this layer's U-force contribution to F (list over link directions)
    # X: P_{\mu*} before the layer; R: adjoint state after the layer (full list)
    def add_force(self, F, X, R):
        Rt = g(self.write * R[self.mu])
        for path, w in self.hops:
            fac, d = self.factors(path)
            Xd = shift(X, d)
            m = len(fac)
            for k in range(m):
                nu, s, o, Vk = fac[k]
                # Ak = self.product([f[3] for f in fac[:k]], self.U[0])
                # Bk = self.product([f[3] for f in fac[k + 1 :]], self.U[0])
                Ak, Bk = self.partials(path, k)
                BXB = g(Bk * Xd * g.adj(Bk))
                if s > 0:
                    M = g(Vk * BXB * g.adj(Vk))
                    At = g(g.adj(Ak) * Rt * Ak)
                    f = g((1j * self.eta * w) * (M * At - At * M))
                else:
                    At = g(g.adj(Vk) * g.adj(Ak) * Rt * Ak * Vk)
                    f = g((-1j * self.eta * w) * (BXB * At - At * BXB))
                # field at x belongs to the link at z = x + o  ->  F(z) += f(z - o)
                F[nu] += shift(f, neg_vec(o))


# ---------------- kernel builders ----------------
def coordinate_masks(grid):
    one = g.complex(grid)
    coor = g.coordinates(one)
    return one, coor


def mask_from(grid, values):
    m = g.complex(grid)
    one, coor = coordinate_masks(grid)
    m[coor] = values.astype(np.complex128).reshape(-1, 1)
    return m


def site_parity_masks(grid):
    one, coor = coordinate_masks(grid)
    par = np.sum(coor, axis=1) % 2
    return mask_from(grid, (par == 0) * 1.0), mask_from(grid, (par == 1) * 1.0)


def block_parity_masks(grid, nu):
    one, coor = coordinate_masks(grid)
    assert grid.fdimensions[nu] % 4 == 0, "rectangle kernel needs L_nu divisible by 4"
    b = (coor[:, nu] // 2) % 2
    return mask_from(grid, (b == 0) * 1.0), mask_from(grid, (b == 1) * 1.0)


def hops_T1(nd, mu):
    return [([(nu, s)], 1.0) for nu in range(nd) if nu != mu for s in (+1, -1)]


def hops_R(nu):
    return [([(nu, s), (nu, s)], 1.0) for s in (+1, -1)]


def hops_C(nd, mu):
    hops = []
    for nu in range(nd):
        if nu == mu:
            continue
        rhos = [r for r in range(nd) if r not in (mu, nu)]
        w = 1.0 / (2 * len(rhos))
        for s in (+1, -1):
            for rho in rhos:
                for sr in (+1, -1):
                    hops.append(([(rho, sr), (nu, s), (rho, -sr)], w))
    return hops


# one "iteration" of a kernel type: list of layers in application order
class kernel_builder:
    def __init__(self, grid):
        self.grid = grid
        self.nd = grid.nd
        self.even, self.odd = site_parity_masks(grid)
        self.block = {}

    def blocks(self, nu):
        if nu not in self.block:
            self.block[nu] = block_parity_masks(self.grid, nu)
        return self.block[nu]

    def layers(self, U, kind, eta):
        L = []
        for mu in range(self.nd):
            if kind in ("T1", "C"):
                hops = hops_T1(self.nd, mu) if kind == "T1" else hops_C(self.nd, mu)
                L.append(path_layer(U, mu, hops, self.even, self.odd, eta, f"{kind} mu={mu} even"))
                L.append(path_layer(U, mu, hops, self.odd, self.even, eta, f"{kind} mu={mu} odd"))
            elif kind == "R":
                for nu in range(self.nd):
                    if nu == mu:
                        continue
                    b0, b1 = self.blocks(nu)
                    L.append(path_layer(U, mu, hops_R(nu), b0, b1, eta, f"R mu={mu} nu={nu} b0"))
                    L.append(path_layer(U, mu, hops_R(nu), b1, b0, eta, f"R mu={mu} nu={nu} b1"))
            else:
                raise ValueError(kind)
        return L


# ---------------- flow = composition of layers ----------------
# spec: list of (kind, eta), applied in order; e.g. [("T1", 0.05), ("R", 0.02), ("C", 0.03)] * niter
class path_flow:
    def __init__(self, U, spec, builder):
        self.U = U
        self.nd = len(U)
        self.seq = []
        for kind, eta in spec:
            self.seq += builder.layers(U, kind, eta)

    def forward(self, P):
        Q = g.copy(P)
        for l in self.seq:
            l.apply(Q, +1)
        return Q

    def inverse(self, P):
        Q = g.copy(P)
        for l in reversed(self.seq):
            l.apply(Q, -1)
        return Q

    def adjoint(self, P):
        Q = g.copy(P)
        for l in reversed(self.seq):
            l.apply(Q, +1, adjoint=True)
        return Q

    # U-force of (1/2) <A P, A P> (backprop; see plan entry (3) and "General force for any path")
    def force(self, P):
        X = self.forward(P)
        R = g.copy(X)
        F = [g.lattice(P[0]) for nu in range(self.nd)]
        for f in F:
            f[:] = 0
        for l in reversed(self.seq):
            l.apply(X, -1)  # X = state before layer l
            l.add_force(F, X[l.mu], R)
            l.apply(R, +1, adjoint=True)  # R <- L^\dagger R
        return F


# kinetic term (1/2) <A(U) P, A(U) P> on fields = U + P (same interface as transverse_kinetic)
class path_kinetic(g.group.differentiable_functional):
    def __init__(self, spec):
        self.spec = spec
        self.builder = None
        self.__name__ = f"path_kinetic({spec})"

    def flow(self, U):
        if self.builder is None:
            self.builder = kernel_builder(U[0].grid)
        return path_flow(U, self.spec, self.builder)

    def split(self, fields):
        n = len(fields)
        assert n % 2 == 0
        return fields[: n // 2], fields[n // 2 :]

    def __call__(self, fields):
        U, P = self.split(fields)
        AP = self.flow(U).forward(P)
        return 0.5 * sum(g.group.inner_product(x, x) for x in AP)

    def draw(self, fields, rng):
        U, P = self.split(fields)
        xi = g.copy(P)
        rng.normal_element(xi)
        P_new = self.flow(U).inverse(xi)
        for mu in range(len(P)):
            P[mu] @= P_new[mu]
        return 0.5 * sum(g.group.inner_product(x, x) for x in xi)

    @g.group.differentiable_functional.multi_field_gradient
    def gradient(self, fields, dfields):
        U, P = self.split(fields)
        A = self.flow(U)
        need_P = any(any(f is p for p in P) for f in dfields)
        need_U = any(any(f is u for u in U) for f in dfields)
        KP = A.adjoint(A.forward(P)) if need_P else None
        FU = A.force(P) if need_U else None
        dS = []
        for f in dfields:
            mu = next(i for i, x in enumerate(fields) if x is f)
            dS.append(FU[mu] if mu < len(U) else KP[mu - len(U)])
        return dS
