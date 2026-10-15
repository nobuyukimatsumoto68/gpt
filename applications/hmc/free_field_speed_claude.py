#
# Leading-order (free-field) model of MD mode speeds for the coupling-layer RMHMC kernels.
# Design and equations: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, "E3(a) LO model".
# numpy only (no gpt dependency).
# Sources: improved gauge action quadratic kernel: P. Weisz, Nucl. Phys. B 212, 1 (1983); Iwasaki c1 = -0.331
#   (Y. Iwasaki, UTHEP-118 (1983)); Fourier acceleration: C. Davies et al., Phys. Rev. D 37, 1581 (1988);
#   RMHMC: M. Girolami, B. Calderhead, J. R. Stat. Soc. B 73 (2011) 123.
#
# Conventions: site-labelled link fields a_mu(x) (link x -> x + mu); phi(x) = sum_k e^{i k.x} phi~(k);
#   hop: phi(x + d) -> e^{i k.d} phi~(k);  mask m(x) = sum_q mhat(q) e^{i q.x}:  (m phi)~(k) = sum_q mhat(q) phi~(k - q).
# Layer (component mu*, at U = 1):  L = 1 + eta M_w T M_r   (M_r = 1 - M_w).
#
import numpy as np
import itertools

ND = 4


# ---------------- masks on the period-4 cell ----------------
def cell_coords():
    return np.array(list(itertools.product(range(4), repeat=ND)))  # (256, 4)


def mask_cell(kind, nu=None):
    c = cell_coords()
    if kind == "even":
        return (np.sum(c, axis=1) % 2 == 0) * 1.0
    if kind == "odd":
        return (np.sum(c, axis=1) % 2 == 1) * 1.0
    if kind == "b0":
        return ((c[:, nu] // 2) % 2 == 0) * 1.0
    if kind == "b1":
        return ((c[:, nu] // 2) % 2 == 1) * 1.0
    raise ValueError(kind)


# mhat(q) for q = (pi/2) m, m in Z_4^4 (same ordering as cell_coords)
def mask_fourier(mvals):
    c = cell_coords()
    m = cell_coords()
    phase = np.exp(-1j * (np.pi / 2) * (m @ c.T))  # (q, x)
    return phase @ mvals / len(c)


# ---------------- layers (kind, eta) -> per component list of (t-symbol fn, write mask, read mask, eta) ----------------
def layer_list(spec, mu):
    L = []
    for kind, eta in spec:
        if kind in ("T1", "C"):  # C == T1 at U = 1 (staple transport = identity, weights sum to 1)
            t = lambda k, mu=mu: sum(2.0 * np.cos(k[..., nu]) for nu in range(ND) if nu != mu)
            L.append((t, ("even", None), ("odd", None), eta))
            L.append((t, ("odd", None), ("even", None), eta))
        elif kind == "R":
            for nu in range(ND):
                if nu == mu:
                    continue
                t = lambda k, nu=nu: 2.0 * np.cos(2.0 * k[..., nu])
                L.append((t, ("b0", nu), ("b1", nu), eta))
                L.append((t, ("b1", nu), ("b0", nu), eta))
        else:
            raise ValueError(kind)
    return L


# ---------------- real-space reference implementation (scalar fields) ----------------
class real_space_flow:
    def __init__(self, L, spec):
        self.L = L
        x = np.indices((L,) * ND)
        self.mask = {("even", None): (x.sum(0) % 2 == 0) * 1.0, ("odd", None): (x.sum(0) % 2 == 1) * 1.0}
        for nu in range(ND):
            self.mask[("b0", nu)] = ((x[nu] // 2) % 2 == 0) * 1.0
            self.mask[("b1", nu)] = ((x[nu] // 2) % 2 == 1) * 1.0
        self.spec = spec

    def hop(self, f, kind, mu, nu=None):
        r = np.zeros_like(f)
        if kind in ("T1", "C"):
            for n in range(ND):
                if n != mu:
                    r += np.roll(f, -1, axis=n) + np.roll(f, 1, axis=n)  # f(x + n) + f(x - n)
        else:
            r += np.roll(f, -2, axis=nu) + np.roll(f, 2, axis=nu)
        return r

    # phi: (ND, L, L, L, L); A applied component-wise
    def forward(self, phi, sign=+1):
        phi = phi.copy()
        for mu in range(ND):
            seq = []
            for kind, eta in self.spec:
                if kind in ("T1", "C"):
                    seq += [(kind, None, ("even", None), eta), (kind, None, ("odd", None), eta)]
                else:
                    for nu in range(ND):
                        if nu != mu:
                            seq += [(kind, nu, ("b0", nu), eta), (kind, nu, ("b1", nu), eta)]
            if sign < 0:
                seq = list(reversed(seq))
            for kind, nu, w, eta in seq:
                phi[mu] += sign * eta * self.mask[w] * self.hop(phi[mu], kind, mu, nu)
        return phi


# ---------------- Fourier-orbit model ----------------
class free_field_model:
    def __init__(self, L, c1=-0.331):
        assert L % 4 == 0
        self.L = L
        self.c1 = c1
        self.m = cell_coords()  # orbit offsets in units of pi/2
        self.mhat = {}
        for key in [("even", None), ("odd", None)] + [(b, nu) for nu in range(ND) for b in ("b0", "b1")]:
            self.mhat[key] = mask_fourier(mask_cell(*key))
        # orbit representatives n0 in Z_{L/4}^4
        self.reps = list(itertools.product(range(L // 4), repeat=ND))
        # index of (m_i - m_j) mod 4 in cell ordering
        diff = (self.m[:, None, :] - self.m[None, :, :]) % 4
        self.diff_idx = np.ravel_multi_index(tuple(diff[..., d] for d in range(ND)), (4,) * ND)

    def momenta(self, n0):
        n = (np.array(n0)[None, :] + (self.L // 4) * self.m) % self.L
        return 2.0 * np.pi * n / self.L  # (256, 4)

    def mask_matrix(self, key):
        return self.mhat[key][self.diff_idx]  # [i, j] = mhat(k_i - k_j)

    def A_orbit(self, spec, mu, k):
        n = len(k)
        A = np.eye(n, dtype=complex)
        for t, w, r, eta in layer_list(spec, mu):
            Lyr = np.eye(n, dtype=complex) + eta * self.mask_matrix(w) @ np.diag(t(k)) @ self.mask_matrix(r)
            A = Lyr @ A
        return A

    # Hessian h(k) (4x4) of the quadratic lattice action (normalization: c0 + 8 c1 = 1)
    def hessian(self, k, continuum=False):
        h = np.zeros((ND, ND), dtype=complex)
        if continuum:
            return (np.dot(k, k) * np.eye(ND) - np.outer(k, k)).astype(complex)
        delta = np.exp(1j * k) - 1.0
        kh2 = 4.0 * np.sin(0.5 * k) ** 2
        for mu in range(ND):
            for nu in range(mu + 1, ND):
                w = 1.0 - self.c1 * (kh2[mu] + kh2[nu])
                v = np.zeros(ND, dtype=complex)
                v[nu] = delta[mu]
                v[mu] = -delta[nu]
                h += w * np.outer(v.conj(), v)
        return h

    # MD frequencies omega^2 over all orbits (physical modes only)
    def omega2(self, spec, continuum=False, zero_tol=1e-9):
        out = []
        for n0 in self.reps:
            k = self.momenta(n0)
            n = len(k)
            B = np.zeros((ND * n, ND * n), dtype=complex)
            for mu in range(ND):
                B[mu * n : (mu + 1) * n, mu * n : (mu + 1) * n] = self.A_orbit(spec, mu, k)
            H = np.zeros((ND * n, ND * n), dtype=complex)
            for i in range(n):
                hk = self.hessian(k[i], continuum)
                idx = [mu * n + i for mu in range(ND)]
                H[np.ix_(idx, idx)] = hk
            # omega^2 = eig(B H B^dag); H acts on a, B^dag maps ... (see plan)
            M = B @ H @ B.conj().T
            ev = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
            out.append(ev)
        ev = np.concatenate(out)
        return ev[ev > zero_tol * max(1.0, ev.max())]

    def objective(self, spec, continuum=False):
        w2 = self.omega2(spec, continuum)
        return np.var(np.log(w2)), w2.max() / w2.min(), w2
