#
# Unmasked polynomial Fourier-acceleration kernel (NM, 2026-10-15; nonzero determinant allowed):
#   T = (4/3) T_1 - (1/12) R   (improved covariant transverse hop; free symbol 15/2 - lambda)
#   A = prod_j q_j(T),  q_j(T) = 1 + alpha_j T + beta_j T^2  (complex roots -> A hermitian positive definite)
#   H = (1/2) <A P, A P> + S(U) + (1/2) <R, A^{-2} R>      (log det A via the auxiliary field R)
# Design and equations: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, entries (13)-(15).
# Sources: Fourier acceleration: C. Davies et al., Phys. Rev. D 37, 1581 (1988); RMHMC: M. Girolami,
#   B. Calderhead, J. R. Stat. Soc. B 73 (2011) 123; auxiliary field for the log det: as in C. Lehner's
#   gpt applications/hmc/rmhmc.py (cf. arXiv:2412.19904); improved Laplacian (4/3, -1/12): standard O(a^4)
#   improvement; CG: M. R. Hestenes, E. Stiefel, J. Res. Natl. Bur. Stand. 49, 409 (1952).
#
import gpt as g
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_smear_claude import path_layer, hops_T1, hops_R

# N = 8 coefficients (plan entry (15)): (alpha_j, beta_j)
COEFFS_N8 = [(-0.240073, 0.017197), (-0.089958, 0.030094), (0.191793, 0.018357), (0.195884, 0.010038)]


def improved_hops(nd, mu):
    hops = [(p, (4.0 / 3.0) * w) for p, w in hops_T1(nd, mu)]
    for nu in range(nd):
        if nu != mu:
            hops += [(p, (-1.0 / 12.0) * w) for p, w in hops_R(nu)]
    return hops


class poly_kernel:
    def __init__(self, U, coeffs, ones=None, cg_tol=1e-12, cg_maxiter=1000, fast_grad=True, transport_cache=False):
        self.U = U
        self.fast_grad = fast_grad  # True: grad_A_fast; False: original grad_A (reference)
        self.nd = len(U)
        self.coeffs = coeffs
        if ones is None:
            ones = g.complex(U[0].grid)
            ones[:] = 1.0
        # one unmasked path layer per component (write = read = all sites, eta = 1): used for hop and force
        # self.T_mu = [path_layer(U, mu, improved_hops(self.nd, mu), ones, ones, 1.0, f"T_imp mu={mu}") for mu in range(self.nd)]
        self.T_mu = [path_layer(U, mu, improved_hops(self.nd, mu), ones, ones, 1.0, f"T_imp mu={mu}", cache=transport_cache) for mu in range(self.nd)]
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        self.cg_iters = []

    # rebind to a U list with identical content (cached transports stay valid; lazily computed ones use the new U)
    def rebind(self, U):
        self.U = U
        for l in self.T_mu:
            l.U = U

    # T P (all components)
    def T(self, P):
        return [g(self.T_mu[mu].hop(P[mu])) for mu in range(self.nd)]

    # q_j(T) P = P + alpha T P + beta T (T P)
    def q(self, j, P):
        a, b = self.coeffs[j]
        TP = self.T(P)
        TTP = self.T(TP)
        return [g(p + a * tp + b * ttp) for p, tp, ttp in zip(P, TP, TTP)]

    def apply_range(self, P, js):
        for j in js:
            P = self.q(j, P)
        return P

    # A P (A hermitian: adjoint = forward)
    def forward(self, P):
        return self.apply_range(P, range(len(self.coeffs)))

    def adjoint(self, P):
        return self.forward(P)

    @staticmethod
    def ip(X, Y):
        return sum(g.group.inner_product(x, y) for x, y in zip(X, Y))

    # A^{-1} B by CG (A hermitian positive definite); optional initial guess x0 (warm start)
    def inverse(self, B, x0=None):
        bb = self.ip(B, B)
        if x0 is None:
            X = [g(0.0 * b) for b in B]
            Rr = g.copy(B)
        else:
            X = g.copy(x0)
            AX = self.forward(X)
            Rr = [g(b - ax) for b, ax in zip(B, AX)]
        Pp = g.copy(Rr)
        rr = self.ip(Rr, Rr)
        if rr <= self.cg_tol**2 * bb:
            self.cg_iters.append(0)
            return X
        for it in range(1, self.cg_maxiter + 1):
            Ap = self.forward(Pp)
            alpha = rr / self.ip(Pp, Ap)
            X = [g(x + alpha * p) for x, p in zip(X, Pp)]
            Rr = [g(r - alpha * ap) for r, ap in zip(Rr, Ap)]
            rr_new = self.ip(Rr, Rr)
            if rr_new <= self.cg_tol**2 * bb:
                self.cg_iters.append(it)
                return X
            Pp = [g(r + (rr_new / rr) * p) for r, p in zip(Rr, Pp)]
            rr = rr_new
        raise RuntimeError(f"poly_kernel CG did not converge: |r|/|b| = {(rr_new / bb) ** 0.5}")

    # g(L, X) = d_U <L, T X>  (added into F)
    def add_grad_T(self, F, L, X):
        for mu in range(self.nd):
            self.T_mu[mu].add_force(F, X[mu], L)

    def grad(self, L, X):
        return self.grad_A_fast(L, X) if self.fast_grad else self.grad_A(L, X)

    # d_U <L, A X>, optimized (same result as grad_A):
    #   prefix products  L_j = q_{<j} L  and suffix products  X_j = q_{>j} X  computed once (2(n-1) q's instead of n(n-1));
    #   bilinearity of g:  alpha g(L,X) + beta g(L,TX) = g(L, alpha X + beta T X)  -> 2 g per factor instead of 3.
    def grad_A_fast(self, L, X):
        n = len(self.coeffs)
        Lp = [L]
        for j in range(n - 1):
            Lp.append(self.q(j, Lp[-1]))  # Lp[j] = q_<j L
        Xs = [None] * n
        Xs[n - 1] = X
        for j in range(n - 1, 0, -1):
            Xs[j - 1] = self.q(j, Xs[j])  # Xs[j] = q_>j X
        F = [g.lattice(X[0]) for nu in range(self.nd)]
        for f in F:
            f[:] = 0
        for j in range(n):
            a, b = self.coeffs[j]
            TX = self.T(Xs[j])
            Y = [g(a * x + b * tx) for x, tx in zip(Xs[j], TX)]
            self.add_grad_T(F, Lp[j], Y)  # alpha <L, dT X> + beta <L, dT T X>
            TL = [g(b * tl) for tl in self.T(Lp[j])]
            self.add_grad_T(F, TL, Xs[j])  # beta <T L, dT X>
        return F

    # d_U <L, A X>  (L, X fixed); A = q_1 ... q_n, all q_j functions of the same T  (original; reference for grad_A_fast)
    def grad_A(self, L, X):
        F = [g.lattice(X[0]) for nu in range(self.nd)]
        for f in F:
            f[:] = 0
        n = len(self.coeffs)
        for j in range(n):
            a, b = self.coeffs[j]
            Lj = self.apply_range(L, range(j))  # q_<j L
            Xj = self.apply_range(X, range(j + 1, n))  # q_>j X
            Fj = [g.lattice(X[0]) for nu in range(self.nd)]
            for f in Fj:
                f[:] = 0
            self.add_grad_T(Fj, Lj, Xj)  # <L, dT X>
            Gj = [g.lattice(X[0]) for nu in range(self.nd)]
            for f in Gj:
                f[:] = 0
            self.add_grad_T(Gj, Lj, self.T(Xj))  # <L, dT T X>
            self.add_grad_T(Gj, self.T(Lj), Xj)  # <T L, dT X>
            for nu in range(self.nd):
                F[nu] += a * Fj[nu] + b * Gj[nu]
        return F


class poly_functional_base(g.group.differentiable_functional):
    fast_grad = True
    # transport_cache = True: reuse the kernel (with memoized path transports) while U is unchanged, detected by a
    # fingerprint f(U) = sum_mu <U_mu, Z_mu> with a fixed random field Z; shared between all poly functionals
    # (kinetic and log det). Default False = original behaviour.
    transport_cache = False
    _shared = {"key": None, "kernel": None, "Z": None}

    def __init__(self, coeffs, cg_tol=1e-12):
        self.coeffs = coeffs
        self.cg_tol = cg_tol
        self.ones = None
        self.last = None

    @classmethod
    def fingerprint(cls, U):
        Z = cls._shared["Z"]
        if Z is None or Z[0].grid is not U[0].grid:
            nprng = np.random.default_rng(12345)
            Z = [g.lattice(u) for u in U]
            for z in Z:  # arbitrary complex matrices (storage only; fingerprint, not a group element)
                shape = z[:].shape
                z[:] = nprng.normal(size=shape) + 1j * nprng.normal(size=shape)
            cls._shared["Z"] = Z
        return tuple(complex(g.inner_product(z, u)) for z, u in zip(Z, U))

    def kernel(self, U):
        if self.ones is None:
            self.ones = g.complex(U[0].grid)
            self.ones[:] = 1.0
        if not self.transport_cache:
            self.last = poly_kernel(U, self.coeffs, self.ones, self.cg_tol, fast_grad=self.fast_grad)
            return self.last
        key = (self.fingerprint(U), tuple(self.coeffs), self.cg_tol, self.fast_grad)
        sh = poly_functional_base._shared
        if sh["key"] == key:
            sh["kernel"].rebind(U)
        else:
            sh["key"] = key
            sh["kernel"] = poly_kernel(U, self.coeffs, self.ones, self.cg_tol, fast_grad=self.fast_grad, transport_cache=True)
        self.last = sh["kernel"]
        return self.last

    def split(self, fields):
        n = len(fields)
        assert n % 2 == 0
        return fields[: n // 2], fields[n // 2 :]


# H_kin = (1/2) <A P, A P> on fields = U + P
class poly_kinetic(poly_functional_base):
    __name__ = "poly_kinetic"

    def __call__(self, fields):
        U, P = self.split(fields)
        AP = self.kernel(U).forward(P)
        return 0.5 * poly_kernel.ip(AP, AP)

    def draw(self, fields, rng):
        U, P = self.split(fields)
        xi = g.copy(P)
        rng.normal_element(xi)
        Pn = self.kernel(U).inverse(xi)
        for mu in range(len(P)):
            P[mu] @= Pn[mu]
        return 0.5 * poly_kernel.ip(xi, xi)

    @g.group.differentiable_functional.multi_field_gradient
    def gradient(self, fields, dfields):
        U, P = self.split(fields)
        A = self.kernel(U)
        need_P = any(any(f is p for p in P) for f in dfields)
        need_U = any(any(f is u for u in U) for f in dfields)
        AP = A.forward(P) if (need_P or need_U) else None
        KP = A.forward(AP) if need_P else None
        # FU = A.grad_A(AP, P) if need_U else None  # d (1/2)<AP,AP> = <AP, dA P>
        FU = A.grad(AP, P) if need_U else None  # d (1/2)<AP,AP> = <AP, dA P>
        dS = []
        for f in dfields:
            mu = next(i for i, x in enumerate(fields) if x is f)
            dS.append(FU[mu] if mu < len(U) else KP[mu - len(U)])
        return dS


# H_R = (1/2) <R, A^{-2} R> on fields = U + R  (R: auxiliary field, fixed during MD)
class poly_logdet(poly_functional_base):
    __name__ = "poly_logdet"

    # warm start: keep the previous solutions Y = A^{-1} R, Z = A^{-2} R as CG initial guesses
    # (the gauge field changes little between MD steps; results unchanged up to the CG tolerance)
    warm_start = True
    Y_prev = None
    Z_prev = None
    cg_log = None

    def solve(self, A, B, prev):
        x0 = prev if (self.warm_start and prev is not None) else None
        X = A.inverse(B, x0)
        if self.cg_log is None:
            self.cg_log = []
        self.cg_log.append(A.cg_iters[-1])
        return X

    def __call__(self, fields):
        U, R = self.split(fields)
        # Y = self.kernel(U).inverse(R)
        Y = self.solve(self.kernel(U), R, self.Y_prev)
        self.Y_prev = Y
        return 0.5 * poly_kernel.ip(Y, Y)

    def draw(self, fields, rng):
        U, R = self.split(fields)
        xi = g.copy(R)
        rng.normal_element(xi)
        Rn = self.kernel(U).forward(xi)
        for mu in range(len(R)):
            R[mu] @= Rn[mu]
        self.Y_prev, self.Z_prev = g.copy(xi), None  # A^{-1} R = xi exactly
        return 0.5 * poly_kernel.ip(xi, xi)

    @g.group.differentiable_functional.multi_field_gradient
    def gradient(self, fields, dfields):
        U, R = self.split(fields)
        A = self.kernel(U)
        # Y = A.inverse(R)
        # Z = A.inverse(Y)
        Y = self.solve(A, R, self.Y_prev)
        Z = self.solve(A, Y, self.Z_prev)
        self.Y_prev, self.Z_prev = Y, Z
        need_U = any(any(f is u for u in U) for f in dfields)
        need_R = any(any(f is r for r in R) for f in dfields)
        # FU = [g(-x) for x in A.grad_A(Y, Z)] if need_U else None  # d (1/2)<R, A^-2 R> = -<Y, dA Z>
        FU = [g(-x) for x in A.grad(Y, Z)] if need_U else None  # d (1/2)<R, A^-2 R> = -<Y, dA Z>
        dS = []
        for f in dfields:
            mu = next(i for i, x in enumerate(fields) if x is f)
            dS.append(FU[mu] if mu < len(U) else (Z[mu - len(U)] if need_R else None))
        return dS


# gauge action plus the R term (R fixed), as a functional of U only: plugs into implicit_leapfrog as `action`
class action_with_R(g.group.differentiable_functional):
    def __init__(self, S, logdet, R):
        self.S = S
        self.logdet = logdet
        self.R = R
        self.__name__ = "action_with_R"

    def __call__(self, U):
        return self.S(U) + self.logdet(U + self.R)

    def gradient(self, U, dU):
        FS = self.S.gradient(U, U)
        FR = self.logdet.gradient(U + self.R, U)
        return [g(a + b) for a, b in zip(FS, FR)]
