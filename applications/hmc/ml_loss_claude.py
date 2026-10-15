#
# ML loss for the RMHMC kernel (NM, 2026-10-15): speed alignment of all modes on ONE interacting configuration.
#   M = A h A^dagger  (similar to K h, K = A^dagger A; eigenvalues = squared MD frequencies of the linearized dynamics)
#   loss  R(theta) = tr M^2 / (tr M)^2  >=  1/N_p   (Cauchy-Schwarz; equality iff K h ~ 1 on the physical modes)
#   reported also as the relative variance  N_p R - 1  (N_p = physical d.o.f. = 3/4 of the link-algebra d.o.f.)
# Traces by Hutchinson (Gaussian probes; independent probe sets for numerator and denominator):
#   tr M ~ <phi, M phi>,  tr M^2 ~ <phi, M (M phi)>  (exact for non-symmetric M as well)
# h v: derivative of the gauge force along our update curve (= linearized MD operator), central finite difference:
#   h v = [F_S(e^{i e v} U) - F_S(e^{-i e v} U)] / (2 e)
# Gradient for the polynomial kernel (A hermitian, all q_j functions of T, commuting):
#   dA/dalpha_j = q_{<j} T q_{>j},  dA/dbeta_j = q_{<j} T^2 q_{>j},  dM = dA h A + A h dA,
#   d tr M = <phi, dM phi>,  d tr M^2 = <phi, (dM M + M dM) phi>  (stochastic, same probes)
# Design: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, "ML loss" sections and entry (18).
# Sources: stochastic trace: M. F. Hutchinson, Commun. Stat. Simul. Comput. 18, 1059 (1989);
#   Fourier acceleration: C. Davies et al., Phys. Rev. D 37, 1581 (1988).
#
import gpt as g
import numpy as np


def ip(X, Y):
    return sum(g.group.inner_product(x, y) for x, y in zip(X, Y))


# J v = d/de F_S(e^{i e v} U) is NOT symmetric on a Lie group: measured exactly (to 1e-6, entry (18))
#   J v - J^T v = i [v, F_S]   (site-local commutator with the force),
# so the symmetric Hessian is  h_sym v = J v - (i/2) [v, F_S]  (used for the loss: Cauchy-Schwarz needs symmetric M).
# Along a gauge direction W = -D phi:  J W = i [phi, F_S] exactly (the force rotates covariantly; checked to 4e-10).
class hessian_fd:
    def __init__(self, S, eps=1e-4, symmetric=True):
        self.S = S
        self.eps = eps
        self.symmetric = symmetric
        self._F = None
        self._Fkey = None

    def force(self, U):
        key = tuple(id(u) for u in U)
        if self._Fkey != key:
            self._F = self.S.gradient(U, U)
            self._Fkey = key
        return self._F

    def __call__(self, U, v):
        Up = [g(g.group.compose(g(self.eps * x), u)) for x, u in zip(v, U)]
        Um = [g(g.group.compose(g(-self.eps * x), u)) for x, u in zip(v, U)]
        Fp = self.S.gradient(Up, Up)
        Fm = self.S.gradient(Um, Um)
        Jv = [g((fp - fm) * (0.5 / self.eps)) for fp, fm in zip(Fp, Fm)]
        if not self.symmetric:
            return Jv
        F = self.force(U)
        return [g(j - 0.5j * (x * f - f * x)) for j, x, f in zip(Jv, v, F)]


# kernel wrappers: A, A^dagger as functions of a list of algebra fields
class kernel_identity:
    def A(self, P):
        return g.copy(P)

    def Adag(self, P):
        return g.copy(P)


class kernel_transverse:
    def __init__(self, U, eta, niter):
        from transverse_smear_claude import transverse_smear

        self.k = transverse_smear(U, eta, niter)

    def A(self, P):
        return self.k.forward(P)

    def Adag(self, P):
        return self.k.adjoint(P)


class kernel_poly:
    def __init__(self, U, coeffs, longitudinal=False):
        from poly_kernel_claude import poly_kernel

        # self.k = poly_kernel(U, coeffs, transport_cache=True)
        self.k = poly_kernel(U, coeffs, transport_cache=True, longitudinal=longitudinal)
        self.coeffs = coeffs

    def A(self, P):
        return self.k.forward(P)

    def Adag(self, P):
        return self.k.forward(P)

    # dA/dalpha_j P and dA/dbeta_j P
    def dA(self, P, j, which):
        n = len(self.coeffs)
        X = self.k.apply_range(P, range(j + 1, n))  # q_>j P
        X = self.k.T(X)
        if which == "beta":
            X = self.k.T(X)
        return self.k.apply_range(X, range(j))  # q_<j (...)


class speed_loss:
    def __init__(self, U, kernel, hvp):
        self.U = U
        self.kernel = kernel
        self.hvp = hvp
        self.n_hvp = 0

    def M(self, phi):
        self.n_hvp += 1
        return self.kernel.A(self.hvp(self.U, self.kernel.Adag(phi)))

    def estimate(self, probes_num, probes_den):
        trM2 = np.mean([ip(p, self.M(self.M(p))) for p in probes_num])
        trM = np.mean([ip(p, self.M(p)) for p in probes_den])
        return trM2 / trM**2, trM, trM2

    # gradient of R = tr M^2 / (tr M)^2 w.r.t. the polynomial coefficients (kernel_poly only)
    def gradient_poly(self, probes_num, probes_den):
        k = self.kernel
        n = len(k.coeffs)
        R, trM, trM2 = self.estimate(probes_num, probes_den)
        grad = np.zeros((n, 2))
        hA = lambda P: self.hvp(self.U, k.Adag(P))  # h A^dagger P
        for j in range(n):
            for w, which in enumerate(["alpha", "beta"]):
                dM = lambda P: [g(a + b) for a, b in zip(k.dA(hA(P), j, which), k.A(self.hvp(self.U, k.dA(P, j, which))))]
                dtrM = np.mean([ip(p, dM(p)) for p in probes_den])
                dtrM2 = np.mean([ip(p, self.M(dM(p))) + ip(p, dM(self.M(p))) for p in probes_num])
                grad[j, w] = dtrM2 / trM**2 - 2.0 * trM2 * dtrM / trM**3
        return R, grad
