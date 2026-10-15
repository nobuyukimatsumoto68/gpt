#
# Gauge-covariant Gaussian momentum filter for SU(N) link momenta, and cosine similarity.
# Design and equations: genesis/rmhmc_trainable_kernel_impl_plan_claude.md,
#   section "Study: momentum decorrelation along a trajectory" -> "Covariant Gaussian momentum filter".
#
# Covariant adjoint Laplacian on each direction component P_\mu (all \nu):
#   (\Delta P_\mu)(x) = \sum_\nu [ U_\nu(x) P_\mu(x+\nu) U_\nu^\dagger(x) + U_\nu^\dagger(x-\nu) P_\mu(x-\nu) U_\nu(x-\nu) - 2 P_\mu(x) ]
# -\Delta: hermitian, >= 0, spectrum in [0, 4 nd]; free field (U = 1): eigenvalues \hat k^2 = \sum_\nu 4 sin^2(k_\nu / 2).
# Filter G = g(-\Delta),  g(\lambda) = exp(-(\lambda - \lambda_0)^2 / (2 \sigma_\lambda^2)),
# evaluated by a Chebyshev expansion on [0, 4 nd] with the Clenshaw recurrence
#   (Chebyshev / Clenshaw: e.g. W. H. Press et al., Numerical Recipes, 3rd ed., Sec. 5.8 and 5.4).
#
import gpt as g
import numpy as np
from numpy.polynomial import chebyshev as cheb


def khat2(k):
    return sum(4.0 * np.sin(0.5 * x) ** 2 for x in k)


# lambda_0 and sigma_lambda for an on-axis k_0 = (k, 0, 0, 0) and k-space width sigma (plan: image of the k window)
def shell_window(k, sigma):
    lam0 = khat2([k])
    sig_lam = max(abs(khat2([k + sigma]) - lam0), abs(khat2([k - sigma]) - lam0))
    return lam0, sig_lam


class covariant_gaussian_filter:
    def __init__(self, U, lam0, sigma_lam, tol=1e-8, max_degree=2000):
        self.U = U
        self.nd = len(U)
        self.lam_max = 4.0 * self.nd
        self.lam0 = lam0
        self.sigma_lam = sigma_lam
        self.coef, self.cheb_error = self.chebyshev_coefficients(tol, max_degree)
        self.degree = len(self.coef) - 1

    def g_of_lambda(self, lam):
        return np.exp(-((lam - self.lam0) ** 2) / (2.0 * self.sigma_lam**2))

    # x = 2 \lambda / \lambda_max - 1 in [-1, 1]
    def chebyshev_coefficients(self, tol, max_degree):
        xs = np.linspace(-1.0, 1.0, 20001)
        f = lambda x: self.g_of_lambda(0.5 * (x + 1.0) * self.lam_max)
        deg = 8
        while True:
            c = cheb.chebinterpolate(f, deg)
            err = np.max(np.abs(cheb.chebval(xs, c) - f(xs)))
            if err < tol or deg >= max_degree:
                return c, err
            deg = int(deg * 1.5)

    # (\Delta P_\mu) for a single component
    def laplace(self, Pm):
        res = g(-2.0 * self.nd * Pm)
        for nu in range(self.nd):
            res += self.U[nu] * g.cshift(Pm, nu, 1) * g.adj(self.U[nu])
            res += g.cshift(g.adj(self.U[nu]) * Pm * self.U[nu], nu, -1)
        return res

    # X v = (2 (-\Delta) / \lambda_max - 1) v
    def X(self, v):
        return g((-2.0 / self.lam_max) * self.laplace(v) - v)

    # G P_\mu via Clenshaw:  b_m = c_m v + 2 X b_{m+1} - b_{m+2};  G v = c_0 v + X b_1 - b_2
    def apply_component(self, v):
        c = self.coef
        b1 = g.lattice(v)
        b1[:] = 0
        b2 = g.copy(b1)
        for m in range(self.degree, 0, -1):
            b0 = g(c[m] * v + 2.0 * self.X(b1) - b2)
            b2 = b1
            b1 = b0
        return g(c[0] * v + self.X(b1) - b2)

    def __call__(self, P):
        return [self.apply_component(p) for p in P]


def cosine_similarity(P0, P1):
    ip = sum(g.group.inner_product(a, b) for a, b in zip(P0, P1))
    n0 = sum(g.group.inner_product(a, a) for a in P0)
    n1 = sum(g.group.inner_product(a, a) for a in P1)
    return ip / (n0 * n1) ** 0.5
