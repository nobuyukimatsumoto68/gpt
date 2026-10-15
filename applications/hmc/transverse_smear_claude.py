#
# Covariant transverse momentum-smearing kernel for SU(N) RMHMC (hand-coded coupling layers).
#
# Design and equations: genesis/rmhmc_trainable_kernel_impl_plan_claude.md,
#   section "Covariant smearing kernel for SU(3) link momenta".
# Sources:
#   kernel idea: R. Abbott, ..., N. Matsumoto et al., "Kernel for the Riemannian Manifold hybrid
#     Monte Carlo in the AI/ML era" (draft, TrainableKernel-2.pdf), Eqs. (1)-(3), (12)-(14)
#   coupling layers: L. Dinh et al., arXiv:1410.8516, arXiv:1605.08803
#   RMHMC: M. Girolami, B. Calderhead, J. R. Stat. Soc. B 73 (2011) 123
#   CPN prototype: julia_in_script2/header_claude.jl (add_Aeo_p_nu_to!, A_smear_act, A_smear_inv_act)
#
# Conventions (gpt): U_\mu(x) <- exp(\varepsilon P_\mu(x)) U_\mu(x), so P_\mu(x) transforms at x:
#   P_\mu(x) -> V(x) P_\mu(x) V^\dagger(x).
#
# Hop (transverse, single-link parallel transport):
#   (T_{\mu*} P)(x) = \sum_{\nu != \mu*} [ U_\nu(x) P_{\mu*}(x+\nu) U_\nu^\dagger(x)
#                                       + U_\nu^\dagger(x-\nu) P_{\mu*}(x-\nu) U_\nu(x-\nu) ]
# Layer on mask (parity, \mu*):
#   P_{\mu*} <- P_{\mu*} + \eta \Pi_{parity} T_{\mu*} P      (reads only the opposite parity)
# One iteration: for \mu* = 0..nd-1: layer(even), then layer(odd).  A = iteration^niter.
# Inverse: reverse order with -\eta (exact: a layer never reads the sites it writes).
# Adjoint: reverse order with the parity swapped, same +\eta  (T is hermitian).
#
import gpt as g


class transverse_smear:
    def __init__(self, U, eta, niter, mask=None):
        self.U = U
        self.eta = eta
        self.niter = niter
        self.nd = len(U)
        if mask is None:
            even, odd = g.even_odd_projectors(U[0].grid)
            mask = {"even": even, "odd": odd}
        self.mask = mask
        self.swap = {"even": "odd", "odd": "even"}

    # (T_{\mu*} P)(x), full lattice
    def hop(self, P, mu_star):
        Pm = P[mu_star]
        res = g.lattice(Pm)
        res[:] = 0
        for nu in range(self.nd):
            if nu == mu_star:
                continue
            res += self.U[nu] * g.cshift(Pm, nu, 1) * g.adj(self.U[nu])
            res += g.cshift(g.adj(self.U[nu]) * Pm * self.U[nu], nu, -1)
        return res

    # P_{\mu*} <- P_{\mu*} + eta \Pi_{parity} T_{\mu*} P   (in place)
    def layer(self, P, mu_star, parity, eta):
        T_P = self.hop(P, mu_star)
        P[mu_star] @= P[mu_star] + eta * self.mask[parity] * T_P

    # layer sequence of A: [(mu*, parity), ...] in application order
    def sequence(self):
        seq = []
        for k in range(self.niter):
            for mu in range(self.nd):
                seq.append((mu, "even"))
                seq.append((mu, "odd"))
        return seq

    # A P
    def forward(self, P):
        Q = g.copy(P)
        for mu, parity in self.sequence():
            self.layer(Q, mu, parity, self.eta)
        return Q

    # A^{-1} P
    def inverse(self, P):
        Q = g.copy(P)
        for mu, parity in reversed(self.sequence()):
            self.layer(Q, mu, parity, -self.eta)
        return Q

    # A^\dagger P
    def adjoint(self, P):
        Q = g.copy(P)
        for mu, parity in reversed(self.sequence()):
            self.layer(Q, mu, self.swap[parity], self.eta)
        return Q

    # U-force of H_kin = (1/2) <A P, A P>, by backpropagation through the layers.
    # gpt conventions: U_nu <- exp(i W_nu) U_nu with W hermitian traceless; the force F_nu satisfies
    #   dH_kin = sum_nu <W_nu, F_nu>   (g.group.inner_product).
    # Layer j: L_j = 1 + \eta \Pi_\sigma T_{\mu*};  X = P_{j-1} (state before layer j);
    #   R_j = (L_N ... L_{j+1})^\dagger A P,   Rm = \Pi_\sigma R_j[\mu*].
    # For \nu != \mu*:
    #   F_\nu(y) += i \eta [ U_\nu(y) X(y+\nu) U_\nu^\dagger(y), Rm(y) ]
    #             - i \eta [ X(y), U_\nu(y) Rm(y+\nu) U_\nu^\dagger(y) ]
    # Recursions (backwards over j): X <- L_j^{-1} X (exact undo), R <- L_j^\dagger R.
    def force(self, P):
        X = self.forward(P)  # = A P = P_N
        R = g.copy(X)  # R_N = A P
        F = [g.lattice(P[0]) for nu in range(self.nd)]
        for f in F:
            f[:] = 0
        for mu, parity in reversed(self.sequence()):
            self.layer(X, mu, parity, -self.eta)  # X = P_{j-1}
            Rm = g(self.mask[parity] * R[mu])
            Xm = X[mu]
            for nu in range(self.nd):
                if nu == mu:
                    continue
                G1 = g(self.U[nu] * g.cshift(Xm, nu, 1) * g.adj(self.U[nu]))
                G2 = g(self.U[nu] * g.cshift(Rm, nu, 1) * g.adj(self.U[nu]))
                F[nu] += (1j * self.eta) * (G1 * Rm - Rm * G1)
                F[nu] -= (1j * self.eta) * (Xm * G2 - G2 * Xm)
            self.layer(R, mu, self.swap[parity], self.eta)  # R = R_{j-1} = L_j^\dagger R_j
        return F


# Kinetic term H_kin(U, P) = (1/2) <A(U) P, A(U) P> as a gpt differentiable functional on fields = U + P
# (same calling convention as gpt's general_mass_term).
#   draw:            P = A^{-1} \xi,  \xi ~ rng.normal_element   =>  A P = \xi exactly
#   gradient wrt P:  A^\dagger A P   (velocity:  U <- exp(i \varepsilon K P) U,  K = A^\dagger A)
#   gradient wrt U:  transverse_smear.force(P)
class transverse_kinetic(g.group.differentiable_functional):
    def __init__(self, eta, niter):
        self.eta = eta
        self.niter = niter
        self.mask = None
        self.__name__ = f"transverse_kinetic(eta={eta}, niter={niter})"

    def kernel(self, U):
        if self.mask is None:
            even, odd = g.even_odd_projectors(U[0].grid)
            self.mask = {"even": even, "odd": odd}
        return transverse_smear(U, self.eta, self.niter, self.mask)

    def split(self, fields):
        n = len(fields)
        assert n % 2 == 0
        return fields[: n // 2], fields[n // 2 :]

    def __call__(self, fields):
        U, P = self.split(fields)
        AP = self.kernel(U).forward(P)
        return 0.5 * sum(g.group.inner_product(x, x) for x in AP)

    def draw(self, fields, rng):
        U, P = self.split(fields)
        xi = g.copy(P)
        rng.normal_element(xi)
        P_new = self.kernel(U).inverse(xi)
        for mu in range(len(P)):
            P[mu] @= P_new[mu]
        return 0.5 * sum(g.group.inner_product(x, x) for x in xi)

    @g.group.differentiable_functional.multi_field_gradient
    def gradient(self, fields, dfields):
        U, P = self.split(fields)
        A = self.kernel(U)
        need_P = any(any(f is p for p in P) for f in dfields)
        need_U = any(any(f is u for u in U) for f in dfields)
        KP = A.adjoint(A.forward(P)) if need_P else None
        FU = A.force(P) if need_U else None
        dS = []
        for f in dfields:
            mu = next(i for i, x in enumerate(fields) if x is f)
            dS.append(FU[mu] if mu < len(U) else KP[mu - len(U)])
        return dS
