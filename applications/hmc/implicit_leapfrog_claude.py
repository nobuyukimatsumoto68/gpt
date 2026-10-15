#
# Hand-coded implicit (generalized) leapfrog for the non-separable RMHMC Hamiltonian
#   H(U, P) = H_kin(U, P) + S(U),   H_kin = (1/2) <A(U) P, A(U) P>   (transverse_kinetic)
# Design and equations: genesis/rmhmc_trainable_kernel_impl_plan_claude.md, "Implementation log" entry (4).
# Sources:
#   generalized leapfrog: M. Girolami, B. Calderhead, J. R. Stat. Soc. B 73 (2011) 123;
#     B. Leimkuhler, S. Reich, Simulating Hamiltonian Dynamics (CUP, 2004)
#   CPN prototype: julia_in_script2/header_claude.jl (implicit_update_p!, implicit_update_phi!)
#
# One step of size \varepsilon (a = \varepsilon / 2), gpt conventions (U <- exp(i W) U, W hermitian):
#   (1) P_h = P_0 - a F(U_0, P_h)                                      fixed point in P_h
#   (2) U_1 = exp(i a V(U_1, P_h)) exp(i a V(U_0, P_h)) U_0            fixed point in U_1
#   (3) P_1 = P_h - a F(U_1, P_h)                                      explicit
# with V = \partial_P H = A^\dagger A P (velocity) and F = \partial_U H (force; gpt gradient convention).
# The product form in (2) makes the step exactly reversible: (U_1, -P_1) -> (U_0, -P_0).
# TODO (NM): Aitken / Steffensen acceleration of the fixed points (CPN implicit_update_p!).
#
import gpt as g


import numpy as np


class implicit_leapfrog:
    def __init__(self, kinetic, action, tol=1e-12, max_iter=100, verbose=False, use_aitken=False, aitken_mode="vector", cache_action=False, reuse_action=True):
        self.kinetic = kinetic
        # cache_action: compute the U-only action force once per U in step() (it does not depend on P_h);
        # default False keeps the original behaviour. Useful when the action force is expensive (e.g. the R / log det term).
        self.cache_action = cache_action
        self.reuse_action = reuse_action  # with cache_action: reuse the U_1 action force as the next step's U_0 force
        self.action = action
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.use_aitken = use_aitken  # False: original plain fixed point (step); True: step_aitken
        self.aitken_mode = aitken_mode  # "componentwise" (as CPN) or "vector" (one global contraction factor)
        self.niter_log = []  # (n_P, n_U) per step

    def hamiltonian(self, U, P):
        return self.kinetic(U + P) + self.action(U)

    def force(self, U, P):
        F_S = self.action.gradient(U, U)
        F_K = self.kinetic.gradient(U + P, U)
        return [g(a + b) for a, b in zip(F_S, F_K)]

    def velocity(self, U, P):
        return self.kinetic.gradient(U + P, P)

    def rms(self, X, Y):
        return (sum(g.norm2(x - y) for x, y in zip(X, Y)) / (len(X) * X[0].grid.gsites)) ** 0.5

    # exp(i a W) U, link by link
    def compose(self, a, W, U):
        return [g(g.group.compose(g(a * w), u)) for w, u in zip(W, U)]

    def step(self, U, P, eps):
        a = 0.5 * eps

        # (1) P_h = P_0 - a F(U_0, P_h)
        P0 = g.copy(P)
        Ph = g.copy(P)
        # FS0 = self.action.gradient(U, U) if self.cache_action else None
        if self.cache_action:
            # reuse the action force at U_1 of the previous step (same U; valid within one __call__ only)
            FS0 = self._FS_next if (self.reuse_action and getattr(self, "_FS_next", None) is not None) else self.action.gradient(U, U)
        else:
            FS0 = None
        for n_P in range(1, self.max_iter + 1):
            # F = self.force(U, Ph)
            F = self.force(U, Ph) if FS0 is None else [g(fs + fk) for fs, fk in zip(FS0, self.kinetic.gradient(U + Ph, U))]
            Pn = [g(p0 - a * f) for p0, f in zip(P0, F)]
            res = self.rms(Pn, Ph)
            Ph = Pn
            if res < self.tol:
                break
        else:
            raise RuntimeError(f"implicit P update did not converge: res = {res}")

        # (2) U_1 = exp(i a V(U_1, P_h)) W_0,   W_0 = exp(i a V(U_0, P_h)) U_0
        W0 = self.compose(a, self.velocity(U, Ph), U)
        U1 = g.copy(W0)
        for n_U in range(1, self.max_iter + 1):
            Un = self.compose(a, self.velocity(U1, Ph), W0)
            res = self.rms(Un, U1)
            U1 = Un
            if res < self.tol:
                break
        else:
            raise RuntimeError(f"implicit U update did not converge: res = {res}")

        # (3) P_1 = P_h - a F(U_1, P_h)
        # F = self.force(U1, Ph)
        if self.cache_action:
            FS1 = self.action.gradient(U1, U1)
            F = [g(fs + fk) for fs, fk in zip(FS1, self.kinetic.gradient(U1 + Ph, U1))]
        else:
            F = self.force(U1, Ph)
        for mu in range(len(P)):
            U[mu] @= U1[mu]
            P[mu] @= Ph[mu] - a * F[mu]
        if self.cache_action:
            self._FS_next = FS1  # action force at the new U (= U_1) for the next step

        self.niter_log.append((n_P, n_U))
        if self.verbose:
            g.message(f"implicit_leapfrog step: n_P = {n_P}, n_U = {n_U}")

    # ---------------- Aitken / Steffensen-accelerated variant ----------------
    # Same scheme as step(); the fixed points are accelerated by Aitken's \Delta^2 process
    # (Steffensen's method), as in the CPN prototype implicit_update_p! (julia_in_script2/header_claude.jl:524):
    #   x_1 = f(x), x_2 = f(x_1),  x <- x - (x_1 - x)^2 / (x_2 - 2 x_1 + x)   (componentwise; x_2 where |denom| < 1e-14)
    # Applied to algebra-valued unknowns only:
    #   P step:  x = P_h,  f(x) = P_0 - a F(U_0, x)
    #   U step:  x = V_1,  f(x) = V(exp(i a x) W_0, P_h),  then U_1 = exp(i a V_1) W_0
    # (iterating on V_1 instead of U_1 keeps the unknown in a linear space; same iterates as step()).
    # The componentwise result is projected back to traceless hermitian.

    # vector Aitken: one global contraction factor \lambda = <d_1, d_0> / <d_0, d_0>,
    #   d_0 = x_1 - x_0, d_1 = x_2 - x_1;   x <- x_2 + \lambda / (1 - \lambda) d_1   (x_2 if not 0 < \lambda < 1)
    def aitken_vector(self, x0, x1, x2):
        d0 = [g(b - a) for a, b in zip(x0, x1)]
        d1 = [g(b - a) for a, b in zip(x1, x2)]
        n00 = sum(g.group.inner_product(a, a) for a in d0)
        lam = sum(g.group.inner_product(b, a) for a, b in zip(d0, d1)) / n00 if n00 > 0 else 0.0
        if not 0.0 < lam < 1.0:
            return x2
        c = lam / (1.0 - lam)
        return [g(b + c * d) for b, d in zip(x2, d1)]

    def aitken(self, x0, x1, x2):
        if self.aitken_mode == "vector":
            return self.aitken_vector(x0, x1, x2)
        res = []
        for a0, a1, a2 in zip(x0, x1, x2):
            v0, v1, v2 = a0[:], a1[:], a2[:]
            denom = v2 - 2.0 * v1 + v0
            safe = np.abs(denom) > 1e-14
            out = v2.copy()
            out[safe] = v0[safe] - (v1[safe] - v0[safe]) ** 2 / denom[safe]
            r = g.lattice(a0)
            r[:] = out
            # project to traceless hermitian (componentwise Aitken keeps hermiticity, not exact tracelessness)
            r @= 0.5 * (r + g.adj(r))
            r @= r - g.identity(r) * (g.trace(r) / r.otype.Nc)
            res.append(r)
        return res

    def fixed_point(self, f, x, what):
        n = 0
        while True:
            x1 = f(x)
            n += 1
            if self.rms(x1, x) < self.tol:
                return x1, n
            if n >= self.max_iter:
                break
            x2 = f(x1)
            n += 1
            if self.rms(x2, x1) < self.tol:
                return x2, n
            if n >= self.max_iter:
                break
            x = self.aitken(x, x1, x2)
        raise RuntimeError(f"implicit {what} update (Aitken) did not converge in {n} iterations")

    def step_aitken(self, U, P, eps):
        a = 0.5 * eps
        P0 = g.copy(P)

        # (1) P_h = P_0 - a F(U_0, P_h)
        Ph, n_P = self.fixed_point(
            lambda x: [g(p0 - a * f) for p0, f in zip(P0, self.force(U, x))], g.copy(P), "P"
        )

        # (2) V_1 = V(exp(i a V_1) W_0, P_h),  U_1 = exp(i a V_1) W_0
        W0 = self.compose(a, self.velocity(U, Ph), U)
        V1, n_U = self.fixed_point(
            lambda x: self.velocity(self.compose(a, x, W0), Ph), self.velocity(W0, Ph), "U"
        )
        U1 = self.compose(a, V1, W0)

        # (3) P_1 = P_h - a F(U_1, P_h)
        F = self.force(U1, Ph)
        for mu in range(len(P)):
            U[mu] @= U1[mu]
            P[mu] @= Ph[mu] - a * F[mu]

        self.niter_log.append((n_P, n_U))
        if self.verbose:
            g.message(f"implicit_leapfrog step_aitken: n_P = {n_P}, n_U = {n_U}")

    # callback(k, U, P) is called after MD step k = 1..nstep (optional; for measurements along the trajectory)
    def __call__(self, U, P, tau, nstep, callback=None):
        eps = tau / nstep
        self._FS_next = None  # cross-step action-force reuse is only valid within one trajectory
        for i in range(nstep):
            # self.step(U, P, eps)
            if self.use_aitken:
                self.step_aitken(U, P, eps)
            else:
                self.step(U, P, eps)
            if callback is not None:
                callback(i + 1, U, P)
