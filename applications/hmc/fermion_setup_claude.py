#
# N_f = 2 plain Wilson pseudofermion for the multilevel RMHMC (NM 2026-10-16; plan:
# gpt/applications/hmc/multilevel_fermion_impl_plan_claude.md, chunk 1).
#   S_f(U; phi) = phi^dagger (Mhat Mhat^dagger)^{-1} phi,   Mhat = M_ee - M_eo M_oo^{-1} M_oe  (even-odd Schur complement)
#   plain Wilson (c_sw = 0): M_oo = 1 (kappa normalization), so det M = det Mhat up to a constant -> exact N_f = 2.
#   boundary: antiperiodic in time.
# Built on gpt's g.qcd.fermion.wilson_clover (csw = 0) and g.qcd.pseudofermion.action.two_flavor_evenodd_schur
#   (C. Lehner, M. Bruno), CG inverter.
# Sources: pseudofermions / HMC: S. Duane, A. D. Kennedy, B. J. Pendleton, D. Roweth, Phys. Lett. B 195, 216 (1987);
#   Wilson fermions: K. G. Wilson, in New Phenomena in Subnuclear Physics (1977); even-odd preconditioning:
#   T. DeGrand, P. Rossi, Comput. Phys. Commun. 60, 211 (1990); CG: M. R. Hestenes, E. Stiefel, J. Res. NBS 49, 409 (1952).
#
# Interface matches the U-only actions used by implicit_leapfrog_claude / hmc drivers:
#   fa = wilson_nf2(U, kappa); fa.draw(U, rng); fa(U); fa.gradient(U, U)  (gpt gradient convention, list over mu)
# phi is held by the object (background field during a trajectory).
#
import gpt as g


class wilson_nf2:
    def __init__(self, U, kappa, cg_tol=1e-10, cg_maxiter=10000):
        self.kappa = kappa
        self.params = {
            "kappa": kappa,
            "csw_r": 0.0,
            "csw_t": 0.0,
            "xi_0": 1,
            "nu": 1,
            "isAnisotropic": False,
            "boundary_phases": [1.0, 1.0, 1.0, -1.0],
        }
        self.M = g.qcd.fermion.wilson_clover(U, self.params)
        self.cg = g.algorithms.inverter.cg({"eps": cg_tol, "maxiter": cg_maxiter})
        self.action = g.qcd.pseudofermion.action.two_flavor_evenodd_schur(self.M, self.cg)
        self.phi = g.vspincolor(self.M.F_grid_eo)  # odd checkerboard (gpt Schur convention)
        self.phi.checkerboard(g.odd)
        self.phi[:] = 0
        self.n_solve = 0

    def fields(self, U):
        return list(U) + [self.phi]

    # heat-bath: phi = Mhat eta, eta Gaussian -> S_f = |eta|^2 (returned)
    def draw(self, U, rng):
        return self.action.draw(self.fields(U), rng)

    def __call__(self, U):
        self.n_solve += 1
        return self.action(self.fields(U))

    def gradient(self, U, dU):
        self.n_solve += 1
        return self.action.gradient(self.fields(U), dU)
