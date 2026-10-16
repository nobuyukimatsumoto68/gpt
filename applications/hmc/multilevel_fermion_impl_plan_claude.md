# Multilevel (Sexton-Weingarten) implicit leapfrog with a fermion force: implementation plan

Created 2026-10-16 by genesis-6c, from NM's note `genesis/exact_rmhmc-1.pdf` (reply to referee, item 5, Algorithm 1, eqs. (9)-(22)).

**Sources (cite in code):**
- multilevel integration: J. Sexton, D. Weingarten, Nucl. Phys. B 380, 665 (1992);
- hierarchical RMHMC: T. Nguyen, P. Boyle, N. H. Christ, Y.-C. Jang, C. Jung, PoS LATTICE2021, 582 (2022), arXiv:2112.04556;
- RMHMC with fermions: C. Jung, N. H. Christ, PoS LATTICE2023, 009 (2024), arXiv:2401.13226;
- exact extended framework: N. H. Christ, L.-C. Jin, C. Lehner, E. Lundstrum, N. Matsumoto, PRD 112, 034507 (2025), arXiv:2412.19904;
- generalized leapfrog / symplectic Euler: B. Leimkuhler, S. Reich, *Simulating Hamiltonian Dynamics* (CUP, 2004); M. Girolami, B. Calderhead, J. R. Stat. Soc. B 73, 123 (2011);
- pseudofermions / HMC: S. Duane, A. D. Kennedy, B. J. Pendleton, D. Roweth, Phys. Lett. B 195, 216 (1987);
- clover term: B. Sheikholeslami, R. Wohlert, Nucl. Phys. B 259, 572 (1985).

## Goal
- Add a two-flavour pseudofermion $S_f = \phi^\dagger (D^\dagger D)^{-1}\phi$ to the trainable-kernel RMHMC.
- Integrate it with the multilevel scheme of Algorithm 1:
  - fermion force $F_f$ on the outer level (explicit kicks);
  - kernel force $F_K$, gauge force $F_g$ and $\det K$ pseudofermion force $F_d$ on the inner, implicit level.
- The fermion solve cost is then additive to the implicit iteration (one $\eta = (D^\dagger D)^{-1}\phi$ per outer step).
- Physics question (NM): the fermion force carries the quark-mass scale, the cost-relevant scale. How does the kernel $K$ align with, or interact with, the fermion force spectrum?

## Algorithm 1 (NM) and the inner loops

$$
H = \tfrac12 p\cdot K^{-1}(w)\cdot p + S_g(w) + S_f(w;\phi) + S_d(w;\varphi),\qquad S_d = \tfrac12\varphi^T K(w)\varphi .
$$

Our code already has $S_d$: `poly_logdet` / `action_with_R` with $\tfrac12\langle R, A^{-2}R\rangle$, where our $K^{-1}_\text{NM} = A^\dagger A$, so $K_\text{NM} = A^{-2}$.

With $h = \tau/(2n)$:
- outer: $p_0 = p - \tfrac\tau2 F_f(w)$;
- first inner loop, (18)-(19): $p_{i+1} = p_i - h\,F(w_i, p_{i+1})$ (implicit in $p$), then $w_{i+1} = w_i + h\,K^{-1}(w_i)\,p_{i+1}$ (explicit). This is the symplectic Euler map $\Phi_h$ ("Euler A").
- second inner loop, (20)-(21), as written: $w'_{i+1} = w'_i + h\,K^{-1}(w'_i)\,p'_{i+1}$ and $p'_{i+1} = p'_i - h\,F(w'_{i+1}, p'_{i+1})$.
- outer: $p' = p'_n - \tfrac\tau2 F_f(w')$.

**Q1: the second loop as written is not the adjoint of the first and is not volume preserving.**

Separable check: $K = 1$, $F = w$ (harmonic oscillator). Then (20)-(21) give $p_1 = (p - hw)/(1+h^2)$ and $w_1 = (w + hp)/(1+h^2)$:

$$
\begin{pmatrix} w_1\\ p_1\end{pmatrix} = \frac{1}{1+h^2}\begin{pmatrix} 1 & h\\ -h & 1\end{pmatrix}\begin{pmatrix} w\\ p\end{pmatrix},\qquad \det = \frac{1}{1+h^2}\neq 1 .
$$

The adjoint of Euler A, i.e. the time reverse with $p \to -p$, using $F_K$ even in $p$, is the other symplectic Euler map ("Euler B"):

$$
w'_{i+1} = w'_i + h\,K^{-1}(w'_{i+1})\,p'_i \quad(\text{implicit in } w),\qquad
p'_{i+1} = p'_i - h\big[F_K(w'_{i+1}, p'_i) + F_g(w'_{i+1}) + F_d(w'_{i+1})\big] \quad(\text{explicit}).
$$

- Then $B_h^n \circ A_h^n$ is symmetric, so the scheme is reversible.
- With $n = 1$ it is exactly our current generalized leapfrog step (`implicit_leapfrog_claude.step`, eps $= \tau$), with fermion half-kicks around it.
- The implicit equations are (18) (in $p$) and (20) (in $w$; on the group, the product-form update), not (18) and (21).

**Proposed reading (pending NM): (20)-(21) := Euler B as above.**

**Q2: inner ordering $B^n A^n$ (NM) vs the standard Sexton-Weingarten $(B A)^n$.**
Both are symmetric and second order, with the same cost: $2n$ half-steps and $n$ implicit $p$ and $n$ implicit $w$ solves per outer step. Their inner (gauge-force) splitting errors scale differently:
- $(BA)^n$: $n$ leapfrog steps of size $\tau/n$, leading inner error $\propto \tau^3/n^2$;
- $B^n A^n$: $A^n = \exp\big(\tfrac\tau2 X + \tfrac{\tau^2}{8n}E + \dots\big)$ and $B^n$ has $-E$, so after BCH the leading inner error is $\propto \tau^3/n$ (a commutator $[X, E]$).

Recommendation: implement both, with $(BA)^n$ as the default because it reuses `implicit_leapfrog_claude` unchanged. Compare $\langle\Delta H\rangle$ vs $n$ in the tests.

**Q3: group (inexact) or the $W$ embedding (exact)?**
The note is written in the $W \in M_N(\mathbb C)$ framework (exact). Our trainable-kernel code integrates on SU(3) (product-form $U$ update), which is not volume preserving with a $U$-dependent kernel.
Proposal: first build the multilevel scheme on our SU(3) code. The fermion question (scales, cost) does not depend on exactness; label results "inexact". The $W$-space version is a separate project (it needs the $(U,\theta,\Phi)$ decomposition, $S_0(\Phi)$, and the Jacobian force).

**Q4: fermion discretization and mass.**
Proposal: $N_f = 2$ Wilson-clover (gpt `g.qcd.fermion.wilson_clover`, $c_{sw}$ tree level $= 1$ or a nonperturbative guess, antiperiodic in time), with the even-odd Schur pseudofermion `g.qcd.pseudofermion.action.two_flavor_evenodd_schur`, CG via `g.algorithms.inverter.cg`, and heavy quarks first ($m_\pi L \gtrsim 4$ on $8^4$ is unrealistic; start with $\kappa$ well below critical, e.g. $am \approx 0.1$-$0.3$).
Alternatives: Mobius DWF (gpt `applications/hmc/dwf/ensemble*.py` has the full setup; more expensive) or staggered/rational (not now). The Iwasaki $\beta = 2.95$ gauge action stays (not retuned for $N_f = 2$; exploration only).

**Q5: what to measure first (scale question).** Proposal:
1. Force norms per level: $|F_f|$, $|F_g|$, $|F_K|$, $|F_d|$ (rms per link) vs quark mass, with and without the poly kernel. This shows which level dominates the step size.
2. $\langle\Delta H\rangle$ (or $\langle\Delta H^2\rangle$) vs (outer $\tau$, inner $n$): find the outer step at which the fermion level limits acceptance, for kernel = identity / analytic $N=8$ / trained $N=8$.
3. Spectrum of the fermion force: the linearized fermion force $h_f = \partial F_f$ by finite differences (`hessian_fd` on $S_f$ with fixed $\phi$). Measure the shell-resolved $\omega_j^2$ of $A h_f A^\dagger$ as in entry (18) and check whether the quark mass sets an IR scale in $h_f$ (the cost-relevant scale).
4. U-turn $C_j(t)$ with fermions (decorrelation driver), the same metric as before.

## Files
- **New** `gpt/applications/hmc/multilevel_leapfrog_claude.py`: class `multilevel_leapfrog(inner, fermion_action, n_inner, order="BA"|"BnAn")`. Outer explicit half-kicks with $F_f$; the inner level uses `implicit_leapfrog_claude.implicit_leapfrog` for "BA", or new `step_A` / `step_B` methods for "BnAn".
- **Edit** `gpt/applications/hmc/implicit_leapfrog_claude.py`: add the side-by-side methods `step_A(U, P, h)` and `step_B(U, P, h)` (Euler A/B on the group). `step` stays untouched.
- **New** `gpt/applications/hmc/fermion_setup_claude.py`: Wilson-clover operator + two-flavour even-odd pseudofermion + CG; `draw`, action, force; antiperiodic time BC.
- **New** `gpt/applications/hmc/test_multilevel_claude.py`:
  - fermion force vs finite difference;
  - reversibility of the full multilevel step (forward, flip $p$, back; $10^{-12}$ expected up to CG tolerance);
  - $\Delta H \propto \tau^2$ scaling;
  - $n = 1$ "BA" reproduces `implicit_leapfrog.step` plus half-kicks exactly;
  - Euler A/B adjointness on the group.
- **New** `gpt/applications/hmc/hmc_fermion_claude.py`: HMC driver (draw $p$, $R$, $\phi$; multilevel MD; Metropolis), logging the per-level force norms and solver iterations.
- **Edit** `gpt/applications/hmc/decorrelation_claude.py`: `--fermion` option (U-turn with fermions); old lines kept.
- **New** `gpt/applications/hmc/aurora_fermion_claude.pbs` + `_submit_claude.sh` (written with aurora-build-script-setup).

## Chunks
1. **Fermion setup + force test** (local, $4^4$). Files: `fermion_setup_claude.py`, `test_multilevel_claude.py` (force test part).
2. **Euler A/B steps + multilevel integrator + reversibility/$\Delta H$ tests** (local, $4^4$ and $4^3\times8$). Files: `implicit_leapfrog_claude.py` (new methods), `multilevel_leapfrog_claude.py`, `test_multilevel_claude.py`.
3. **HMC driver + force-norm/$\Delta H$ diagnostics** (local smoke test). Files: `hmc_fermion_claude.py`, `decorrelation_claude.py`.
4. **Aurora** (NM spend OK): $8^4$; kernels identity / analytic $N=8$ / trained $N=8$; 2-3 quark masses; force norms, $\Delta H$ vs $(\tau, n)$, $h_f$ shell spectrum, U-turn. Files: `aurora_fermion_claude.pbs`, `_submit_claude.sh`.

## Open questions for NM
- Q1: second loop = Euler B (adjoint of (18)-(19))? (The note's (20)-(21) are not volume preserving as written.)
- Q2: inner ordering: $(BA)^n$ default with $B^nA^n$ as an option, or only $B^nA^n$?
- Q3: SU(3) (inexact) first, $W$ framework later?
- Q4: Wilson-clover $N_f = 2$, heavy masses first? Which $\kappa$ / $c_{sw}$? Or DWF?
- Q5: is the measurement list right, and in which order?

## Update 2026-10-16: NM's revised Algorithm 1 (interleaved; resolves Q1/Q2)
NM's revision interleaves the halves inside one loop over $i$ (standard Sexton-Weingarten $(BA)^n$):

$$
\tilde p_{i+1} = p_i - h\big[F_K(w_i,\tilde p_{i+1}) + F_g(w_i) + F_d(w_i)\big],\qquad
\tilde w_{i+1} = w_i + h\,K^{-1}(w_i)\,\tilde p_{i+1},
$$

$$
w_{i+1} = \tilde w_{i+1} + h\,K^{-1}(w_{i+1})\,\tilde p_{i+1},\qquad
p_{i+1} = \tilde p_{i+1} - h\big[F_K(w_{i+1},\tilde p_{i+1}) + F_g(w_{i+1}) + F_d(w_{i+1})\big],
$$

with $h = \tau/(2n)$.

- **Fix (proposed to NM):** the RHS of the last two equations must use $\tilde p_{i+1}$. With $p_{i+1}$, as in the draft, the half-step has $\det = 1/(1+h^2)$ for $K=1$, $F=w$.
- With the fix, each iteration is the generalized leapfrog (implicit: $\tilde p$ and $w_{i+1}$; explicit: $p_{i+1}$). On the group it is exactly `implicit_leapfrog_claude.step(U, P, eps = tau/n)`: step (2) is the product form of the $\tilde w$ and $w_{i+1}$ equations.
- Notation: after the loop, $w' \equiv w_n$ and $p' = p_n - \tfrac\tau2 F_f(w')$.
- Consequences for the files: no new Euler A/B methods are needed, and `implicit_leapfrog_claude.py` is unchanged. `multilevel_leapfrog_claude.py` is outer $F_f$ half-kicks around `implicit_leapfrog.__call__(U, P, tau, n)`, which must handle the $R$/$F_d$ term as now. The $B^nA^n$ option is dropped.

## NM decisions (2026-10-16)
- Algorithm: NM's revised Algorithm 1 with the $\tilde p_{i+1}$ fix (confirmed); interleaved $(BA)^n$.
- Q3: **SU(3) first** (inexact, reusing `implicit_leapfrog_claude`); $W$ framework later.
- Q4: **$N_f = 2$ plain Wilson** (no clover), antiperiodic in time, heavy masses first.
- Chunk 1 started.
