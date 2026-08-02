"""Nonlinear axisymmetric magnetostatic FEM for electropermanent magnet cores.

Why this exists
---------------
``simulations/Force_compute/python/magnet_force.py`` solves magnet-in-free-space
problems exactly, but cannot represent soft-magnetic material.  Every EPM in the
literature relies on a soft-magnetic flux path, so evaluating any EPM geometry
needs a solver that handles nonlinear iron.  This module provides one, and
``verify_fem.py`` cross-validates it against the exact solver in the limit where
the iron is removed.

Formulation
-----------
Axisymmetric magnetostatics in the azimuthal vector potential ``A = A_phi``:

    B_r = -dA/dz            B_z = A/r + dA/dr

    int nu [ B_r(A) B_r(v) + B_z(A) B_z(v) ] r dr dz
        = int nu [ Br_r B_r(v) + Br_z B_z(v) ] r dr dz

Nonlinear solution strategy
---------------------------
Two nested loops, because the two nonlinearities behave very differently.

* Soft iron is mild and monotone, so an inner damped fixed-point on the
  reluctivity nu(|B|) converges reliably.

* A low-coercivity permanent magnet is stiff: near the knee dJ/dH greatly
  exceeds the recoil slope, so the implied feedback gain of a fixed-point on the
  magnet state exceeds one and it cannot converge.  Each magnet is therefore
  divided into a small number of axial slabs of uniform remanence, and those few
  values are solved by damped Newton with a finite-difference Jacobian, with a
  continuation fallback.  This mirrors the structure of the validated
  free-space rod solver, so the two can be compared directly.

Meshing
-------
All geometries here are unions of axis-aligned rectangles, so the mesh is a
graded tensor grid whose node coordinates land exactly on every material
boundary.  No element straddles two materials, which removes staircase error
without needing an external mesh generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import splu
from skfem import (Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri,
                   asm, condense, solve)
from skfem.helpers import grad

MU0 = 4.0e-7 * np.pi

# --------------------------------------------------------------------------
# Soft-magnetic material: AISI 1018 low-carbon steel, a standard EPM pole
# material.  Tabulated normal B-H curve.
# --------------------------------------------------------------------------
STEEL_1018_H = np.array([0., 159., 318., 477., 636., 795., 1590., 3180.,
                         4770., 6360., 7950., 15900., 31800., 47700.,
                         63600., 79500., 159000., 318000.])
STEEL_1018_B = np.array([0., 0.75, 1.05, 1.20, 1.28, 1.34, 1.48, 1.58,
                         1.63, 1.66, 1.68, 1.75, 1.83, 1.88,
                         1.92, 1.95, 2.05, 2.20])


def steel_nu(Bmag):
    """Reluctivity nu = H/B of 1018 steel at flux density |B|."""
    B = np.maximum(np.asarray(Bmag, dtype=float), 1e-6)
    H = np.interp(B, STEEL_1018_B, STEEL_1018_H)
    tail = B > STEEL_1018_B[-1]
    if np.any(tail):
        H = np.where(tail, STEEL_1018_H[-1] + (B - STEEL_1018_B[-1]) / MU0, H)
    return H / B


# --------------------------------------------------------------------------
@dataclass
class Region:
    """An axis-aligned rectangle in the (r, z) half-plane.

    ``kind`` is 'air', 'steel' or 'magnet'.  For magnets, ``material`` is a
    ``magnet_force.Material`` and ``direction`` is +1 or -1 for magnetisation
    along +z or -z.
    """

    rmin: float
    rmax: float
    zmin: float
    zmax: float
    kind: str
    name: str = ""
    material: object = None
    direction: float = 1.0
    mu_r: float = None          # override: constant permeability for steel

    def contains(self, r, z):
        return ((r >= self.rmin - 1e-15) & (r <= self.rmax + 1e-15) &
                (z >= self.zmin - 1e-15) & (z <= self.zmax + 1e-15))


# --------------------------------------------------------------------------
def _graded_axis(breaks, target_h, pad_to=None, growth=1.3):
    """Node coordinates hitting every break point exactly, graded outward."""
    breaks = np.unique(np.asarray(breaks, dtype=float))
    pts = [breaks[0]]
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        n = max(int(np.ceil((hi - lo) / target_h)), 1)
        pts.extend(np.linspace(lo, hi, n + 1)[1:])
    pts = list(np.array(pts))
    if pad_to is not None:
        for sign, limit in ((+1, pad_to[1]), (-1, pad_to[0])):
            cur = pts[-1] if sign > 0 else pts[0]
            h = target_h
            out = []
            while (limit - cur) * sign > 1e-12:
                h *= growth
                cur = cur + sign * h
                if (cur - limit) * sign > 0:
                    cur = limit
                out.append(cur)
            pts = pts + out if sign > 0 else out[::-1] + pts
    return np.unique(np.array(pts))


# --------------------------------------------------------------------------
@dataclass
class AxisymSolution:
    A: np.ndarray
    Br_elem: np.ndarray
    Bz_elem: np.ndarray
    nu_elem: np.ndarray
    Brz_elem: np.ndarray
    tag: np.ndarray
    slab: np.ndarray
    mesh: object
    basis: object
    regions: list
    slab_H: np.ndarray
    slab_J: np.ndarray
    slab_region: np.ndarray
    iterations: int = 0
    residual: float = np.inf

    def magnet_state(self, name):
        """Volume-averaged (J, H) along the axis for a named magnet region."""
        idx = [i for i, g in enumerate(self.regions) if g.name == name][0]
        m = self.tag == idx
        w = _elem_volume(self.mesh)[m]
        g = self.regions[idx]
        mu = 1.0 / (MU0 * self.nu_elem[m])
        Hz = (self.Bz_elem[m] - self.Brz_elem[m]) / (MU0 * mu)
        Jz = self.Bz_elem[m] - MU0 * Hz
        return (float(np.sum(Jz * w) / np.sum(w)),
                float(np.sum(Hz * w) / np.sum(w)))


def _elem_volume(mesh):
    p, t = mesh.p, mesh.t
    x, y = p[0][t], p[1][t]
    area = 0.5 * np.abs((x[1] - x[0]) * (y[2] - y[0]) -
                        (x[2] - x[0]) * (y[1] - y[0]))
    return 2 * np.pi * x.mean(axis=0) * area


# --------------------------------------------------------------------------
class AxisymModel:
    """Mesh, material tagging and nonlinear solve for one geometry."""

    def __init__(self, regions, r_far, z_far, h, n_slabs=8):
        self.regions = list(regions)
        self.n_slabs = n_slabs

        rb, zb = {0.0}, set()
        for g in self.regions:
            rb.update((g.rmin, g.rmax))
            zb.update((g.zmin, g.zmax))
            if g.kind == "magnet":
                zb.update(np.linspace(g.zmin, g.zmax, n_slabs + 1))
        rs = _graded_axis(sorted(rb), h, pad_to=(0.0, r_far))
        zs = _graded_axis(sorted(zb), h, pad_to=(-z_far, z_far))

        self.mesh = MeshTri.init_tensor(rs, zs)
        self.basis = Basis(self.mesh, ElementTriP1())
        self.vol = _elem_volume(self.mesh)

        rc = self.mesh.p[0][self.mesh.t].mean(axis=0)
        zc = self.mesh.p[1][self.mesh.t].mean(axis=0)
        self.tag = np.full(self.mesh.t.shape[1], -1, dtype=int)
        self.slab = np.full(self.mesh.t.shape[1], -1, dtype=int)

        slab_region = []
        k = 0
        for i, g in enumerate(self.regions):
            inside = g.contains(rc, zc)
            self.tag[inside] = i
            if g.kind == "magnet":
                edges = np.linspace(g.zmin, g.zmax, n_slabs + 1)
                for s in range(n_slabs):
                    m = inside & (zc >= edges[s]) & (zc <= edges[s + 1])
                    self.slab[m] = k
                    slab_region.append(i)
                    k += 1
        self.n_slab_total = k
        self.slab_region = np.array(slab_region, dtype=int)

        # cached geometry for field recovery
        gc = self.basis.global_coordinates().value
        self.rq, self.zq = gc[0], gc[1]
        self.nq = self.rq.shape[1]

        self.D = self.basis.get_dofs(
            lambda x: (np.abs(x[0]) < 1e-14) |
                      (np.abs(x[0] - rs[-1]) < 1e-12) |
                      (np.abs(x[1] - zs[0]) < 1e-12) |
                      (np.abs(x[1] - zs[-1]) < 1e-12))

        self._steel_mask = np.isin(
            self.tag, [i for i, g in enumerate(self.regions)
                       if g.kind == "steel" and g.mu_r is None])
        self._nu_cache = None

    # ---- linear solve ----------------------------------------------------
    def _forms(self):
        if getattr(self, "_form_cache", None) is None:
            @BilinearForm
            def bilin(u, v, w):
                r = np.maximum(w.x[0], 1e-12)
                gu, gv = grad(u), grad(v)
                return w["nu"] * ((-gu[1]) * (-gv[1]) +
                                  (u / r + gu[0]) * (v / r + gv[0])) * r

            @LinearForm
            def lin(v, w):
                r = np.maximum(w.x[0], 1e-12)
                gv = grad(v)
                return w["nu"] * w["Brz"] * (v / r + gv[0]) * r

            self._form_cache = (bilin, lin)
        return self._form_cache

    def _q(self, v):
        return np.tile(v[:, None], (1, self.nq))

    def _factorise(self, nu):
        """Factorise the stiffness matrix once for a given reluctivity field.

        Every slab of every magnet drives the same operator; only the source
        term changes.  Factorising once and back-substituting is what makes the
        permeance matrix below affordable.
        """
        bilin, lin = self._forms()
        K = asm(bilin, self.basis, nu=self._q(nu))
        f0 = asm(lin, self.basis, nu=self._q(nu),
                 Brz=self._q(np.zeros(self.mesh.t.shape[1])))
        Kc, fc, xx, I = condense(K, f0, D=self.D)
        lu = splu(Kc.tocsc())
        return lu, I, xx.shape[0]

    def _rhs(self, nu, Brz):
        _, lin = self._forms()
        return asm(lin, self.basis, nu=self._q(nu), Brz=self._q(Brz))

    def _fields_from_A(self, A):
        Ab = self.basis.interpolate(A)
        gA = Ab.grad
        r = np.maximum(self.rq, 1e-12)
        return (-gA[1]).mean(axis=1), (Ab.value / r + gA[0]).mean(axis=1)

    def _linear(self, nu, Brz):
        bilin, lin = self._forms()
        K = asm(bilin, self.basis, nu=self._q(nu))
        f = asm(lin, self.basis, nu=self._q(nu), Brz=self._q(Brz))
        A = solve(*condense(K, f, D=self.D))
        Br_e, Bz_e = self._fields_from_A(A)
        return A, Br_e, Bz_e

    def _slab_H(self, Br_e, Bz_e, Brz):
        """Volume-averaged axial H inside each magnet slab."""
        H = np.zeros(self.n_slab_total)
        for k in range(self.n_slab_total):
            m = self.slab == k
            w = self.vol[m]
            mu = self.regions[self.slab_region[k]].material.mu_rec
            H[k] = float(np.sum(((Bz_e[m] - Brz[m]) / (MU0 * mu)) * w)
                         / np.sum(w))
        return H

    def _permeance(self, nu):
        """H = M @ J, the exact linear map from slab remanence to slab field.

        Magnetostatics is linear in the source for a fixed reluctivity, so the
        self-consistency problem is really a small n-by-n algebraic system, not
        a sequence of field solves.  Building M costs n back-substitutions on a
        single factorisation; after that the nonlinear material law is solved
        in microseconds with an analytic Jacobian, instead of the n+1 full
        field solves per Newton step the finite-difference version needed.
        """
        lu, I, N = self._factorise(nu)
        n = self.n_slab_total
        M = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            Brz = self._expand(e)
            f = self._rhs(nu, Brz)
            A = np.zeros(N)
            A[I] = lu.solve(f[I])
            Br_e, Bz_e = self._fields_from_A(A)
            M[:, j] = self._slab_H(Br_e, Bz_e, Brz)
        return M

    # ---- material law and the small algebraic system ----------------------
    def _target(self, H, scale=1.0):
        """Slab remanence demanded by the material law at field H."""
        t = np.empty(self.n_slab_total)
        for k in range(self.n_slab_total):
            g = self.regions[self.slab_region[k]]
            Hs = min(g.direction * H[k] * scale, 0.0)
            t[k] = g.direction * (g.material.J(Hs) -
                                  (g.material.mu_rec - 1.0) * MU0 * Hs)
        return t

    def _dtarget(self, H, scale=1.0, eps=1.0):
        """d target_k / d H_k, by central difference on the 1-D law.

        This is a scalar curve evaluation, not a field solve, so a tight
        difference is affordable and the knee is resolved properly - which is
        exactly what the old element-wise finite-difference Jacobian could not
        do.
        """
        hp = self._target(H + eps, scale)
        hm = self._target(H - eps, scale)
        return (hp - hm) / (2 * eps)

    def _solve_small(self, M, x0, tol, scale=1.0, max_iter=200):
        """Solve x = target(M x) on the slab unknowns alone.

        No field solves happen here, so it can afford a robust globalised
        Newton, continuation and a bisection safeguard, and still cost nothing.

        The safeguards matter.  Alnico grades have a knee, and a rod whose
        length-to-diameter ratio puts its load line near that knee has a root
        sitting on a nearly vertical piece of the curve: the equivalent
        remanence falls from 0.99 T to 0.19 T over a 7 kA/m change in H.  A
        plain Newton steps straight over it.  That is not an exotic case - it
        is precisely the short, fat rod the optimiser wants to use.
        """
        def resid(x, s):
            return x - self._target(M @ x, s)

        def newton(x, s):
            r = resid(x, s)
            best = (float(np.max(np.abs(r))), x.copy())
            eye = np.eye(self.n_slab_total)
            for _ in range(max_iter):
                if best[0] < tol:
                    break
                Jac = eye - np.diag(self._dtarget(M @ x, s)) @ M
                try:
                    step = np.linalg.solve(Jac, -r)
                except np.linalg.LinAlgError:
                    step = -r
                moved = False
                rn = float(np.max(np.abs(r)))
                for alpha in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01):
                    xt = x + alpha * step
                    rt = resid(xt, s)
                    if float(np.max(np.abs(rt))) < rn:
                        x, r, moved = xt, rt, True
                        break
                if not moved:
                    x = x - 0.2 * r
                    r = resid(x, s)
                if float(np.max(np.abs(r))) < best[0]:
                    best = (float(np.max(np.abs(r))), x.copy())
            return best[1], best[0]

        x, res = newton(x0, scale)
        if res <= tol:
            return x, res

        # continuation: deform the load line in from weakly coupled, so the
        # root is tracked continuously through the knee instead of jumped over
        xc = x0.copy()
        for s in np.linspace(0.05, scale, 40):
            xc, resc = newton(xc, s)
        if resc < res:
            x, res = xc, resc
        if res <= tol:
            return x, res

        # bisection safeguard along the descent from saturation.  The residual
        # changes sign across the knee, so a bracket always exists even when
        # the derivative there is effectively infinite.
        lo, hi = 0.0, 1.0
        f = lambda u: float(np.max(resid(u * x0, scale) *  # noqa: E731
                                   np.sign(x0)))
        if f(lo) * f(hi) < 0:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            xb = 0.5 * (lo + hi) * x0
            xb, resb = newton(xb, scale)
            if resb < res:
                x, res = xb, resb
        return x, res

    # ---- iron loop -------------------------------------------------------
    def _with_iron(self, Brz, iron_iter=250, tol=1e-5, damping=0.2):
        """Solve with the nonlinear steel B-H curve.

        The chord iteration nu = H(|B|)/|B| is only conditionally stable for a
        saturating material: at damping 0.5 or above it oscillates and drives
        |B| in the steel to unphysical values above 3 T.  Damping 0.2 was found
        empirically to converge fastest and to settle at a physical 1.54 T.

        The reluctivity from the previous call is reused as the starting guess.
        This matters a great deal: the outer Newton evaluates this routine once
        per Jacobian column, and consecutive columns differ only by a 1e-4 nudge
        to a single slab, so from a warm start the iron converges in one or two
        passes instead of tens.
        """
        if self._nu_cache is not None:
            nu = self._nu_cache.copy()
        else:
            nu = np.full(self.mesh.t.shape[1], 1.0 / MU0)
            for i, g in enumerate(self.regions):
                m = self.tag == i
                if g.kind == "steel":
                    nu[m] = 1.0 / (MU0 * (g.mu_r if g.mu_r else 1000.0))
                elif g.kind == "magnet":
                    nu[m] = 1.0 / (MU0 * g.material.mu_rec)

        A = Br_e = Bz_e = None
        for _ in range(iron_iter if np.any(self._steel_mask) else 1):
            A, Br_e, Bz_e = self._linear(nu, Brz)
            if not np.any(self._steel_mask):
                break
            Bm = np.hypot(Br_e, Bz_e)[self._steel_mask]
            nu_new = steel_nu(Bm)
            step = float(np.max(np.abs(np.log(nu_new / nu[self._steel_mask]))))
            nu[self._steel_mask] = np.exp(
                (1 - damping) * np.log(nu[self._steel_mask]) +
                damping * np.log(nu_new))
            if step < tol:
                break
        # Return fields that are consistent with the reluctivity being
        # returned.  Without this final pass the loop hands back A/B computed
        # with the reluctivity from BEFORE the last update, so the permeance
        # matrix built from nu describes a slightly different problem than the
        # fields do.  That mismatch put a hard floor of ~2e-3 T on the outer
        # residual and made deeply saturated pot cores look unconvergeable.
        if np.any(self._steel_mask):
            A, Br_e, Bz_e = self._linear(nu, Brz)
        self._nu_cache = nu
        return A, Br_e, Bz_e, nu

    # ---- magnet residual -------------------------------------------------
    def _expand(self, slab_vals):
        """Slab remanence vector -> per-element Brz."""
        Brz = np.zeros(self.mesh.t.shape[1])
        m = self.slab >= 0
        Brz[m] = slab_vals[self.slab[m]]
        return Brz

    def _residual(self, slab_vals, scale=1.0):
        """r_k = Brz_k - target_k(H_k), with H_k from a full field solve."""
        A, Br_e, Bz_e, nu = self._with_iron(self._expand(slab_vals))
        H = np.zeros(self.n_slab_total)
        for k in range(self.n_slab_total):
            m = self.slab == k
            w = self.vol[m]
            g = self.regions[self.slab_region[k]]
            mu = g.material.mu_rec
            Hz = (Bz_e[m] - self._expand(slab_vals)[m]) / (MU0 * mu)
            H[k] = float(np.sum(Hz * w) / np.sum(w))

        target = np.zeros(self.n_slab_total)
        for k in range(self.n_slab_total):
            g = self.regions[self.slab_region[k]]
            mat = g.material
            Hs = min(g.direction * H[k] * scale, 0.0)
            target[k] = g.direction * (mat.J(Hs) -
                                       (mat.mu_rec - 1.0) * MU0 * Hs)
        return slab_vals - target, H, (A, Br_e, Bz_e, nu)

    # ---- Newton ----------------------------------------------------------
    def _newton(self, x, scale, tol, max_iter):
        """Damped Newton with a reused finite-difference Jacobian.

        Each residual evaluation is a full nonlinear field solve, so a fresh
        n+1 evaluation Jacobian every iteration dominates the cost.  The
        residual is close to linear in the slab remanences away from the knee,
        so the Jacobian is recomputed only when a step fails or stops making
        good progress.
        """
        r, H, fields = self._residual(x, scale)
        best = (float(np.max(np.abs(r))), x.copy(), H, fields)
        n = len(x)
        eps = 1e-4
        Jac = None
        for _ in range(max_iter):
            rn = float(np.max(np.abs(r)))
            if rn < tol:
                break
            if Jac is None:
                Jac = np.empty((n, n))
                for j in range(n):
                    xp = x.copy()
                    xp[j] += eps
                    rp, _, _ = self._residual(xp, scale)
                    Jac[:, j] = (rp - r) / eps
            try:
                step = np.linalg.solve(Jac, -r)
            except np.linalg.LinAlgError:
                break
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
                xt = x + alpha * step
                rt, Ht, ft = self._residual(xt, scale)
                if float(np.max(np.abs(rt))) < rn:
                    ratio = float(np.max(np.abs(rt))) / rn
                    x, r, H, fields, improved = xt, rt, Ht, ft, True
                    if ratio > 0.5:        # poor progress: Jacobian is stale
                        Jac = None
                    break
            if not improved:
                if Jac is None:
                    break
                Jac = None                 # retry once with a fresh Jacobian
                continue
            if float(np.max(np.abs(r))) < best[0]:
                best = (float(np.max(np.abs(r))), x.copy(), H, fields)
        return best[1], best[0], best[2], best[3]

    def _picard(self, x, tol, max_iter=60, alpha=0.4):
        """Heavily damped fixed-point fallback.

        Newton uses a finite-difference Jacobian.  Near the knee of the
        demagnetisation curve that Jacobian is taken across a kink in J(H) and
        is simply wrong, so the line search stalls at a large residual.  This
        is exactly the operating point a bare, self-demagnetising rod sits at
        when its neighbour is far away, so it is not a corner case - it is the
        physics the study is about.

        Under-relaxed Picard has no derivative to get wrong.  The self-consistent
        map J -> target(H(J)) has slope < 1 (the demagnetising response opposes
        the change that caused it), so damping with a small alpha turns it into
        a contraction and it converges, slowly but unconditionally.

        Each iteration is a full nonlinear field solve, so the iteration budget
        is deliberately small: this is a rescue, not a primary solver.
        """
        r, H, fields = self._residual(x)
        best = (float(np.max(np.abs(r))), x.copy(), H, fields)
        for _ in range(max_iter):
            if best[0] < tol:
                break
            x = x - alpha * r                      # r = x - target(H(x))
            r, H, fields = self._residual(x)
            rn = float(np.max(np.abs(r)))
            if rn < best[0]:
                best = (rn, x.copy(), H, fields)
            elif rn > 2.0 * best[0]:               # diverging: damp harder
                alpha *= 0.4
                x = best[1].copy()
                r, H, fields = self._residual(x)
                if alpha < 5e-3:
                    break
        return best[1], best[0], best[2], best[3]

    def solve(self, tol=1e-7, max_iter=25, continuation=True):
        """Solve for the magnet slab states.

        Two nested nonlinearities are separated:

        * the MAGNET law is nonlinear in the slab remanences but the field is
          LINEAR in them, so for a frozen reluctivity the whole coupling is the
          n-by-n permeance matrix M and the problem collapses to x = t(Mx) on n
          unknowns.  That is solved exactly, with an analytic Jacobian, at no
          field-solve cost.
        * the IRON law is nonlinear in |B|, so M itself drifts.  That is handled
          by an outer loop that rebuilds M from the updated reluctivity.

        The previous version treated both together and paid n+1 nonlinear field
        solves per Newton step; on a pot-core design that was over 600 seconds.
        This is the same physics, evaluated in the right order.

        ``continuation`` and ``max_iter`` are kept for API compatibility;
        robustness now comes from the structure rather than from ramping.
        """
        if self.n_slab_total == 0:
            A, Br_e, Bz_e, nu = self._with_iron(np.zeros(self.mesh.t.shape[1]))
            return AxisymSolution(
                A=A, Br_elem=Br_e, Bz_elem=Bz_e, nu_elem=nu,
                Brz_elem=np.zeros(self.mesh.t.shape[1]), tag=self.tag,
                slab=self.slab, mesh=self.mesh, basis=self.basis,
                regions=self.regions, slab_H=np.zeros(0), slab_J=np.zeros(0),
                slab_region=self.slab_region, iterations=0, residual=0.0)

        def saturated():
            return np.array([self.regions[self.slab_region[k]].direction *
                             self.regions[self.slab_region[k]].material.J(0.0)
                             for k in range(self.n_slab_total)])

        # Acceptance is RELATIVE to the remanence.  An absolute threshold is
        # meaningless: 1e-5 T on a 1 T magnet is a fully converged answer, but
        # on a 0.05 T operating point it is not.  Using an absolute test
        # rejected perfectly good solutions for short wide magnets, which sit
        # deep below the knee and are exactly the geometries of interest.
        Jsat = max(abs(float(saturated()[0])), 1e-9)
        tol_ok = 1e-4 * Jsat
        tol_accept = 1e-3 * Jsat

        has_iron = bool(np.any(self._steel_mask))
        n_outer = 30 if has_iron else 3
        x = saturated()
        beta = 1.0 if not has_iron else 0.8
        prev_res, stalls = np.inf, 0
        best = (np.inf, x.copy(), None, None)
        for it in range(n_outer):
            # measure the TRUE residual of the current state: field solved
            # with the real nonlinear steel, magnet law evaluated at the field
            # that state actually produces
            A, Br_e, Bz_e, nu = self._with_iron(self._expand(x))
            H = self._slab_H(Br_e, Bz_e, self._expand(x))
            res = float(np.max(np.abs(x - self._target(H))))
            if res < best[0]:
                best = (res, x.copy(), (A, Br_e, Bz_e, nu), H)
            if res < tol_ok:
                break
            # freeze the reluctivity, build the exact linear coupling, solve
            # the material law on the slabs alone
            M = self._permeance(nu)
            x_new, _ = self._solve_small(M, x, tol_ok)
            # Under-relax.  The magnet and the steel fight each other: a
            # stronger magnet saturates the steel, which lowers the permeance,
            # which weakens the magnet.  Taking the full step makes that
            # ping-pong instead of converge whenever the steel is deep in
            # saturation - which is exactly the high-force designs of interest.
            # beta is only reduced on actual divergence: decaying it on a
            # schedule throttled a sequence that was contracting at 0.4 per
            # step and left it stranded two decades short of tolerance.
            x = x + beta * (x_new - x)
            if res > prev_res:
                stalls += 1
                if stalls >= 2:
                    beta = max(0.2, 0.5 * beta)
                    stalls = 0
            else:
                stalls = 0
            prev_res = res

        res, x, fields, H = best
        if res > tol_accept:
            raise RuntimeError(f"magnet solve stalled at residual {res:.1e} T "
                               f"(tolerance {tol_accept:.1e} T)")
        A, Br_e, Bz_e, nu = fields
        mu_rec = np.array([self.regions[self.slab_region[k]].material.mu_rec
                           for k in range(self.n_slab_total)])
        J_slab = x + (mu_rec - 1.0) * MU0 * H      # true polarisation
        return AxisymSolution(
            A=A, Br_elem=Br_e, Bz_elem=Bz_e, nu_elem=nu,
            Brz_elem=self._expand(x), tag=self.tag, slab=self.slab,
            mesh=self.mesh, basis=self.basis, regions=self.regions,
            slab_H=H, slab_J=J_slab, slab_region=self.slab_region,
            iterations=it + 1, residual=res)

    def region_state(self, sol, name):
        """Volume-weighted (J, H) of a named magnet region, from slab values."""
        idx = [i for i, g in enumerate(self.regions) if g.name == name][0]
        ks = np.where(self.slab_region == idx)[0]
        w = np.array([self.vol[self.slab == k].sum() for k in ks])
        return (float(np.sum(sol.slab_J[ks] * w) / w.sum()),
                float(np.sum(sol.slab_H[ks] * w) / w.sum()))


def solve_axisym(regions, r_far, z_far, h, n_slabs=8, tol=1e-7, max_iter=25):
    """Convenience wrapper: build the model and solve it."""
    return AxisymModel(regions, r_far, z_far, h, n_slabs).solve(tol, max_iter)


# --------------------------------------------------------------------------
def sample_B(sol, r, z):
    """Interpolate (B_r, B_z) at arbitrary points."""
    finder = sol.mesh.element_finder()
    el = finder(r, z)
    m, A = sol.mesh, sol.A
    t = m.t[:, el]
    x0, y0 = m.p[0][t[0]], m.p[1][t[0]]
    x1, y1 = m.p[0][t[1]], m.p[1][t[1]]
    x2, y2 = m.p[0][t[2]], m.p[1][t[2]]
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    l0 = ((y1 - y2) * (r - x2) + (x2 - x1) * (z - y2)) / det
    l1 = ((y2 - y0) * (r - x2) + (x0 - x2) * (z - y2)) / det
    Aval = A[t[0]] * l0 + A[t[1]] * l1 + A[t[2]] * (1 - l0 - l1)
    dNdx = np.vstack([(y1 - y2), (y2 - y0), (y0 - y1)]) / det
    dNdy = np.vstack([(x2 - x1), (x0 - x2), (x1 - x0)]) / det
    dAdx = (A[t] * dNdx).sum(axis=0)
    dAdy = (A[t] * dNdy).sum(axis=0)
    return -dAdy, Aval / np.maximum(r, 1e-12) + dAdx


def axial_force(sol, z_plane, r_max, n=4000):
    """Axial force across the plane z = z_plane, by Maxwell stress.

        F_z = int (B_z^2 - B_r^2) / (2 mu0) * 2 pi r dr
    """
    r = np.linspace(1e-9, r_max, n)
    Br, Bz = sample_B(sol, r, np.full_like(r, z_plane))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz((Bz**2 - Br**2) / (2 * MU0) * 2 * np.pi * r, r))

