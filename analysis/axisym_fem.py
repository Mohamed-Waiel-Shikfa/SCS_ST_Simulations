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
    def _linear(self, nu, Brz):
        q = lambda v: np.tile(v[:, None], (1, self.nq))  # noqa: E731

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

        K = asm(bilin, self.basis, nu=q(nu))
        f = asm(lin, self.basis, nu=q(nu), Brz=q(Brz))
        A = solve(*condense(K, f, D=self.D))

        Ab = self.basis.interpolate(A)
        gA = Ab.grad
        r = np.maximum(self.rq, 1e-12)
        Br_e = (-gA[1]).mean(axis=1)
        Bz_e = (Ab.value / r + gA[0]).mean(axis=1)
        return A, Br_e, Bz_e

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

    def solve(self, tol=1e-7, max_iter=25, continuation=True):
        """Solve for the magnet slab states.

        ``continuation`` walks in from a weakly-coupled problem when the direct
        Newton stalls.  It is reliable but costs several extra Newton solves, so
        screening runs can switch it off and treat a stall as a failed design
        rather than paying for it.
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

        x, res, H, fields = self._newton(saturated(), 1.0, tol, max_iter)
        if res > 1e-5 and continuation:
            # continuation: ramp the demagnetising response from weak to full
            x = saturated()
            for lam in np.linspace(0.2, 1.0, 5):
                x, res, H, fields = self._newton(x, lam, tol, max_iter)
        if res > 1e-5:
            raise RuntimeError(f"magnet solve stalled at residual {res:.1e} T")

        A, Br_e, Bz_e, nu = fields
        mu_rec = np.array([self.regions[self.slab_region[k]].material.mu_rec
                           for k in range(self.n_slab_total)])
        J_slab = x + (mu_rec - 1.0) * MU0 * H      # true polarisation
        return AxisymSolution(
            A=A, Br_elem=Br_e, Bz_elem=Bz_e, nu_elem=nu,
            Brz_elem=self._expand(x), tag=self.tag, slab=self.slab,
            mesh=self.mesh, basis=self.basis, regions=self.regions,
            slab_H=H, slab_J=J_slab, slab_region=self.slab_region,
            iterations=0, residual=res)

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
