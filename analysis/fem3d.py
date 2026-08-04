r"""Three-dimensional magnetostatics for two interacting modules.

Why this exists
---------------
The axisymmetric FEM answers one question very well: two EPMs nose to nose on
a shared axis, ``||``.  That is the latching case, and it is worth keeping -
it is cheap, it is verified against measured data, and it resolves the
nonlinear magnet interior properly.

It cannot answer the other half of the problem.  Modules in a lattice also
meet at an angle: during a pivot the driving pair rotates away from coaxial
and a second pair swings towards it, so the geometry is ``/\`` and there is no
axis of revolution anywhere in it.  An axisymmetric solver has nothing to say
about that, and until now the pipeline simply assumed the angled pairs did not
exist.

Method
------
A magnetostatic method of moments on the magnetised bodies alone.

Each magnet and each steel piece is diced into cuboid cells carrying a uniform
magnetisation ``M``.  A uniformly magnetised cuboid is equivalent to a pair of
uniformly charged rectangles on its end faces, whose field is available in
closed form, so the field at any point follows from a sum over cells with no
mesh in the air at all.  Collocating at cell centres gives a dense linear map
``H = N M + H_ext``; closing it with the material law at every cell,

    magnets:  M = M_r(H . e) e + (mu_rec - 1) H      (uniaxial, easy axis e)
    steel:    M = (mu_r(|H|) - 1) H                  (isotropic, saturating)

leaves a fixed-point problem in the cell magnetisations, solved by damped
iteration with Anderson acceleration.

Why not a tetrahedral FEM.  A 3-D FEM has to mesh the air box as well as the
bodies, and the air is where most of the elements go: resolving two 4 mm
magnets 0.1 mm apart inside a 100 mm box costs hundreds of thousands of
elements before any physics happens.  The integral method puts unknowns only
where the material is - a few hundred - and gets the far field exactly right
by construction instead of by boundary truncation.  The price is a dense
matrix, which at this size is nothing.

What is approximate
-------------------
* Cell-centre collocation is a midpoint rule.  It converges as the cells
  shrink, and ``verify_fem3d.py`` measures the remaining error against the
  axisymmetric FEM on the coaxial case, which both solvers can do.
* Cylindrical magnets are diced into cuboid cells that tile the circular
  cross-section, so the pole outline is stepped.  The area is corrected, and
  the residual error is included in the same measurement.
* Only the faces in play are meshed.  Faces pointing away from the neighbour
  contribute a few parts in a thousand and are dropped by default; the cutoff
  is a parameter and the convergence study varies it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

from axisym_fem import STEEL_1018_B, STEEL_1018_H  # noqa: E402

MU0 = 4.0e-7 * np.pi


# ==========================================================================
# Field of a uniformly charged rectangle
# ==========================================================================
def _sheet_field(corners, z0, obs):
    """H at ``obs`` from a unit-density charged rectangle in a local frame.

    The rectangle spans ``corners = (x1, x2, y1, y2)`` in the plane z = z0.
    Charge density is 1 A/m; scale the result by the actual sigma.

    Closed form of

        H = -grad (1/4pi) int sigma / |r - r'| dA'

    over a rectangle.  Verified against 2-D Gauss-Legendre quadrature of the
    integral itself in the self-test at the bottom of this file.
    """
    x1, x2, y1, y2 = corners
    x, y, z = obs[..., 0], obs[..., 1], obs[..., 2] - z0
    # The Hz term is arctan(uv / zR) on the PRINCIPAL branch.  Using atan2
    # instead adds a jump of pi wherever uv changes sign, which cancels in the
    # corner sum only when the observation point lies outside the rectangle's
    # shadow.  Inside it - which is exactly the self-field of a cell - it does
    # not cancel, and a cube then comes out with a demagnetising factor of
    # -2/3 instead of 1/3.
    zs = np.where(np.abs(z) < 1e-18, 1e-18, z)

    Hx = np.zeros_like(x)
    Hy = np.zeros_like(x)
    Hz = np.zeros_like(x)
    for i, xi in enumerate((x1, x2)):
        for j, yj in enumerate((y1, y2)):
            s = 1.0 if (i == j) else -1.0
            u = x - xi
            v = y - yj
            R = np.sqrt(u * u + v * v + z * z)
            R = np.maximum(R, 1e-14)
            Hx -= s * np.log(np.maximum(v + R, 1e-300))
            Hy -= s * np.log(np.maximum(u + R, 1e-300))
            Hz += s * np.arctan(u * v / (zs * R))
    return np.stack([Hx, Hy, Hz], axis=-1) / (4.0 * np.pi)


def cuboid_field(half, centre, axes, M, obs):
    """H at ``obs`` from one uniformly magnetised cuboid cell.

    ``half`` are the half-extents along the cell's own ``axes`` (3x3, rows are
    unit vectors), ``M`` is the magnetisation vector in world coordinates.
    Superposes the three charge-sheet pairs, one per local axis.
    """
    axes = np.asarray(axes, dtype=float)
    rel = np.asarray(obs, dtype=float) - np.asarray(centre, dtype=float)
    loc = rel @ axes.T                       # into the cell frame
    Ml = np.asarray(M, dtype=float) @ axes.T

    Hl = np.zeros_like(loc)
    perm = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for k, (a, b, c) in enumerate(perm):
        if abs(Ml[k]) < 1e-300:
            continue
        # sheets normal to local axis a, spanning b and c
        p = loc[..., [b, c, a]]
        corners = (-half[b], half[b], -half[c], half[c])
        h = (_sheet_field(corners, +half[a], p) -
             _sheet_field(corners, -half[a], p)) * Ml[k]
        Hl[..., b] += h[..., 0]
        Hl[..., c] += h[..., 1]
        Hl[..., a] += h[..., 2]
    return Hl @ axes


# ==========================================================================
# Geometry
# ==========================================================================
@dataclass
class Cell:
    centre: np.ndarray
    half: np.ndarray            # half extents along ``axes``
    axes: np.ndarray            # 3x3, rows are the local unit axes
    kind: str                   # "magnet" | "steel"
    easy: np.ndarray = None     # easy axis for magnets, world frame
    material: object = None
    body: int = 0               # which module (0 or 1)
    face: int = 0               # which EPM on that module
    sign: float = 1.0           # +1 / -1 commanded polarity
    volume: float = 0.0

    def __post_init__(self):
        self.volume = 8.0 * float(np.prod(self.half))


def _disc_tiles(radius, n_across):
    """Square tiling of a disc (legacy; kept for the kernel self-test).

    Naively scaling the tiles up to recover the circle's area makes them
    overlap, so the lattice PITCH is chosen instead so that the tiles kept by
    the centre-inside test have exactly the true area between them.  Tiles of
    side equal to the pitch then tile without gaps or overlaps.

    This is still not good enough for an assembly: a square tiling of a disc
    necessarily reaches past the disc's own radius - by 25 % at the corners
    here - so the outer tiles run into the steel annulus that starts only
    0.5 mm away.  ``_disc_polar`` below is what the EPM builder uses.
    """
    n = max(int(n_across), 1)
    area_true = np.pi * radius ** 2
    if n == 1:
        s = np.sqrt(area_true)
        return np.zeros((1, 2)), s / 2.0

    p0 = 2.0 * radius / n
    g0 = (np.arange(n) - (n - 1) / 2.0) * p0
    X, Y = np.meshgrid(g0, g0, indexing="ij")
    keep = (X ** 2 + Y ** 2) <= radius ** 2
    k = int(keep.sum())
    if k == 0:
        s = np.sqrt(area_true)
        return np.zeros((1, 2)), s / 2.0

    p = np.sqrt(area_true / k)
    g = (np.arange(n) - (n - 1) / 2.0) * p
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X[keep], Y[keep]], axis=-1)
    return pts, p / 2.0


def _disc_polar(radius, n_rings, max_sect=10):
    """Dice a disc into area-exact cells that stay inside its own radius.

    Equal-area rings, each cut into sectors of roughly square aspect.  Returns
    a list of ``(r_mid, angle, half_radial, half_tangential)``; a ``r_mid`` of
    zero means the central square core, whose side is set by equal area.

    A square tiling cannot do this - its corners always stick out past the
    radius - and here that is fatal rather than cosmetic: with a 0.5 mm
    clearance the overhanging magnet tiles interpenetrated the steel annulus,
    50 to 96 cell pairs of it, which is why the pot-core repelling force came
    out five times too large and refused to converge.
    """
    n_r = max(int(n_rings), 1)
    edges = radius * np.sqrt(np.arange(n_r + 1) / n_r)
    out = [(0.0, 0.0, np.sqrt(np.pi) * edges[1] / 2.0,
            np.sqrt(np.pi) * edges[1] / 2.0)]
    for k in range(1, n_r):
        ra, rb = edges[k], edges[k + 1]
        r_mid = 0.5 * (ra + rb)
        dr = rb - ra
        ns = int(np.clip(round(2.0 * np.pi * r_mid / max(dr, 1e-12)), 4,
                         max_sect))
        w = 2.0 * np.pi * r_mid / ns
        for i in range(ns):
            out.append((r_mid, 2.0 * np.pi * (i + 0.5) / ns, dr / 2.0,
                        w / 2.0))
    return out


def _polar_reach(cells_spec):
    """Furthest radius any polar cell reaches, for the clearance check."""
    m = 0.0
    for r_mid, _a, hr, ht in cells_spec:
        m = max(m, float(np.hypot(r_mid + hr, ht)) if r_mid > 0
                else float(np.hypot(hr, ht)))
    return m


def _frame(normal):
    """Right-handed frame whose third axis is ``normal``."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    t = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t, n)) > 0.9:
        t = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.stack([u, v, n])


def _axial_slabs(length, n, grade=2.2):
    """Slab boundaries along the magnet, refined towards the pole face.

    The polarisation is far from uniform: in a repelling pair it collapses to
    a third of its bulk value in the first half-millimetre behind the pole and
    recovers over the next two.  Uniform slabs put most of their resolution
    where nothing is happening, so the repelling force converges slowly - it
    was still 17 % high with six uniform slabs.  A geometric grading towards
    the pole fixes that at no extra cost.
    """
    n = max(int(n), 1)
    if n == 1:
        return np.array([0.0, length])
    w = grade ** np.arange(n)
    w = w / w.sum() * length
    return np.concatenate([[0.0], np.cumsum(w)])


def epm_cells(centre, normal, d_mag, l_mag, material, sign=1.0,
              t_steel=0.0, r_clear=0.0, n_across=3, n_axial=3,
              body=0, face=0, with_steel=True, n_sect=8, n_rad=1,
              n_ax_steel=2):
    """Dice one EPM assembly into cells.

    The pole face sits at ``centre``; the assembly extends inward along
    ``-normal``.  ``sign`` is the commanded polarity: +1 presents a north pole
    outward, -1 a south.  ``r_clear`` is the radial space between the magnet
    and the steel, which in a built module is occupied by the winding.
    """
    axes = _frame(normal)
    n_hat = axes[2]
    r_m = d_mag / 2.0
    spec = _disc_polar(r_m, n_across)
    C = np.asarray(centre, dtype=float)

    def place(r_mid, ang, hr, ht, depth, hz, kind, **extra):
        if r_mid <= 0:
            rad, tan = axes[0], axes[1]
            off = np.zeros(3)
        else:
            rad = axes[0] * np.cos(ang) + axes[1] * np.sin(ang)
            tan = -axes[0] * np.sin(ang) + axes[1] * np.cos(ang)
            off = rad * r_mid
        return Cell(centre=C - n_hat * depth + off,
                    half=np.array([hr, ht, hz]),
                    axes=np.stack([rad, tan, n_hat]), kind=kind,
                    body=body, face=face, **extra)

    cells = []
    edges = _axial_slabs(l_mag, n_axial)
    for k in range(len(edges) - 1):
        z0, z1 = edges[k], edges[k + 1]
        depth = 0.5 * (z0 + z1)
        hz = 0.5 * (z1 - z0)
        for (r_mid, ang, hr, ht) in spec:
            cells.append(place(r_mid, ang, hr, ht, depth, hz, "magnet",
                               easy=n_hat * sign, material=material,
                               sign=sign))

    if with_steel and t_steel > 0:
        # the annulus must clear whatever the magnet dicing actually reaches,
        # not merely the nominal magnet radius
        r_in = max(r_m + r_clear, _polar_reach(spec) * (1.0 + 1e-6))
        r_out = r_m + r_clear + t_steel
        if r_out <= r_in:
            r_out = r_in + max(t_steel, 1e-5)
        nr = max(int(n_rad), 1)
        ns = max(int(n_sect), 4)
        nz = max(int(n_ax_steel), 1)
        dr = (r_out - r_in) / nr
        dz = l_mag / nz
        for ir in range(nr):
            ra = r_in + ir * dr
            r_mid = ra + dr / 2.0
            w = 2.0 * np.pi * r_mid / ns
            for iz in range(nz):
                depth = (iz + 0.5) * dz
                for i in range(ns):
                    cells.append(place(r_mid, 2.0 * np.pi * (i + 0.5) / ns,
                                       dr / 2.0, w / 2.0, depth, dz / 2.0,
                                       "steel"))
        # back plate, diced the same way so its outline is round too
        bspec = _disc_polar(r_out, max(n_across, 2))
        for (r_mid, ang, hr, ht) in bspec:
            cells.append(place(r_mid, ang, hr, ht,
                               l_mag + t_steel / 2.0, t_steel / 2.0, "steel"))
    return cells


# ==========================================================================
# Solver
# ==========================================================================
@dataclass
class Solution3D:
    cells: list
    M: np.ndarray               # (n, 3) magnetisation
    H: np.ndarray               # (n, 3) field at cell centres
    iters: int
    residual: float
    converged: bool
    meta: dict = field(default_factory=dict)

    def body_mask(self, b):
        return np.array([c.body == b for c in self.cells])

    def magnet_state(self, body=None, face=None):
        """Volume-averaged J and axial H over the selected magnet cells."""
        sel = [i for i, c in enumerate(self.cells)
               if c.kind == "magnet"
               and (body is None or c.body == body)
               and (face is None or c.face == face)]
        if not sel:
            return 0.0, 0.0
        w = np.array([self.cells[i].volume for i in sel])
        e = np.array([self.cells[i].easy for i in sel])
        J = MU0 * np.einsum("ij,ij->i", self.M[sel], e)
        H = np.einsum("ij,ij->i", self.H[sel], e)
        return float(np.average(J, weights=w)), float(np.average(H, weights=w))


def _steel_chi(Hmag):
    """Susceptibility of 1018 steel at field magnitude |H|."""
    H = np.maximum(np.asarray(Hmag, dtype=float), 1e-9)
    B = np.interp(H, STEEL_1018_H, STEEL_1018_B)
    tail = H > STEEL_1018_H[-1]
    if np.any(tail):
        B = np.where(tail, STEEL_1018_B[-1] + (H - STEEL_1018_H[-1]) * MU0, B)
    return np.maximum(B / (MU0 * H) - 1.0, 0.0)


def coupling_matrix(cells, block=64):
    """Dense (3n, 3n) map from cell magnetisations to cell-centre fields."""
    n = len(cells)
    N = np.zeros((3 * n, 3 * n))
    obs = np.array([c.centre for c in cells])
    for j in range(n):
        cj = cells[j]
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            H = cuboid_field(cj.half, cj.centre, cj.axes, e, obs)
            N[0::3, 3 * j + k] = H[:, 0]
            N[1::3, 3 * j + k] = H[:, 1]
            N[2::3, 3 * j + k] = H[:, 2]
    return N


def solve3d(cells, h_ext=None, tol=1e-6, max_iter=60, relax=1.0, N=None,
            verbose=False, continuation=True):
    """Self-consistent magnetisation of every cell, by damped Newton.

    Why not simple iteration.  Substituting the material law repeatedly works
    for magnets, whose effective susceptibility is only mu_rec - 1 (one to
    three), but it diverges outright once soft iron is present: 1018 steel has
    a susceptibility of about 2000 near 1 T, so the map has a gain far above
    one and the iterates grow without bound.  Damping it enough to be stable
    makes it far too slow to be useful.  With a pot core the plain iteration
    was still at a residual of 1.1 after two thousand passes - not converged at
    all, while reporting numbers that looked plausible.

    Instead the material law is linearised at every cell each pass,

        M = A(H) H + b(H)

    with A the local slope (the secant susceptibility for iron, the
    demagnetisation-curve slope along the easy axis and mu_rec - 1 across it
    for a magnet), and the whole coupled system

        (I - A N) M = A H_ext + b

    is solved directly.  That is a Newton step on the same fixed point and it
    converges in a handful of passes however permeable the iron is.

    Continuation.  Newton alone still fails on the REPELLING state, and for a
    physical reason rather than a numerical one: two like poles a tenth of a
    millimetre apart drive each other past the knee, where the curve is nearly
    vertical.  The iteration then flips between "fully magnetised, so the
    neighbour's field is huge, so demagnetise" and "demagnetised, so there is
    no field, so remagnetise", and sits at a residual of exactly 1.  Ramping
    the magnet strength from weak to full and warm-starting each step walks
    around the knee instead of jumping over it.  This is the same device the
    axisymmetric FEM uses, for the same reason.
    """
    n = len(cells)
    if n == 0:
        return Solution3D([], np.zeros((0, 3)), np.zeros((0, 3)), 0, 0.0, True)
    if N is None:
        N = coupling_matrix(cells)
    H0 = np.zeros((n, 3)) if h_ext is None else np.asarray(h_ext, dtype=float)
    if H0.ndim == 1:
        H0 = np.tile(H0, (n, 1))

    is_mag = np.array([c.kind == "magnet" for c in cells])
    easy = np.array([c.easy if c.easy is not None else np.zeros(3)
                     for c in cells])
    mu_rec = np.array([c.material.mu_rec if c.material is not None else 1.0
                       for c in cells])
    mag_idx = np.where(is_mag)[0]
    steel_idx = np.where(~is_mag)[0]

    def law(H, sf):
        """Material law: magnetisation demanded at field H.

        Magnets sit on the MAJOR demagnetisation curve with no recoil history
        carried across iterations.  Tracking the deepest reverse field each
        cell had seen - which sounds more physical - is a trap: the first pass
        overshoots wildly from the Br starting guess, the history latches onto
        that overshoot, and the magnet is permanently demagnetised by a field
        that was never a physical operating point.  It converged to J = 0.45 T
        where the validated one-dimensional solver gives 0.62 T.

        This matches the axisymmetric FEM's convention exactly, which is what
        makes the two comparable.  Irreversible loss along a real duty cycle is
        applied afterwards, outside the field solve, where the actual sequence
        of operating points is known.
        """
        out = np.zeros((n, 3))
        for i in mag_idx:
            e = easy[i]
            ha = float(np.dot(H[i], e))
            J = sf * float(cells[i].material.J(np.array([min(ha, 0.0)]))[0])
            out[i] = e * (J / MU0) + (mu_rec[i] - 1.0) * (H[i] - ha * e)
        if len(steel_idx):
            hm = np.linalg.norm(H[steel_idx], axis=1)
            out[steel_idx] = _steel_chi(hm)[:, None] * H[steel_idx]
        return out

    def linearise(H, sf):
        """Block-diagonal A and offset b of the local law at field H."""
        A = np.zeros((3 * n, 3 * n))
        b = np.zeros((n, 3))
        for i in mag_idx:
            sl = slice(3 * i, 3 * i + 3)
            e = easy[i]
            ha = float(np.dot(H[i], e))
            hs = min(ha, 0.0)
            mt = cells[i].material
            J = sf * float(mt.J(np.array([hs]))[0])
            d = max(abs(hs) * 1e-3, 10.0)
            hp, hm_ = min(hs + d, 0.0), hs - d
            slope = sf * (float(mt.J(np.array([hp]))[0]) -
                          float(mt.J(np.array([hm_]))[0])) / (hp - hm_) / MU0
            slope = max(slope, 0.0)
            P = np.outer(e, e)
            A[sl, sl] = slope * P + (mu_rec[i] - 1.0) * (np.eye(3) - P)
            b[i] = e * (J / MU0 - slope * ha)
        for i in steel_idx:
            sl = slice(3 * i, 3 * i + 3)
            A[sl, sl] = float(_steel_chi(np.linalg.norm(H[i]))) * np.eye(3)
        return A, b

    def newton(M, sf, iters, om):
        res = np.inf
        used = 0
        for used in range(1, iters + 1):
            H = H0 + (N @ M.reshape(-1)).reshape(n, 3)
            Mtgt = law(H, sf)
            scale = max(np.abs(M).max(), np.abs(Mtgt).max(), 1.0)
            res = float(np.abs(Mtgt - M).max() / scale)
            if res < tol:
                return Mtgt, res, used
            A, b = linearise(H, sf)
            try:
                Mnew = np.linalg.solve(np.eye(3 * n) - A @ N,
                                       (A @ H0.reshape(-1)) +
                                       b.reshape(-1)).reshape(n, 3)
            except np.linalg.LinAlgError:
                Mnew = Mtgt
            if not np.all(np.isfinite(Mnew)):
                Mnew = Mtgt
            M = M + om * (Mnew - M)
        return M, res, used

    steps = (0.2, 0.45, 0.7, 0.9, 1.0) if continuation else (1.0,)
    M = np.zeros((n, 3))
    for i in mag_idx:
        M[i] = easy[i] * (cells[i].material.Br / MU0) * steps[0]

    total = 0
    res = np.inf
    for si, sf in enumerate(steps):
        if si:
            M = M * (sf / steps[si - 1])
        M, res, used = newton(M, sf, max_iter, relax)
        total += used
    if res >= tol:
        # last resort: heavier damping on the final, hardest step
        M, res2, used = newton(M, 1.0, max_iter * 2, 0.35)
        total += used
        res = min(res, res2)

    H = H0 + (N @ M.reshape(-1)).reshape(n, 3)
    if verbose:
        print(f"    solve3d: {total} passes, residual {res:.2e}, {n} cells")
    return Solution3D(cells=cells, M=M, H=H, iters=total, residual=res,
                      converged=res < max(tol * 20, 1e-4))


# ==========================================================================
# Forces
# ==========================================================================
def _face_patches(cell, M, k, s, n_sub):
    """Charge patches on one face of a cell.

    ``k`` selects the local axis the face is normal to, ``s`` which of the two
    faces.  Split out per face so the quadrature can be refined only where it
    needs to be.
    """
    A = cell.axes
    h = cell.half
    sigma = float(np.dot(M, A[k]))
    if abs(sigma) < 1e-12:
        return np.zeros((0, 3)), np.zeros(0)
    a, b = [i for i in range(3) if i != k]
    n = max(int(n_sub), 1)
    g = (np.arange(n) + 0.5) / n * 2 - 1
    area = (2 * h[a] / n) * (2 * h[b] / n)
    ua, ub = np.meshgrid(g, g, indexing="ij")
    base = cell.centre + A[k] * s * h[k]
    pts = (base[None, None, :] + A[a][None, None, :] * (ua * h[a])[..., None]
           + A[b][None, None, :] * (ub * h[b])[..., None]).reshape(-1, 3)
    qs = np.full(len(pts), sigma * s * area)
    return pts, qs


def _charges(cell, M, n_sub=2):
    """All six faces of a cell at a uniform sub-sampling (used by tests)."""
    P, Q = [], []
    for k in range(3):
        for s in (+1, -1):
            p, q = _face_patches(cell, M, k, s, n_sub)
            if len(p):
                P.append(p)
                Q.append(q)
    if not P:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(P), np.concatenate(Q)


def force_on_body(sol, body, source_body=None, n_sub=None, max_sub=40,
                  target=0.6):
    """Force and torque on ``body`` from ``source_body`` (default: all others).

    Uses the magnetic-charge form, F = mu0 sum(q H_source).  Only the field of
    the OTHER body is evaluated at this body's charges, so the self-force
    cancels exactly by construction rather than numerically.

    Quadrature is refined PER FACE, from that face's own distance to the
    nearest source cell.  This is not a detail.  Two mated pole faces sit
    0.1 mm apart while the cells are a millimetre or two across, so a patch
    that is not far smaller than the gap samples the steepest part of the field
    at forty times its own scale.  When both bodies are coaxial the resulting
    error is symmetric and cancels, which is why the aligned force converged
    beautifully while the same model at 15 degrees scattered by a factor of
    eight between discretisations - the two square tilings no longer line up,
    so nothing cancels.  Refining only the faces that are actually close keeps
    the cost bounded: the far faces stay at 2 x 2.
    """
    tgt = [i for i, c in enumerate(sol.cells) if c.body == body]
    src = [i for i, c in enumerate(sol.cells)
           if c.body != body and (source_body is None or
                                  c.body == source_body)]
    if not tgt or not src:
        return np.zeros(3), np.zeros(3)

    src_c = np.array([sol.cells[j].centre for j in src])
    src_r = np.array([np.linalg.norm(sol.cells[j].half) for j in src])

    P, Q = [], []
    for i in tgt:
        c = sol.cells[i]
        for k in range(3):
            for s in (+1, -1):
                fc = c.centre + c.axes[k] * s * c.half[k]
                if n_sub is None:
                    d = float(np.min(np.linalg.norm(src_c - fc, axis=1) -
                                     src_r))
                    d = max(d, 1e-6)
                    a, b = [t for t in range(3) if t != k]
                    span = 2.0 * max(c.half[a], c.half[b])
                    ns = int(np.clip(np.ceil(span / (target * d)), 2,
                                     max_sub))
                else:
                    ns = n_sub
                p, q = _face_patches(c, sol.M[i], k, s, ns)
                if len(p):
                    P.append(p)
                    Q.append(q)
    if not P:
        return np.zeros(3), np.zeros(3)
    P = np.concatenate(P)
    Q = np.concatenate(Q)

    H = np.zeros_like(P)
    for j in src:
        c = sol.cells[j]
        H += cuboid_field(c.half, c.centre, c.axes, sol.M[j], P)

    f = MU0 * Q[:, None] * H
    F = f.sum(axis=0)
    cen = np.mean([sol.cells[i].centre for i in tgt], axis=0)
    T = np.cross(P - cen, f).sum(axis=0)
    return F, T


def sample_field(sol, points, include_ext=None):
    """B at arbitrary points, for the volumetric viewer.

    Returns B in tesla.  Inside a magnetised cell this is mu0 (H + M); the
    cell membership test is cheap because the cells are axis-aligned boxes in
    their own frames.
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    H = np.zeros_like(P)
    for j, c in enumerate(sol.cells):
        H += cuboid_field(c.half, c.centre, c.axes, sol.M[j], P)
    if include_ext is not None:
        H = H + np.asarray(include_ext, dtype=float)
    M = np.zeros_like(P)
    for j, c in enumerate(sol.cells):
        loc = np.abs((P - c.centre) @ c.axes.T)
        inside = np.all(loc <= c.half + 1e-12, axis=1)
        if np.any(inside):
            M[inside] += sol.M[j]
    return MU0 * (H + M)


# ==========================================================================
# Scenes
# ==========================================================================
def roll_pose(n_gon, r_face, theta):
    """Pose of module B part-way through a roll onto module A.

    B starts beside A on the floor and tips forward over its leading bottom
    edge through the exterior angle 2 pi / n.  Returns (centre, R) with the
    rotation matrix R mapping B's body frame into the world.

    Module A is fixed with its centre at the origin and its mating pole face
    at +x.  This is the single source of truth for the roll kinematics: the
    Stage 2 gap model and the 3-D scene both read it, so they cannot drift
    apart.
    """
    a = 2.0 * r_face * np.tan(np.pi / n_gon)
    C0 = np.array([2.0 * r_face, 0.0, 0.0])       # B's centre at theta = 0
    E = C0 + np.array([a / 2.0, 0.0, -r_face])    # leading bottom edge
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0.0, s],
                  [0.0, 1.0, 0.0],
                  [-s, 0.0, c]])
    centre = E + R @ (C0 - E)
    return centre, R


def epm_pair_cells(d_mag, l_mag, material, gap, t_steel=0.0, r_clear=0.0,
                   states=(+1, -1), angle=0.0, n_across=3, n_axial=3,
                   with_steel=True, r_pivot=None, **kw):
    r"""Two EPMs facing each other, optionally at a relative angle.

    ``angle`` = 0 is the coaxial ``||`` case the axisymmetric FEM also solves,
    which is what makes verification possible.  Any other angle is the ``/\``
    case, which it cannot.

    The tilt is taken about the RIM of the pole faces, not about their
    centres.  That is both what the physics does - a module pivots over the
    edge it shares with its neighbour - and what keeps the geometry legal:
    rotating about the face centres holds the centres ``gap`` apart while the
    rims swing towards each other, so beyond asin(gap / r_pole), about three
    degrees here, the two magnets INTERPENETRATE.  Cells that overlap produce
    arbitrary numbers, which is what an earlier version of this function did
    at every angle it was asked for.

    ``states`` are the commanded polarities, and the convention is the one a
    controller would use: **+1 means the face presents a north pole outward**.
    Two faces showing the same sign present like poles and REPEL; opposite
    signs ATTRACT.  Note this is the opposite of the magnetisation-direction
    convention used by ``CoaxialRodPair``.
    """
    cells = epm_cells(np.zeros(3), np.array([1.0, 0.0, 0.0]), d_mag, l_mag,
                      material, sign=states[0], t_steel=t_steel,
                      r_clear=r_clear, n_across=n_across, n_axial=n_axial,
                      body=0, face=0, with_steel=with_steel, **kw)

    # The tilt is about the RIM of the assembly, with a small margin.  The
    # cells' stepped outline reaches a couple of per cent past the nominal
    # outer radius, and without the margin the outermost back-plate cells dip
    # below the pivot and swing into the neighbour.
    rp = r_pivot if r_pivot is not None else \
        1.05 * (d_mag / 2.0 + r_clear + t_steel)
    piv = np.array([gap, 0.0, -rp])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    p0 = np.array([gap, 0.0, 0.0])
    pB = piv + R @ (p0 - piv)
    nB = R @ np.array([-1.0, 0.0, 0.0])

    cells += epm_cells(pB, nB, d_mag, l_mag, material, sign=states[1],
                       t_steel=t_steel, r_clear=r_clear, n_across=n_across,
                       n_axial=n_axial, body=1, face=0,
                       with_steel=with_steel, **kw)
    return cells


def check_overlap(cells, tol=1e-9, same_body=True):
    """Cells that interpenetrate.

    A cheap safety net.  Overlapping cells double-count magnetised material
    and make the force meaningless, and the failure is silent - the solve
    still converges, it just converges to nonsense.  Two distinct bugs were
    found by this test: an angled pair rotated about the face centres rather
    than the shared rim, so the two magnets ran through each other beyond
    three degrees; and square magnet tiles whose corners reached past the
    magnet radius into the steel annulus.

    ``same_body`` also checks cells belonging to the same assembly, which is
    where the second of those hid.
    """
    bad = []
    n = len(cells)
    for i in range(n):
        ci = cells[i]
        ri = float(np.linalg.norm(ci.half))
        for j in range(i + 1, n):
            cj = cells[j]
            if ci.body == cj.body and not same_body:
                continue
            if ci.body == cj.body and ci.kind == cj.kind and \
                    ci.face == cj.face:
                continue          # deliberate subdivision of one part
            d = cj.centre - ci.centre
            if np.linalg.norm(d) > ri + float(np.linalg.norm(cj.half)):
                continue
            sep = False
            for A, other in ((ci, cj), (cj, ci)):
                loc = np.abs((other.centre - A.centre) @ A.axes.T)
                reach = A.half + np.abs(other.axes @ A.axes.T) @ other.half
                # ">= reach - tol" so cells that merely TOUCH - which adjacent
                # parts are supposed to do - are not reported as overlapping
                if np.any(loc > reach - tol):
                    sep = True
                    break
            if not sep:
                bad.append((i, j))
    return bad


ATTRACT = (+1, -1)
REPEL = (+1, +1)


def module_pair_cells(normals, r_face, d_mag, l_mag, material, states_a,
                      states_b, pose, t_steel=0.0, r_clear=0.0,
                      n_across=2, n_axial=2, cos_cut=0.0, with_steel=True):
    """Every EPM in play on two modules at an arbitrary relative pose.

    ``states_a`` / ``states_b`` map face index to +1, -1 or 0 (0 means the
    face is not modelled at all).  ``pose`` is (centre, R) for module B, with
    module A at the origin unrotated.

    Faces pointing away from the other module are dropped: at these
    separations their contribution to the pair force is a few parts in a
    thousand, and including all 36 would quadruple the matrix for nothing.
    ``cos_cut`` is the cutoff and is swept in the convergence study.
    """
    cB, R = pose
    cB = np.asarray(cB, dtype=float)
    toB = cB / max(np.linalg.norm(cB), 1e-12)

    cells = []
    for k, nrm in enumerate(normals):
        st = states_a.get(k, 0)
        if st == 0 or float(np.dot(nrm, toB)) < cos_cut:
            continue
        cells += epm_cells(nrm * r_face, nrm, d_mag, l_mag, material,
                           sign=st, t_steel=t_steel, r_clear=r_clear,
                           n_across=n_across, n_axial=n_axial, body=0,
                           face=k, with_steel=with_steel)
    for k, nrm in enumerate(normals):
        st = states_b.get(k, 0)
        nw = R @ nrm
        if st == 0 or float(np.dot(nw, -toB)) < cos_cut:
            continue
        cells += epm_cells(cB + nw * r_face, nw, d_mag, l_mag, material,
                           sign=st, t_steel=t_steel, r_clear=r_clear,
                           n_across=n_across, n_axial=n_axial, body=1,
                           face=k, with_steel=with_steel)
    return cells


# ==========================================================================
if __name__ == "__main__":
    from scipy.special import roots_legendre

    print("=" * 78)
    print("3D MAGNETOSTATICS: ANALYTIC KERNEL AGAINST QUADRATURE")
    print("=" * 78)

    def brute(corners, z0, obs, n=160):
        x1, x2, y1, y2 = corners
        u, w = roots_legendre(n)
        xs = 0.5 * (x2 - x1) * u + 0.5 * (x2 + x1)
        ys = 0.5 * (y2 - y1) * u + 0.5 * (y2 + y1)
        wx = w * 0.5 * (x2 - x1)
        wy = w * 0.5 * (y2 - y1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        W = np.outer(wx, wy)
        d = obs[None, None, :] - np.stack(
            [X, Y, np.full_like(X, z0)], axis=-1)
        R = np.linalg.norm(d, axis=-1)
        return (d * (W / R ** 3)[..., None]).sum(axis=(0, 1)) / (4 * np.pi)

    corners = (-2e-3, 3e-3, -1.5e-3, 2.5e-3)
    worst = 0.0
    for obs in (np.array([0.4e-3, -0.2e-3, 1.1e-3]),
                np.array([5e-3, 4e-3, -2e-3]),
                np.array([0.0, 0.0, 0.35e-3]),
                np.array([-1e-3, 1e-3, 8e-3])):
        a = _sheet_field(corners, 0.0, obs)
        b = brute(corners, 0.0, obs)
        err = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        worst = max(worst, err)
        print(f"  obs {obs*1e3}   analytic {a}   rel err {err:.2e}")
    print(f"\n  worst relative error against quadrature: {worst:.2e}")
    assert worst < 1e-6, "sheet kernel disagrees with quadrature"

    print("\n  cube self-demagnetising factor (exact value 1/3):")
    h = np.array([1.0, 1.0, 1.0])
    Hc = cuboid_field(h, np.zeros(3), np.eye(3), np.array([0.0, 0.0, 1.0]),
                      np.zeros((1, 3)))[0]
    print(f"    N_zz = {-Hc[2]:.6f}    error {abs(-Hc[2]-1/3):.2e}")
    assert abs(-Hc[2] - 1.0 / 3.0) < 1e-9

    print("\n  long rod along z (N_zz -> 0) and thin plate (N_zz -> 1):")
    for hz, label in ((40.0, "rod  L/D = 40"), (0.025, "plate t/w = 0.025")):
        Hc = cuboid_field(np.array([1.0, 1.0, hz]), np.zeros(3), np.eye(3),
                          np.array([0.0, 0.0, 1.0]), np.zeros((1, 3)))[0]
        print(f"    {label:<20} N_zz = {-Hc[2]:.4f}")
    print("\n  kernel verified.")
