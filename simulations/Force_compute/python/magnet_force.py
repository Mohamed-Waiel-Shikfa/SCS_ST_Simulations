"""Accurate permanent-magnet force engine.

Two solvers, no empirical fudge factors:

``cuboid_pair_force``
    Closed-form axial force between two axially magnetised cuboids
    (Akoun & Yonnet, IEEE Trans. Magn. 20(5), 1984).  Exact for materials with
    a straight recoil line and mu_r ~ 1 (NdFeB, SmCo).

``CoaxialRodPair``
    Two coaxial cylinders solved with a *non-uniform* magnetisation.  Each rod
    is split into axial slabs; the polarisation of every slab is solved
    self-consistently against the material's second-quadrant curve, including
    recoil behaviour and irreversible loss history.  Needed for Alnico, whose
    coercivity is so low that the operating point moves substantially with the
    air gap.

Both solvers are built from the same pair of Lipschitz-Hankel kernels, which
reduce to smooth one-dimensional integrals over theta in [0, pi/2]:

    G(b)   = int_0^inf J1(x)^2 exp(-b x) / x   dx
           = 1/pi int_0^{pi/2} 4c^2 / (s + b)^2 dtheta

    Phi(b) = int_0^inf J1(x)^2 (1 - exp(-b x)) / x^2 dx
           = 1/pi int_0^{pi/2} [ 4c/3 - 4c^2 (2s + b) / (3 (s + b)^2) ] dtheta

with c = cos(theta) and s = sqrt(b^2 + 4c^2).  Both forms are free of the
cancellation that plagues the naive algebraic expressions.

Sign convention: forces are returned as signed axial values, negative meaning
attraction between the two bodies.

Units are SI throughout (metres, tesla, amperes per metre, newtons).  The
convenience wrappers at the bottom of the file accept millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

MU0 = 4.0e-7 * np.pi

# --------------------------------------------------------------------------
# Quadrature nodes for the two kernels (theta in [0, pi/2])
# --------------------------------------------------------------------------
_NGL = 400
_x, _w = np.polynomial.legendre.leggauss(_NGL)
_THETA = 0.25 * np.pi * (_x + 1.0)
_WEIGHT = 0.25 * np.pi * _w
_COS = np.cos(_THETA)


def force_kernel(beta):
    """G(beta) with beta = |dz| / R.  G(0) = 1/2."""
    b = np.asarray(beta, dtype=float)[..., None]
    c = _COS
    s = np.sqrt(b * b + 4.0 * c * c)
    return ((4.0 * c * c / (s + b) ** 2) * _WEIGHT).sum(-1) / np.pi


def field_kernel(beta):
    """Phi(beta) with beta = |dz| / R.  Phi(0) = 0, Phi(inf) = 4/(3 pi)."""
    b = np.asarray(beta, dtype=float)[..., None]
    c = _COS
    s = np.sqrt(b * b + 4.0 * c * c)
    psi = (4.0 / 3.0) * c - 4.0 * c * c * (2.0 * s + b) / (3.0 * (s + b) ** 2)
    return (psi * _WEIGHT).sum(-1) / np.pi


def cylinder_demag_factor(radius, length):
    """Magnetometric (volume-averaged) demagnetising factor of a cylinder."""
    return 2.0 * radius * field_kernel(length / radius) / length


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
@dataclass
class Material:
    """Second-quadrant intrinsic curve J(H) plus a recoil permeability.

    The major curve is ``J = Br * (1 - (|H| / Hcj) ** p) ** q`` which is flexible
    enough to represent both the near-straight NdFeB line and the strongly kneed
    Alnico curve.  ``strength`` scales the whole curve and is used to represent a
    rod that is not fully magnetised (or has suffered cumulative losses).
    """

    name: str
    Br: float
    Hcj: float  # positive magnitude, A/m
    mu_rec: float = 1.05
    p: float = 8.0
    q: float = 0.5
    strength: float = 1.0

    # ---- construction from a datasheet row -------------------------------
    @classmethod
    def from_datasheet(cls, name, Br, Hcb, Hcj, BHmax, mu_rec):
        """Fit (p, q) so the curve reproduces the catalogue Hcb and (BH)max."""

        def residual(v):
            p, q = np.exp(v)
            trial = cls(name, Br, Hcj, mu_rec, p, q)
            H = -np.linspace(0.0, Hcj, 4001)
            B = trial.B(H)
            # Hcb: where the normal curve crosses zero
            idx = np.argmax(B <= 0.0)
            hcb = -H[idx] if idx else Hcj
            bh = np.max(B * (-H))
            return [(hcb - Hcb) / Hcb, (bh - BHmax) / BHmax]

        sol = least_squares(residual, np.log([8.0, 0.5]), xtol=1e-12, ftol=1e-12)
        p, q = np.exp(sol.x)
        return cls(name, Br, Hcj, mu_rec, p, q)

    # ---- curve -----------------------------------------------------------
    def J(self, H):
        """Intrinsic polarisation on the major demagnetisation curve."""
        h = np.clip(-np.asarray(H, dtype=float) / self.Hcj, 0.0, 1.0)
        return self.strength * self.Br * (1.0 - h**self.p) ** self.q

    def B(self, H):
        return self.J(H) + MU0 * np.asarray(H, dtype=float)

    def J_recoil(self, H, H_min):
        """Polarisation on the recoil line anchored at the deepest point H_min.

        ``minimum`` with the major curve automatically returns the major curve
        whenever H drops below H_min, so no branching is required.
        """
        anchor = self.J(H_min)
        recoil = anchor + MU0 * (self.mu_rec - 1.0) * (np.asarray(H) - H_min)
        return np.minimum(recoil, self.J(H))

    def scaled(self, factor):
        return Material(self.name, self.Br, self.Hcj, self.mu_rec, self.p,
                        self.q, self.strength * factor)


def alnico_lng37(mu_rec=4.0):
    """LNG37 / Alnico 5, from the supplier table in Force_compute/."""
    return Material.from_datasheet("LNG37 (Alnico 5)", Br=1.20, Hcb=48e3,
                                   Hcj=49e3, BHmax=37e3, mu_rec=mu_rec)


def ndfeb(grade_Br=1.30, Hcj=955e3, mu_rec=1.05):
    """Sintered NdFeB.  p large / q ~ 1 gives the familiar square curve."""
    return Material("NdFeB", Br=grade_Br, Hcj=Hcj, mu_rec=mu_rec, p=60.0, q=0.5)


# --------------------------------------------------------------------------
# Cuboid pair - exact closed form
# --------------------------------------------------------------------------
_SIGNS = np.array(np.meshgrid(*([[0, 1]] * 6), indexing="ij")).reshape(6, -1).T


def cuboid_pair_force(J1, J2, dims1, dims2, offset):
    """Axial force between two z-magnetised cuboids (Akoun & Yonnet).

    Parameters
    ----------
    J1, J2 : polarisation of each magnet in tesla (both along +z)
    dims1, dims2 : full side lengths (dx, dy, dz) in metres
    offset : centre-to-centre vector (dx, dy, dz) in metres

    Returns the z-component of the force on magnet 2 (negative = attraction).
    """
    a, b, c = np.asarray(dims1, dtype=float) / 2.0
    A, B, C = np.asarray(dims2, dtype=float) / 2.0
    al, be, ga = offset

    i, j, k, l, p, q = _SIGNS.T
    u = al + A * (-1.0) ** j - a * (-1.0) ** i
    v = be + B * (-1.0) ** l - b * (-1.0) ** k
    w = ga + C * (-1.0) ** q - c * (-1.0) ** p
    r = np.sqrt(u * u + v * v + w * w)
    sign = (-1.0) ** (i + j + k + l + p + q)

    term = -r * w
    with np.errstate(divide="ignore", invalid="ignore"):
        term = term + np.where(np.abs(r - u) > 1e-18, -u * w * np.log(np.abs(r - u)), 0.0)
        term = term + np.where(np.abs(r - v) > 1e-18, -v * w * np.log(np.abs(r - v)), 0.0)
        term = term + np.where(np.abs(w) > 1e-18, u * v * np.arctan(u * v / (r * w)), 0.0)
    return J1 * J2 / (4.0 * np.pi * MU0) * float(np.sum(sign * term))


# --------------------------------------------------------------------------
# Coaxial rod pair - self-consistent non-uniform magnetisation
# --------------------------------------------------------------------------
@dataclass
class CoaxialRodPair:
    """Two identical coaxial cylinders, north poles facing each other.

    Rod A occupies z in [-L, 0], rod B occupies z in [gap, gap + L].  Each rod is
    divided into ``n_slabs`` axial slabs of uniform polarisation, which are solved
    self-consistently against ``material``.
    """

    radius: float
    length: float
    material: Material
    n_slabs: int = 24
    H_min: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.H_min is None:
            self.H_min = np.zeros(2 * self.n_slabs)
        self._M_cache = {}
        self._J_last = None

    # ---- geometry helpers ------------------------------------------------
    def _edges(self, gap):
        n, L = self.n_slabs, self.length
        a = np.linspace(-L, 0.0, n + 1)
        b = np.linspace(gap, gap + L, n + 1)
        return a, b

    def _coupling(self, gap):
        """Matrix M with H_avg(slab i) = sum_k M[i, k] * J_k.

        Depends only on geometry, so it is cached across the repeated solves of
        a measurement sequence.
        """
        key = round(float(gap), 12)
        cached = self._M_cache.get(key)
        if cached is not None:
            return cached

        R = self.radius
        ea, eb = self._edges(gap)
        lo = np.concatenate([ea[:-1], eb[:-1]])
        hi = np.concatenate([ea[1:], eb[1:]])
        dz = hi - lo

        # sheet positions: bottom and top face of every slab
        z_bot, z_top = lo, hi

        def avg(z_sheet):
            """Average H_z in every slab from a unit (1 T) sheet at z_sheet."""
            u1 = np.abs(lo[:, None] - z_sheet[None, :]) / R
            u2 = np.abs(hi[:, None] - z_sheet[None, :]) / R
            return (R / (MU0 * dz[:, None])) * (field_kernel(u2) - field_kernel(u1))

        M = avg(z_top) - avg(z_bot)
        self._M_cache[key] = M
        return M

    # ---- solver ----------------------------------------------------------
    def solve(self, gap, freeze_history=True, tol=1e-9, max_iter=80, J0=None):
        """Solve the magnetisation integral equation J = law(M @ J).

        A damped Newton iteration is used rather than successive substitution:
        near the knee of a low-coercivity curve dJ/dH is of order Br/2 kA/m, so
        the fixed-point map has a spectral radius in the hundreds and simple
        relaxation cannot converge.
        """
        M = self._coupling(gap)
        law = lambda H: self.material.J_recoil(H, self.H_min)  # noqa: E731
        if J0 is not None:
            J = J0.copy()
        elif self._J_last is not None:
            J = self._J_last.copy()
        else:
            J = np.full(2 * self.n_slabs, self.material.J(0.0))
        eye = np.eye(2 * self.n_slabs)
        dH = max(1.0, 1e-5 * self.material.Hcj)

        r = J - law(M @ J)
        for _ in range(max_iter):
            if np.max(np.abs(r)) < tol:
                break
            H = M @ J
            deriv = (law(H + dH) - law(H - dH)) / (2.0 * dH)
            step = np.linalg.solve(eye - deriv[:, None] * M, -r)
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
                J_try = np.clip(J + alpha * step, 0.0, self.material.J(0.0))
                r_try = J_try - law(M @ J_try)
                if np.max(np.abs(r_try)) < np.max(np.abs(r)):
                    break
            J, r = J_try, r_try
        else:
            if np.max(np.abs(r)) > 1e-6:
                if J0 is None and self._J_last is not None:
                    # the warm start was a bad basin; retry from the saturated state
                    self._J_last = None
                    return self.solve(gap, freeze_history, tol, max_iter)
                raise RuntimeError(f"rod solver did not converge (residual "
                                   f"{np.max(np.abs(r)):.2e} T at gap {gap:g} m)")

        H = M @ J
        self._J_last = J
        if not freeze_history:
            self.H_min = np.minimum(self.H_min, H)
        return J, H

    # ---- force -----------------------------------------------------------
    def _sheet_charges(self, J):
        """Net surface polarisation at each slab interface of both rods."""
        n = self.n_slabs
        JA, JB = J[:n], J[n:]
        sA = np.concatenate([[-JA[0]], JA[:-1] - JA[1:], [JA[-1]]])
        sB = np.concatenate([[-JB[0]], JB[:-1] - JB[1:], [JB[-1]]])
        return sA, sB

    def force(self, gap, freeze_history=True):
        """Axial force on rod B (negative = attraction), in newtons."""
        J, _ = self.solve(gap, freeze_history=freeze_history)
        ea, eb = self._edges(gap)
        sA, sB = self._sheet_charges(J)
        d = eb[None, :] - ea[:, None]
        k = force_kernel(np.abs(d) / self.radius)
        # rod B always sits above rod A, so coincident sheets (gap = 0) are
        # approached from d -> 0+ and must not be zeroed by np.sign
        direction = np.where(d >= 0.0, 1.0, -1.0)
        pref = np.pi * self.radius**2 / MU0
        return pref * float(np.sum(direction * np.outer(sA, sB) * k))

    # ---- history -----------------------------------------------------------
    def open_circuit(self, separation=None):
        """Drive both rods to their open-circuit operating point.

        This is what happens every time the pair is pulled apart; for a low
        coercivity material it is the deepest excursion of the whole test and
        therefore sets the recoil line used by every subsequent measurement.
        """
        if separation is None:
            separation = 200.0 * self.length
        self.solve(separation, freeze_history=False)

    def reset_history(self):
        self.H_min = np.zeros(2 * self.n_slabs)


# --------------------------------------------------------------------------
# Millimetre convenience wrappers
# --------------------------------------------------------------------------
def block_pair_force_mm(Br, face_w, face_h, thickness, gap, lateral_offset=0.0):
    """Attraction between two identical blocks, pole faces facing, all in mm."""
    w, h, t = face_w * 1e-3, face_h * 1e-3, thickness * 1e-3
    return abs(cuboid_pair_force(Br, Br, (w, h, t), (w, h, t),
                                 (lateral_offset * 1e-3, 0.0, t + gap * 1e-3)))


def rod_pair_force_mm(material, diameter, length, gap, n_slabs=24):
    """Attraction between two identical rods, virgin state, all in mm."""
    pair = CoaxialRodPair(diameter * 5e-4, length * 1e-3, material, n_slabs)
    return abs(pair.force(gap * 1e-3))


# --------------------------------------------------------------------------
# Self-tests
# --------------------------------------------------------------------------
def _self_test():
    ok = True

    def check(label, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {label:<52} {got:12.6g} (want {want:g})")

    print("kernels")
    check("G(0) = 1/2", float(force_kernel(0.0)), 0.5, 1e-9)
    check("Phi(0) = 0", float(field_kernel(0.0)), 0.0, 1e-12)
    check("Phi(inf) = 4/(3 pi)", float(field_kernel(1e6)), 4.0 / (3.0 * np.pi), 1e-6)

    print("magnetometric demagnetising factors")
    # Chen, Brug & Goldfarb, IEEE Trans. Magn. 27(4) 1991, magnetometric column.
    # Cross-checked here against an independent elliptic-integral calculation.
    check("cylinder L/D = 1", float(cylinder_demag_factor(0.5, 1.0)), 0.31158, 5e-4)
    check("cylinder L/D = 2", float(cylinder_demag_factor(0.5, 2.0)), 0.18186, 5e-4)
    check("cylinder L/D = 5", float(cylinder_demag_factor(0.5, 5.0)), 0.07991, 5e-4)
    check("thin disc L/D -> 0", float(cylinder_demag_factor(1.0, 1e-4)), 1.0, 1e-3)

    print("uniformly magnetised rods (Maxwell / dipole limits)")
    R, L, J = 5e-3, 500e-3, 1.0
    rigid = Material("rigid", Br=J, Hcj=1e12, mu_rec=1.0, p=60.0, q=0.5)
    pair = CoaxialRodPair(R, L, rigid, n_slabs=6)
    A = np.pi * R * R
    check("contact force -> J^2 A / (2 mu0)", abs(pair.force(0.0)),
          J * J * A / (2.0 * MU0), 0.02 * J * J * A / (2.0 * MU0))

    R, L, sep = 2e-3, 4e-3, 400e-3
    pair = CoaxialRodPair(R, L, rigid, n_slabs=8)
    m = (J / MU0) * np.pi * R * R * L
    z = L + sep
    check("far field -> 3 mu0 m^2 / (2 pi z^4)", abs(pair.force(sep)),
          3.0 * MU0 * m * m / (2.0 * np.pi * z**4), 0.03 * 3.0 * MU0 * m * m / (2.0 * np.pi * z**4))

    print("cuboid closed form")
    s = 4e-3
    f_cube = abs(cuboid_pair_force(1.0, 1.0, (s, s, s), (s, s, s), (0, 0, s)))
    check("touching cubes < J^2 A / (2 mu0)", f_cube < 1.0 * s * s / (2 * MU0), True, 0)
    big = 60e-3
    f_far = abs(cuboid_pair_force(1.0, 1.0, (s, s, s), (s, s, s), (0, 0, big)))
    m = (1.0 / MU0) * s**3
    check("cuboid far field -> dipole", f_far, 3.0 * MU0 * m * m / (2.0 * np.pi * big**4),
          0.03 * 3.0 * MU0 * m * m / (2.0 * np.pi * big**4))

    print("LNG37 curve fitted from the supplier table")
    aln = alnico_lng37()
    H = -np.linspace(0.0, aln.Hcj, 20001)
    B = aln.B(H)
    hcb = -H[np.argmax(B <= 0.0)]
    check("Hcb", hcb, 48e3, 1e3)
    check("(BH)max", float(np.max(B * -H)), 37e3, 1e3)
    check("Br", aln.J(0.0), 1.20, 1e-6)

    print("\n" + ("ALL SELF-TESTS PASSED" if ok else "SELF-TESTS FAILED"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
