"""Stage 0: build the module.  Everything downstream is built on this.

The module is the intersection of three mutually orthogonal regular n-gon
prisms.  For n = 8 this is the rhombicuboctahedron.  The construction requires
n = 8 + 4k (8, 12, 16, 20, ...): four-fold symmetry so the three orthogonal
rings close consistently, and at least 8 so the rings do not degenerate.

Face count
----------
Each ring contributes n square faces, but the 6 faces on the coordinate axes
are each shared by two rings, so a module has

    n_faces = 3n - 6          (18 for n=8, 30 for n=12, 42 for n=16)

square faces, each carrying one EPM.

The six axis faces are special
-----------------------------
Those 6 shared faces are the only ones a module may LATCH on.  They are the
faces whose normals lie along +-x, +-y, +-z, so latching only there is what
keeps an assembly on a cubic lattice: any two modules joined axis-face to
axis-face have parallel, axis-aligned frames, and the structure closes.
Latching on one of the 3n - 12 off-axis faces would join two modules at an
oblique angle and the lattice would never close again.

The off-axis faces are not decorative.  They are what the module ROLLS on -
the pivot angle is the exterior angle 360/n, not the 90 degrees a cube would
need - and they carry EPMs because rolling is driven magnetically.  They just
never become a parking place.  ``latch_faces`` and ``roll_faces`` below make
the distinction explicit so no downstream stage has to rediscover it.

Why the polygon matters
-----------------------
Locomotion is by pivoting about the shared edge with a neighbour, and

    theta = 360 / n           (45 deg for n=8, 22.5 deg for n=16)

A smaller pivot angle lifts the centre of mass less, so the torque and energy
needed to roll fall sharply with n.  That is the reason for this geometry, and
it makes the face count a design variable rather than a styling choice.

What this module now emits
--------------------------
A ``Module`` carries not just mass and inertia but a full PART LIST: every
magnet, every multi-layer coil, every steel pole piece, the capacitor bank,
the battery, the driver board and the reserved electronics envelope, each with
its position, orientation and dimensions.  That list is the single source of
truth for the mass budget, the packing check and the 3-D viewer, so a
component cannot appear in the picture without also appearing in the mass.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from coil import wind  # noqa: E402
from materials import MATERIALS  # noqa: E402

RHO_STEEL = 7870.0
RHO_PLA = 1240.0
RHO_CU = 8960.0
G = 9.81


# --------------------------------------------------------------------------
def ring_normals(n):
    """Face normals of the three orthogonal n-gon rings, deduplicated.

    Ring 0 lies in the xy-plane, ring 1 in yz, ring 2 in zx.  Faces on the
    coordinate axes belong to two rings and are kept once.
    """
    if n < 8 or (n - 8) % 4 != 0:
        raise ValueError(f"n must be 8 + 4k (8, 12, 16, ...), got {n}")
    out = []
    for ring in range(3):
        for i in range(n):
            a = 2 * np.pi * i / n
            c, s = np.cos(a), np.sin(a)
            v = np.array([c, s, 0.0] if ring == 0 else
                         [0.0, c, s] if ring == 1 else
                         [s, 0.0, c])
            if not any(np.allclose(v, w, atol=1e-9) for w in out):
                out.append(v)
    return np.array(out)


def face_count(n):
    return 3 * n - 6


def pivot_angle(n):
    """Exterior angle: rotation needed to roll onto the next face."""
    return 2.0 * np.pi / n


def latch_faces(normals):
    """Indices of the six axis faces - the only faces that may latch.

    These are the faces shared by two rings, whose normals are the coordinate
    axes.  Restricting latching to them is what preserves the cubic lattice.
    """
    return [k for k, v in enumerate(normals)
            if int(np.sum(np.abs(v) > 1e-9)) == 1]


def roll_faces(normals):
    """Indices of the off-axis faces: used for rolling, never for parking."""
    latch = set(latch_faces(normals))
    return [k for k in range(len(normals)) if k not in latch]


def hull_vertices(n_gon, r_face):
    """Vertices of the intersection of three orthogonal regular n-gon prisms.

    The solid is {x : n_k . x <= r_face for every ring normal n_k}.  Every
    vertex is the intersection of three of those planes that satisfies all the
    others, so enumerating triples is exact and cheap at these sizes.
    """
    N = ring_normals(n_gon)
    r = np.full(len(N), r_face)
    pts = []
    for i, j, k in itertools.combinations(range(len(N)), 3):
        A = N[[i, j, k]]
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        p = np.linalg.solve(A, r[[i, j, k]])
        if np.all(N @ p <= r_face + 1e-9):
            pts.append(p)
    P = np.array(pts)
    keep = []
    for p in P:
        if not any(np.linalg.norm(p - q) < 1e-9 for q in keep):
            keep.append(p)
    return np.array(keep)


# --------------------------------------------------------------------------
def _cyl_inertia(m, r, h, axis=2):
    ir = 0.25 * m * r**2 + m * h**2 / 12.0
    I = np.eye(3) * ir
    I[axis, axis] = 0.5 * m * r**2
    return I


def _tube_inertia(m, r_in, r_out, h, axis=2):
    ir = 0.25 * m * (r_in**2 + r_out**2) + m * h**2 / 12.0
    I = np.eye(3) * ir
    I[axis, axis] = 0.5 * m * (r_in**2 + r_out**2)
    return I


def _box_inertia(m, a, b, c):
    return np.diag([m * (b**2 + c**2) / 12.0,
                    m * (a**2 + c**2) / 12.0,
                    m * (a**2 + b**2) / 12.0])


def _shift(I, m, d):
    d = np.asarray(d, dtype=float)
    return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def _rot_to(axis_dir):
    """Rotation taking +z onto ``axis_dir``."""
    z = np.array([0.0, 0.0, 1.0])
    a = np.asarray(axis_dir, dtype=float)
    a = a / np.linalg.norm(a)
    v = np.cross(z, a)
    c = float(np.dot(z, a))
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _oriented(I_local, m, offset, axis_dir):
    R = _rot_to(axis_dir)
    return _shift(R @ I_local @ R.T, m, offset)


# --------------------------------------------------------------------------
@dataclass
class Part:
    """One physical component, with enough geometry to draw and weigh it."""

    name: str
    kind: str                  # magnet|coil|steel|cap|battery|pcb|shell|
                               # electronics
    shape: str                 # cylinder | tube | box | hull
    centre: np.ndarray
    axis: np.ndarray
    mass: float
    dims: dict
    face: int = -1
    colour: str = "#888888"
    note: str = ""

    def to_json(self):
        return dict(name=self.name, kind=self.kind, shape=self.shape,
                    centre=[float(v) for v in self.centre],
                    axis=[float(v) for v in self.axis],
                    mass=float(self.mass),
                    dims={k: float(v) for k, v in self.dims.items()},
                    face=int(self.face), colour=self.colour, note=self.note)


@dataclass
class Module:
    n_gon: int
    r_face: float
    a_face: float
    mass: float
    inertia: np.ndarray
    parts: list
    normals: np.ndarray
    fits: bool
    free_volume: float
    used_volume: float = 0.0
    winding: object = None
    mass_by_kind: dict = field(default_factory=dict)
    latch: list = field(default_factory=list)
    roll: list = field(default_factory=list)

    @property
    def n_faces(self):
        return len(self.normals)

    @property
    def bounding_cube(self):
        """Side of the smallest axis-aligned cube containing the module.

        The solid is the intersection of the half-spaces n_k . x <= r_face,
        and the axis directions are themselves ring normals, so the extent
        along each axis is exactly r_face - not r_face / cos(pi/n).  That
        expression is the CIRCUMRADIUS of the polygon cross-section, which is
        the right quantity for the pivot lift but the wrong one for packaging.
        """
        return 2.0 * self.r_face

    @property
    def r_vertex(self):
        """Distance from the centre to a pivot edge of the cross-section."""
        return self.r_face / np.cos(np.pi / self.n_gon)

    def summary(self):
        d = np.diag(self.inertia)
        return (f"n={self.n_gon} ({self.n_faces} faces, {len(self.latch)} "
                f"latching), cube {self.bounding_cube*1e3:.0f} mm, "
                f"pivot {np.degrees(pivot_angle(self.n_gon)):.1f} deg, "
                f"{self.mass*1e3:.0f} g, I={d[0]*1e6:.0f}e-6 kg m^2")

    def parts_json(self):
        return [p.to_json() for p in self.parts]


def _shape_factor(n):
    """Volume of the three-ring intersection relative to its inscribed sphere.

    The solid lies between the inscribed sphere (radius r_face) and the
    circumscribed one, so the factor is a little above 1 and approaches 1 as n
    grows and the solid rounds off.
    """
    return 1.0 + 0.45 * (np.tan(np.pi / n) ** 2) * 3.0


def build_module(dsg, driver=None, wall=1.5e-3, pcb_fill=0.55):
    """Assemble the module and compute mass properties.

    Pole faces are open apertures.  Any shell material in front of a pole adds
    to the magnetic gap twice over when two modules mate, and the recess study
    showed 0.25 mm of wall costs more than half the holding force.
    """
    n = dsg.n_gon
    normals = ring_normals(n)
    nf = len(normals)
    r_face = dsg.r_face
    a_face = 2.0 * r_face * np.tan(np.pi / n)
    rho_mag = MATERIALS[dsg.material]["rho"]

    Rm, Lm = dsg.d_mag / 2, dsg.l_mag
    has_steel = dsg.circuit == "potcore"

    # The winding is a real multi-layer coil with its own build height, no
    # longer borrowed from the keeper wall thickness.
    w = wind(Rm, Lm, dsg.wire_d, int(getattr(dsg, "n_layers", 4)))
    build = w.build
    r_ann_in = Rm + build + dsg.r_clear
    r_out = r_ann_in + (dsg.t_steel if has_steel else 0.0)

    parts, total_m = [], 0.0
    I = np.zeros((3, 3))

    # ---- shell: thin skin at r_face with an aperture cut for every pole
    v_skin = max(4 * np.pi * r_face**2 * wall - nf * np.pi * r_out**2 * wall,
                 0.0)
    m_shell = v_skin * RHO_PLA
    I += (2.0 / 3.0) * m_shell * r_face**2 * np.eye(3)
    parts.append(Part("shell", "shell", "hull", np.zeros(3),
                      np.array([0.0, 0.0, 1.0]), m_shell,
                      dict(r=r_face, wall=wall), colour="#7a8492",
                      note=f"PLA skin with {nf} open pole apertures"))
    total_m += m_shell

    # ---- EPM assemblies, pole flush with the outer surface
    m_mag = np.pi * Rm**2 * Lm * rho_mag
    m_coil = w.mass
    if has_steel:
        m_back = np.pi * r_out**2 * dsg.t_steel * RHO_STEEL
        m_ann = np.pi * (r_out**2 - r_ann_in**2) * Lm * RHO_STEEL
    else:
        m_back = m_ann = 0.0
    m_unit = m_mag + m_coil + m_back + m_ann

    I_loc = (_cyl_inertia(m_mag, Rm, Lm) +
             _tube_inertia(m_coil, Rm, Rm + build, Lm))
    if has_steel:
        I_loc += _tube_inertia(m_ann, r_ann_in, r_out, Lm)
        I_loc += _shift(_cyl_inertia(m_back, r_out, dsg.t_steel), m_back,
                        [0, 0, -(Lm / 2 + dsg.t_steel / 2)])

    z_mag = r_face - Lm / 2
    latch = latch_faces(normals)
    for k, nrm in enumerate(normals):
        I += _oriented(I_loc, m_unit, nrm * z_mag, nrm)
        total_m += m_unit
        role = "latching" if k in latch else "rolling"
        parts.append(Part(f"magnet {k}", "magnet", "cylinder",
                          nrm * z_mag, nrm, m_mag, dict(r=Rm, h=Lm),
                          face=k, colour="#c85a3c",
                          note=f"{dsg.material}, {role} face"))
        parts.append(Part(f"coil {k}", "coil", "tube", nrm * z_mag, nrm,
                          m_coil,
                          dict(r_in=Rm, r_out=Rm + build, h=Lm,
                               layers=w.n_layers,
                               turns_per_layer=w.turns_per_layer,
                               turns=w.n_turns, wire_d=dsg.wire_d),
                          face=k, colour="#d9a441",
                          note=(f"{w.n_layers} layers x {w.turns_per_layer} "
                                f"turns of {dsg.wire_d*1e3:.2f} mm wire")))
        if has_steel:
            parts.append(Part(f"pole cup {k}", "steel", "tube",
                              nrm * z_mag, nrm, m_ann,
                              dict(r_in=r_ann_in, r_out=r_out, h=Lm),
                              face=k, colour="#5b6675",
                              note="return-path annulus"))
            parts.append(Part(f"back plate {k}", "steel", "cylinder",
                              nrm * (r_face - Lm - dsg.t_steel / 2), nrm,
                              m_back, dict(r=r_out, h=dsg.t_steel),
                              face=k, colour="#4a5460",
                              note="closes the magnetic circuit"))

    # ---- electronics at the centre, as real objects with real sizes
    m_drv = getattr(driver, "mass", 0.0) if driver is not None else 0.0
    v_drv = getattr(driver, "volume", 0.0) if driver is not None else 0.0
    if driver is not None and getattr(driver, "feasible", False) \
            and np.isfinite(v_drv) and v_drv > 0:
        cap_v = getattr(driver, "cap_volume", 0.0) or v_drv * 0.35
        cap_m = getattr(driver, "cap_mass", 0.0) or m_drv * 0.35
        bat_v = getattr(driver, "batt_volume", 0.0) or v_drv * 0.30
        bat_m = getattr(driver, "batt_mass", 0.0) or m_drv * 0.30
        pcb_v = max(v_drv - cap_v - bat_v, 1e-9)
        pcb_m = max(m_drv - cap_m - bat_m, 0.0)

        n_caps = max(int(getattr(driver, "n_caps", 1)), 1)
        v_one = cap_v / n_caps
        r_cap = max((v_one / (2.0 * np.pi)) ** (1 / 3), 1e-4)
        h_cap = v_one / (np.pi * r_cap ** 2)
        for i in range(n_caps):
            off = (i - (n_caps - 1) / 2.0) * (2.2 * r_cap)
            c = np.array([off, 0.0, 0.0])
            parts.append(Part(f"capacitor {i}", "cap", "cylinder", c,
                              np.array([0.0, 0.0, 1.0]), cap_m / n_caps,
                              dict(r=r_cap, h=h_cap), colour="#2f6f8f",
                              note=str(getattr(driver, "cap_name", "bank"))))
            I += _shift(_cyl_inertia(cap_m / n_caps, r_cap, h_cap),
                        cap_m / n_caps, c)
        total_m += cap_m

        a_b = max((bat_v / 0.4) ** (1 / 3), 1e-4)
        c = np.array([0.0, 0.0, -a_b * 0.6])
        parts.append(Part("battery", "battery", "box", c,
                          np.array([0.0, 0.0, 1.0]), bat_m,
                          dict(a=a_b, b=a_b, c=0.4 * a_b), colour="#3f8f5a",
                          note="Li-ion, sized for 200 switching events/face"))
        I += _shift(_box_inertia(bat_m, a_b, a_b, 0.4 * a_b), bat_m, c)
        total_m += bat_m

        a_p = max((pcb_v / pcb_fill / 0.12) ** (1 / 3), 1e-4)
        t_p = 1.6e-3
        c = np.array([0.0, 0.0, a_p * 0.35])
        parts.append(Part("driver board", "pcb", "box", c,
                          np.array([0.0, 0.0, 1.0]), pcb_m,
                          dict(a=a_p, b=a_p, c=t_p), colour="#2e7d5b",
                          note=(f"{getattr(driver, 'n_fets', 0)} x "
                                f"{getattr(driver, 'mosfet_name', 'FET')} "
                                f"({getattr(driver, 'topology', '')})")))
        I += _shift(_box_inertia(pcb_m, a_p, a_p, t_p), pcb_m, c)
        total_m += pcb_m

    # Usable internal volume is NOT the inscribed sphere below the EPMs: the
    # EPMs are discrete cylinders on the faces, and the space between adjacent
    # ones is perfectly usable for electronics.  Take the module volume, remove
    # the shell skin and the EPM cylinders, and keep a packing efficiency for
    # the fact that the remaining space is an awkward shape.
    v_module = (4.0 / 3.0) * np.pi * r_face**3 * _shape_factor(n)
    v_epm = nf * np.pi * r_out**2 * (Lm + (dsg.t_steel if has_steel else 0.0))
    free = max(v_module - v_epm - v_skin, 0.0) * 0.55
    fits = bool(free > v_drv)

    if free > 0:
        r_env = (3.0 * free / (4.0 * np.pi)) ** (1 / 3)
        parts.append(Part("electronics envelope", "electronics", "cylinder",
                          np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0,
                          dict(r=r_env, h=2 * r_env), colour="#e8eef6",
                          note=(f"{free*1e6:.1f} cc usable, "
                                f"{v_drv*1e6:.1f} cc needed")))

    by_kind = {}
    for p in parts:
        by_kind[p.kind] = by_kind.get(p.kind, 0.0) + p.mass

    return Module(n_gon=n, r_face=r_face, a_face=a_face, mass=total_m,
                  inertia=I, parts=parts, normals=normals, fits=fits,
                  free_volume=free, used_volume=v_drv, winding=w,
                  mass_by_kind=by_kind, latch=latch,
                  roll=roll_faces(normals))


if __name__ == "__main__":
    print("=" * 78)
    print("MODULE GEOMETRY: THREE ORTHOGONAL n-GON RINGS")
    print("=" * 78)
    print(f"\n  {'n':>3} {'faces':>6} {'3n-6':>6} {'latch':>6} {'roll':>5} "
          f"{'pivot':>8} {'face side':>10} {'cube':>8}   (r_face = 20 mm)")
    print("  " + "-" * 74)
    for n in (8, 12, 16, 20):
        nrm = ring_normals(n)
        r = 20e-3
        note = "  <- rhombicuboctahedron" if n == 8 else ""
        print(f"  {n:3d} {len(nrm):6d} {face_count(n):6d} "
              f"{len(latch_faces(nrm)):6d} {len(roll_faces(nrm)):5d} "
              f"{np.degrees(pivot_angle(n)):7.1f}d "
              f"{2*r*np.tan(np.pi/n)*1e3:9.1f}mm "
              f"{2*r/np.cos(np.pi/n)*1e3:7.1f}mm{note}")
        assert len(nrm) == face_count(n)
        assert len(latch_faces(nrm)) == 6

    print("""
  Every n has exactly 6 latching faces - the coordinate axes - however many
  faces it has in total.  That is what keeps an assembly on a cubic lattice:
  the other 3n-12 faces exist for rolling and must never become a parking
  place.""")

    from framework import Design                              # noqa: E402
    m = build_module(Design())
    print(f"\n  Example build: {m.summary()}")
    print(f"  winding: {m.winding.summary()}")
    print(f"\n  {'part kind':<16} {'mass':>9} {'share':>7}")
    print("  " + "-" * 36)
    for k, v in sorted(m.mass_by_kind.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v*1e3:8.2f}g {v/m.mass*100:6.1f}%")
    print(f"  {'TOTAL':<16} {m.mass*1e3:8.2f}g")
    print(f"\n  {len(m.parts)} parts emitted for the viewer; free volume "
          f"{m.free_volume*1e6:.1f} cc")
