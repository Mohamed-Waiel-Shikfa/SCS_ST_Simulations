"""Module geometry: three orthogonal regular n-gon rings.

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

Why the polygon matters
-----------------------
Locomotion is by pivoting about the shared edge with a neighbour, and the pivot
angle is the exterior angle of the polygon,

    theta = 360 / n           (45 deg for n=8, 22.5 deg for n=16)

not the 90 degrees a cube would need.  A smaller pivot angle lifts the centre
of mass less, so the torque and energy needed to roll fall sharply with n.
That is the reason for this geometry, and it makes the face count a design
variable rather than a styling choice.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RHO_ALNICO = 7300.0
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


def _shift(I, m, d):
    d = np.asarray(d, dtype=float)
    return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def _oriented(I_local, m, offset, axis_dir):
    """Place an axisymmetric part with its symmetry axis along ``axis_dir``."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, axis_dir)
    c = float(np.dot(z, axis_dir))
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1.0 + c)
    return _shift(R @ I_local @ R.T, m, offset)


@dataclass
class Module:
    n_gon: int
    r_face: float            # centre to pole face
    a_face: float            # square face side
    mass: float
    inertia: np.ndarray
    parts: dict
    normals: np.ndarray
    fits: bool
    free_volume: float

    @property
    def n_faces(self):
        return len(self.normals)

    @property
    def bounding_cube(self):
        return 2.0 * self.r_face / np.cos(np.pi / self.n_gon)

    def summary(self):
        d = np.diag(self.inertia)
        return (f"n={self.n_gon} ({self.n_faces} faces), "
                f"cube {self.bounding_cube*1e3:.0f} mm, "
                f"pivot {np.degrees(pivot_angle(self.n_gon)):.1f} deg, "
                f"{self.mass*1e3:.0f} g, I={d[0]*1e6:.0f}e-6 kg m^2")


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

    Rm, Lm = dsg.d_mag / 2, dsg.l_mag
    has_steel = dsg.circuit == "potcore"
    r_out = Rm + dsg.r_clear + dsg.t_steel if has_steel else Rm
    build = dsg.t_steel if has_steel else 0.5e-3

    parts, total_m = {}, 0.0
    I = np.zeros((3, 3))

    # ---- shell: thin skin at r_face with an aperture cut for every pole
    v_skin = max(4 * np.pi * r_face**2 * wall - nf * np.pi * r_out**2 * wall,
                 0.0)
    m_shell = v_skin * RHO_PLA
    I += (2.0 / 3.0) * m_shell * r_face**2 * np.eye(3)
    parts["shell"] = m_shell
    total_m += m_shell

    # ---- EPM assemblies, pole flush with the outer surface
    m_mag = np.pi * Rm**2 * Lm * RHO_ALNICO
    m_coil = np.pi * ((Rm + build) ** 2 - Rm**2) * Lm * 0.60 * RHO_CU
    if has_steel:
        m_back = np.pi * r_out**2 * dsg.t_steel * RHO_STEEL
        m_ann = np.pi * (r_out**2 - (Rm + dsg.r_clear) ** 2) * Lm * RHO_STEEL
    else:
        m_back = m_ann = 0.0
    m_unit = m_mag + m_coil + m_back + m_ann

    I_loc = (_cyl_inertia(m_mag, Rm, Lm) +
             _tube_inertia(m_coil, Rm, Rm + build, Lm))
    if has_steel:
        I_loc += _tube_inertia(m_ann, Rm + dsg.r_clear, r_out, Lm)
        I_loc += _shift(_cyl_inertia(m_back, r_out, dsg.t_steel), m_back,
                        [0, 0, -(Lm / 2 + dsg.t_steel / 2)])

    z_mag = r_face - Lm / 2
    for nrm in normals:
        I += _oriented(I_loc, m_unit, nrm * z_mag, nrm)
        total_m += m_unit

    parts["magnets"] = nf * m_mag
    parts["coils"] = nf * m_coil
    parts["steel"] = nf * (m_back + m_ann)

    # ---- electronics at the centre
    m_drv = driver.mass if driver is not None else 0.0
    v_drv = driver.volume if driver is not None else 0.0
    if m_drv:
        side = (v_drv / pcb_fill) ** (1 / 3)
        I += np.eye(3) * (m_drv * 2 * side**2 / 12.0)
        parts["driver"] = m_drv
        total_m += m_drv

    r_in = max(r_face - Lm - (dsg.t_steel if has_steel else 0.0) - wall, 0.0)
    # Usable internal volume is NOT the inscribed sphere below the EPMs: the
    # EPMs are discrete cylinders on the faces, and the space between adjacent
    # ones is perfectly usable for electronics.  Take the module volume, remove
    # the shell skin and the EPM cylinders, and keep a packing efficiency for
    # the fact that the remaining space is an awkward shape.
    v_module = (4.0 / 3.0) * np.pi * r_face**3 * _shape_factor(n)
    v_epm = nf * np.pi * r_out**2 * (Lm + (dsg.t_steel if has_steel else 0.0))
    free = max(v_module - v_epm - v_skin, 0.0) * 0.55
    fits = free > v_drv

    return Module(n_gon=n, r_face=r_face, a_face=a_face, mass=total_m,
                  inertia=I, parts=parts, normals=normals, fits=fits,
                  free_volume=free)


def _shape_factor(n):
    """Volume of the three-ring intersection relative to its inscribed sphere.

    The solid lies between the inscribed sphere (radius r_face) and the
    circumscribed one, so the factor is a little above 1 and approaches 1 as n
    grows and the solid rounds off.
    """
    return 1.0 + 0.45 * (np.tan(np.pi / n) ** 2) * 3.0


if __name__ == "__main__":
    print("=" * 74)
    print("MODULE GEOMETRY: THREE ORTHOGONAL n-GON RINGS")
    print("=" * 74)
    print(f"\n  {'n':>3} {'faces':>6} {'3n-6':>6} {'pivot':>8} "
          f"{'face side':>10} {'cube':>8}   (r_face = 20 mm)")
    print("  " + "-" * 60)
    for n in (8, 12, 16, 20):
        nrm = ring_normals(n)
        r = 20e-3
        note = "  <- rhombicuboctahedron" if n == 8 else ""
        print(f"  {n:3d} {len(nrm):6d} {face_count(n):6d} "
              f"{np.degrees(pivot_angle(n)):7.1f}d "
              f"{2*r*np.tan(np.pi/n)*1e3:9.1f}mm "
              f"{2*r/np.cos(np.pi/n)*1e3:7.1f}mm{note}")
        assert len(nrm) == face_count(n)
    print("\n  face count matches 3n - 6 for every n; n=8 gives 18 squares,")
    print("  which is the rhombicuboctahedron.")
