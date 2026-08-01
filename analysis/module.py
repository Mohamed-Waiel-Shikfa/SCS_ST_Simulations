"""Stage 3: module geometry, mass properties and MuJoCo export.

Takes a design plus its selected driver and builds the actual module: an EPM
assembly on each face of a polyhedral shell, electronics in the centre, and a
printed shell sized to contain them.  Returns real mass and a real inertia
tensor, which Stage 4 needs - a dynamics result is only as good as the inertia
that went into it.

Geometry
--------
The module is a cube of side ``a_module`` with an EPM centred on each of six
faces, which is the simplest arrangement that supports motion in three
dimensions and matches the square-face constraint.  (The thesis proposes a
rhombicuboctahedron; that adds 18 more faces without changing the physics of a
single face-to-face interaction, so the cube is used here as the load case and
the face count is a design variable.)

Inertia is assembled from the exact tensors of the primitives - shell walls as
a hollow box, magnets and pole pieces as cylinders and tubes at their real
offsets, electronics as a central block - rather than approximating the module
as a uniform solid, because the EPMs sit at the extremities where they
contribute most to the moment of inertia.
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

FACE_NORMALS = np.array([[+1, 0, 0], [-1, 0, 0], [0, +1, 0],
                         [0, -1, 0], [0, 0, +1], [0, 0, -1]], dtype=float)


# --------------------------------------------------------------------------
# Inertia primitives, all about the body centre of mass, then shifted
# --------------------------------------------------------------------------
def _cyl_inertia(m, r, h, axis):
    """Solid cylinder, axis along 'axis' (0,1,2), about its own centre."""
    ir = 0.25 * m * r**2 + m * h**2 / 12.0
    ia = 0.5 * m * r**2
    I = np.diag([ir, ir, ir])
    I[axis, axis] = ia
    return I


def _tube_inertia(m, r_in, r_out, h, axis):
    ir = 0.25 * m * (r_in**2 + r_out**2) + m * h**2 / 12.0
    ia = 0.5 * m * (r_in**2 + r_out**2)
    I = np.diag([ir, ir, ir])
    I[axis, axis] = ia
    return I


def _box_inertia(m, sx, sy, sz):
    return np.diag([m * (sy**2 + sz**2) / 12.0,
                    m * (sx**2 + sz**2) / 12.0,
                    m * (sx**2 + sy**2) / 12.0])


def _shift(I, m, d):
    """Parallel axis: move an inertia tensor by offset d."""
    d = np.asarray(d, dtype=float)
    return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


# --------------------------------------------------------------------------
@dataclass
class Module:
    a: float                 # outer side length
    mass: float
    inertia: np.ndarray      # 3x3 about the centre of mass
    parts: dict
    n_faces: int
    face_offset: float       # distance from centre to the EPM pole face
    wall: float

    def summary(self):
        d = np.diag(self.inertia)
        return (f"side {self.a*1e3:.0f} mm, mass {self.mass*1e3:.0f} g, "
                f"I = diag({d[0]*1e6:.0f}, {d[1]*1e6:.0f}, {d[2]*1e6:.0f}) "
                f"g mm^2 x1e3")


def build_module(dsg, driver=None, wall=2.0e-3, pcb_fill=0.55,
                 face_recess=0.0):
    """Assemble the module and compute its mass properties.

    ``driver`` is a Driver from analysis/driver.py; its mass and volume are
    placed at the module centre.  If the electronics do not fit inside the
    shell cavity the module is reported as not buildable, which is a real
    constraint at small module sizes.

    ``face_recess`` is how far the magnet pole face sits BEHIND the outer
    surface of the module.  It defaults to zero - the pole must be flush - and
    this is not a cosmetic detail: any recess adds twice over to the magnetic
    gap between two mated modules, and at these pole diameters the force falls
    off over a couple of millimetres.  Putting a 2 mm shell wall across the
    pole face would cost most of the holding force.
    """
    a = dsg.a_module
    Rm = dsg.d_mag / 2
    Lm = dsg.l_mag
    parts = {}
    total_m = 0.0
    I = np.zeros((3, 3))

    # ---- shell: hollow box, walls of thickness `wall`, with a clear aperture
    # at each face so the pole is not shadowed by plastic
    v_shell = a**3 - (a - 2 * wall) ** 3
    r_ap = Rm + (dsg.r_clear + dsg.t_steel if dsg.circuit == "potcore" else 0)
    v_shell -= min(dsg.n_faces, 6) * np.pi * r_ap**2 * wall
    m_shell = v_shell * RHO_PLA
    I_shell = (_box_inertia(a**3 * RHO_PLA, a, a, a) -
               _box_inertia((a - 2 * wall) ** 3 * RHO_PLA,
                            a - 2 * wall, a - 2 * wall, a - 2 * wall))
    parts["shell"] = m_shell
    total_m += m_shell
    I += I_shell

    # ---- per-face EPM assembly, pole face flush with the outer surface
    has_steel = dsg.circuit == "potcore"
    r_out = Rm + dsg.r_clear + dsg.t_steel if has_steel else Rm
    m_mag = np.pi * Rm**2 * Lm * RHO_ALNICO

    # coil: annulus of copper over the magnet, packed at 60 %
    build = dsg.t_steel if has_steel else 0.5e-3
    v_coil = np.pi * ((Rm + build) ** 2 - Rm**2) * Lm
    m_coil = v_coil * 0.60 * RHO_CU

    if has_steel:
        m_back = np.pi * r_out**2 * dsg.t_steel * RHO_STEEL
        m_ann = np.pi * (r_out**2 - (Rm + dsg.r_clear) ** 2) * Lm * RHO_STEEL
    else:
        m_back = m_ann = 0.0

    m_unit = m_mag + m_coil + m_back + m_ann
    n = min(dsg.n_faces, 6)
    # pole face sits at the outer surface unless deliberately recessed
    z_face = a / 2 - face_recess
    z_mag = z_face - Lm / 2

    for k in range(n):
        nrm = FACE_NORMALS[k]
        axis = int(np.argmax(np.abs(nrm)))
        I_mag = _cyl_inertia(m_mag, Rm, Lm, axis)
        I_coil = _tube_inertia(m_coil, Rm, Rm + build, Lm, axis)
        I_unit = I_mag + I_coil
        if has_steel:
            I_unit = I_unit + _tube_inertia(m_ann, Rm + dsg.r_clear, r_out,
                                            Lm, axis)
            I_unit = I_unit + _shift(
                _cyl_inertia(m_back, r_out, dsg.t_steel, axis), m_back,
                nrm * (Lm / 2 + dsg.t_steel / 2))
        I += _shift(I_unit, m_unit, nrm * z_mag)
        total_m += m_unit

    parts["magnets"] = n * m_mag
    parts["coils"] = n * m_coil
    parts["steel"] = n * (m_back + m_ann)

    # ---- electronics at the centre
    m_drv = driver.mass if driver is not None else 0.0
    v_drv = driver.volume if driver is not None else 0.0
    if m_drv:
        side = (v_drv / pcb_fill) ** (1 / 3)
        I += _box_inertia(m_drv, side, side, side)
        parts["driver"] = m_drv
        total_m += m_drv

    # ---- fit check: does the electronics volume fit in the cavity left over?
    cavity = (a - 2 * wall) ** 3
    used = n * np.pi * r_out**2 * (Lm + (dsg.t_steel if has_steel else 0))
    free = cavity - used
    fits = free > (v_drv / pcb_fill)

    return Module(a=a, mass=total_m, inertia=I, parts=parts, n_faces=n,
                  face_offset=z_face, wall=wall), fits, free


# --------------------------------------------------------------------------
def module_xml(mod, forces_disabled=True):
    """MuJoCo MJCF for a single module body, with a site on each pole face.

    Magnetic interaction is applied externally as forces between the face
    sites, because MuJoCo has no magnetics: Stage 4 reads the face separations
    each step and applies the Stage 1 force law.
    """
    a, off = mod.a, mod.face_offset
    d = np.diag(mod.inertia)
    sites = "\n".join(
        f'        <site name="f{k}" pos="{off*n[0]:.5f} {off*n[1]:.5f} '
        f'{off*n[2]:.5f}" size="0.002" rgba="0.9 0.2 0.2 1"/>'
        for k, n in enumerate(FACE_NORMALS[:mod.n_faces]))
    return f"""      <body name="mod" pos="0 0 0">
        <freejoint/>
        <inertial pos="0 0 0" mass="{mod.mass:.6f}"
                  diaginertia="{d[0]:.9f} {d[1]:.9f} {d[2]:.9f}"/>
        <geom type="box" size="{a/2:.5f} {a/2:.5f} {a/2:.5f}"
              rgba="0.35 0.45 0.75 1" friction="0.9 0.02 0.001"/>
{sites}
      </body>"""


if __name__ == "__main__":
    from driver import select_driver
    from framework import Design, stage3_switching

    print("=" * 84)
    print("MODULE MASS PROPERTIES")
    print("=" * 84)
    print()
    for lab, kw in (("as built (LNG37, bare)",
                     dict(material="LNG37", circuit="none", v_cap=30.0)),
                    ("recommended (LNGT72, potcore)",
                     dict(material="LNGT72", circuit="potcore", v_cap=70.0))):
        for a_mm in (40, 60, 80):
            d = Design(a_module=a_mm * 1e-3, **kw)
            sw = stage3_switching(d)
            drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                                sw["n_turns"], sw["mmf_need"],
                                n_faces=d.n_faces)
            mod, fits, free = build_module(d, drv if drv.feasible else None)
            bits = "  ".join(f"{k} {v*1e3:.0f}g" for k, v in mod.parts.items())
            print(f"  {lab:<30} a={a_mm}mm  {mod.summary()}")
            print(f"  {'':<30} {bits}   fits={fits}")
        print()
