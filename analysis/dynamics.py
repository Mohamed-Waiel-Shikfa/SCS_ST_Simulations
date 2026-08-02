"""Stage 4: rigid-body dynamics of modules interacting through their EPMs.

MuJoCo has no magnetics, so the magnetic interaction is applied as an external
force/torque field each step.

The force model
---------------
A point dipole is not usable here: modules interact at separations comparable
to the magnet size, where a dipole is badly wrong, and the pivot manoeuvre puts
faces at large relative angles where only a model with real geometry gives
sensible torques.

Each EPM is therefore represented by its two equivalent magnetic pole charges,

    q = J * A            at the front pole face
   -q                    at the back face, one magnet length behind

which is the same Coulombian representation used by the validated free-space
solver in ``magnet_force.py``.  It reduces to the correct dipole in the far
field, behaves correctly at close range, and gives forces AND torques at
arbitrary relative pose, which is what a pivot needs.

``J`` is not assumed: it is taken from the Stage 1 FEM operating point for that
design and working state, so the dynamics inherit the self-demagnetisation
physics.  Because attraction and repulsion have different operating points, the
two states carry different charges - which is precisely the asymmetry that
matters for locomotion.

Calibration
-----------
``calibrate_charge`` scales the charge so the charge-pair model reproduces the
FEM axial force at the working gap.  Everything after that is geometry.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

import mujoco  # noqa: E402

from module import build_module, ring_normals  # noqa: E402

MU0 = 4.0e-7 * np.pi
G = 9.81


# --------------------------------------------------------------------------
@dataclass
class EPMSpec:
    """Everything the dynamics needs to know about one face's magnet."""

    area: float            # pole face area
    radius: float          # pole face radius
    length: float          # magnet length (charge separation)
    q_attract: float       # total pole charge at the nominal gap
    q_repel: float
    sub: np.ndarray        # sub-charge offsets in the face plane (n x 2)
    gaps: np.ndarray = None        # calibration separations
    q_att_of_gap: np.ndarray = None
    q_rep_of_gap: np.ndarray = None
    f_att_of_gap: np.ndarray = None
    f_rep_of_gap: np.ndarray = None

    def charge(self, sep, repel=False):
        """Pole charge at face separation ``sep``.

        The charge is a function of separation, not a constant.  A magnet's
        polarisation depends on the permeance of the circuit it sits in, and
        that circuit includes its neighbour: pulling two modules apart drives
        both magnets further down their demagnetisation curves.  Holding the
        charge fixed would attribute all of the change to distance and none to
        the magnets weakening, and the force would decay too fast.
        """
        tab = self.q_rep_of_gap if repel else self.q_att_of_gap
        if tab is None or self.gaps is None:
            return self.q_repel if repel else self.q_attract
        return float(np.interp(abs(sep), self.gaps, tab))

    def force(self, sep, repel=False):
        """Aligned axial force at separation ``sep``, straight from the FEM."""
        tab = self.f_rep_of_gap if repel else self.f_att_of_gap
        if tab is None or self.gaps is None:
            return 0.0
        return float(np.interp(abs(sep), self.gaps, tab))


def disc_points(radius, n_rings=2, n_per_ring=6):
    """Equal-area sample points over a disc.

    A pole face must not be treated as a single point charge.  Two mated
    modules sit a fraction of a millimetre apart while the pole is several
    millimetres across, so at the working gap the face is nowhere near a point:
    a lumped charge diverges as 1/r^2 where the true force saturates at
    J^2 A / (2 mu0).  Spreading the charge over the face regularises this at the
    correct physical scale instead of with an arbitrary softening length.
    """
    pts = [(0.0, 0.0)]
    for i in range(1, n_rings + 1):
        # equal-area radii
        r = radius * np.sqrt(i / (n_rings + 0.5))
        m = n_per_ring * i
        for j in range(m):
            th = 2 * np.pi * j / m + (0.5 * np.pi / m) * (i % 2)
            pts.append((r * np.cos(th), r * np.sin(th)))
    return np.array(pts)


def _soft_radius(spec):
    """Regularisation length: the sub-charge spacing, not the pole size.

    It must be well below the working gap or it suppresses the very near-field
    force the model exists to represent.  Crucially the SAME value is used when
    calibrating, so the calibration absorbs it exactly and the model reproduces
    the FEM at the calibration gap by construction.
    """
    return 0.25 * spec.radius / max(len(spec.sub) ** 0.5, 1.0)


def charge_pair_force(q, radius, length, gap, sub, r_soft):
    """Axial force between two coaxial, face-to-face charge discs."""
    n = len(sub)
    qi = q / n
    za = [(0.0, +1), (-length, -1)]
    zb = [(gap, -1), (gap + length, +1)]
    dxy = sub[:, None, :] - sub[None, :, :]
    d2 = dxy[..., 0] ** 2 + dxy[..., 1] ** 2
    f = 0.0
    for z1, s1 in za:
        for z2, s2 in zb:
            dz = z2 - z1
            r = np.maximum(np.sqrt(d2 + dz * dz), r_soft)
            f += s1 * s2 * qi * qi / (4 * np.pi * MU0) * np.sum(dz / r**3)
    return f


def calibrate_charge(F_target, radius, length, gap, sub, r_soft):
    """Charge magnitude that reproduces ``F_target`` at ``gap``."""
    q0 = 1.0 * np.pi * radius**2
    f0 = abs(charge_pair_force(q0, radius, length, gap, sub, r_soft))
    if f0 <= 0:
        return 0.0
    return q0 * np.sqrt(abs(F_target) / f0)


def make_spec(dsg, mag, gaps_mm=(0.05, 0.1, 0.3, 0.6, 1.0, 2.0, 4.0),
              fidelity="screen"):
    """Build an EPMSpec, calibrating the charge against the FEM at each gap.

    The attracting and repelling states carry DIFFERENT charges, because the
    magnet sits at a different operating point in each - that asymmetry is the
    whole point and must survive into the dynamics.  Calibrating across a range
    of separations additionally captures the fact that the magnets weaken as
    they are pulled apart, which a single-point calibration cannot.
    """
    from framework import Design, stage1_magnetics

    R = dsg.d_mag / 2
    sub = disc_points(R)
    spec = EPMSpec(area=np.pi * R**2, radius=R, length=dsg.l_mag, sub=sub,
                   q_attract=0.0, q_repel=0.0)
    rs = _soft_radius(spec)

    gaps, qa, qr, fa, fr = [], [], [], [], []
    for g_mm in gaps_mm:
        g = g_mm * 1e-3
        m = (mag if abs(g - dsg.gap) < 1e-9 else
             stage1_magnetics(Design(**{**dsg.as_row(), "gap": g}),
                              fidelity=fidelity))
        gaps.append(g)
        qa.append(calibrate_charge(m["F_attract"], R, dsg.l_mag, g, sub, rs))
        qr.append(calibrate_charge(m["F_repel"], R, dsg.l_mag, g, sub, rs))
        fa.append(m["F_attract"])
        fr.append(m["F_repel"])

    spec.gaps = np.array(gaps)
    spec.q_att_of_gap = np.array(qa)
    spec.q_rep_of_gap = np.array(qr)
    spec.f_att_of_gap = np.array(fa)
    spec.f_rep_of_gap = np.array(fr)
    spec.q_attract = float(np.interp(dsg.gap, spec.gaps, spec.q_att_of_gap))
    spec.q_repel = float(np.interp(dsg.gap, spec.gaps, spec.q_rep_of_gap))
    return spec


# --------------------------------------------------------------------------
def face_charges(pos, quat, mod, spec, states, q_total):
    """World positions and signs of every pole sub-charge on one module.

    ``states`` is one entry per face: +1 attract-polarity, -1 reversed, 0 off.
    Each pole face carries ``q_total`` spread over ``spec.sub`` sample points.
    """
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    nsub = len(spec.sub)
    pts, qs = [], []
    for k in range(mod.n_faces):
        s = states[k]
        if s == 0:
            continue
        n_world = R @ mod.normals[k]
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tmp, n_world))) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(n_world, tmp)
        nrm = np.linalg.norm(e1)
        if nrm < 1e-12:                       # degenerate basis, pick any
            e1 = np.array([0.0, 1.0, 0.0])
            nrm = 1.0
        e1 = e1 / nrm
        e2 = np.cross(n_world, e1)

        front_c = pos + n_world * mod.r_face
        back_c = front_c - n_world * spec.length
        q = q_total * s / nsub
        for (u, v) in spec.sub:
            off = e1 * u + e2 * v
            pts.append(front_c + off)
            qs.append(+q)
            pts.append(back_c + off)
            qs.append(-q)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.array(pts), np.array(qs)


def magnetic_wrenches(posA, quatA, stA, posB, quatB, stB, mod, spec,
                      repel=None):
    """Net force and torque on module B from module A.

    Hybrid model.  The charge-disc sum supplies the DIRECTIONAL structure -
    how force and torque vary with relative orientation and lateral offset -
    while the MAGNITUDE is taken from the Stage 1 FEM at the current
    separation.

    The reason for splitting them: at the working gap the pole faces are a
    fraction of a millimetre apart but several millimetres across, so a
    tractable discretisation of the face is coarse compared with the gap, and
    the raw sum becomes sensitive to how the sample points on the two faces
    happen to line up.  Rescaling to the FEM removes that sensitivity and makes
    the aligned case exact by construction, which is the case that governs
    latching and holding.  The angular dependence, which the FEM cannot give
    because it is axisymmetric, is left to the geometric model.

    ``repel`` selects which operating point the magnets are on.  If left None
    it is inferred from the commanded states: two faces showing the same sign
    present the same pole to each other and are therefore repelling, where the
    magnets partly demagnetise one another and carry a smaller charge.
    """
    if repel is None:
        sa = next((s for s in stA if s), 0)
        sb = next((s for s in stB if s), 0)
        repel = (sa * sb) > 0

    dvec = np.asarray(posB) - np.asarray(posA)
    sep = max(np.linalg.norm(dvec) - 2.0 * mod.r_face, 0.0)
    q = spec.charge(sep, repel=repel)

    F, T = _raw_wrench(posA, quatA, stA, posB, quatB, stB, mod, spec, q)

    # reference: same separation, both modules aligned along the pair axis
    axis = dvec / max(np.linalg.norm(dvec), 1e-12)
    ref_pos = np.asarray(posA) + axis * (2.0 * mod.r_face + sep)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    Fr, _ = _raw_wrench(np.asarray(posA), ident, stA, ref_pos, ident, stB,
                        mod, spec, q)
    mag_ref = np.linalg.norm(Fr)
    if mag_ref <= 0:
        return F, T
    F_fem = spec.force(sep, repel=repel)
    scale = F_fem / mag_ref
    return F * scale, T * scale


def _raw_wrench(posA, quatA, stA, posB, quatB, stB, mod, spec, q):
    r_soft = _soft_radius(spec)
    pa, qa = face_charges(posA, quatA, mod, spec, stA, q)
    pb, qb = face_charges(posB, quatB, mod, spec, stB, q)
    if len(pa) == 0 or len(pb) == 0:
        return np.zeros(3), np.zeros(3)

    d = pb[None, :, :] - pa[:, None, :]                  # A -> B
    r = np.maximum(np.linalg.norm(d, axis=-1), r_soft)
    coef = (qa[:, None] * qb[None, :]) / (4 * np.pi * MU0 * r**3)
    fvec = coef[:, :, None] * d                          # force on B
    F = fvec.sum(axis=(0, 1))

    lever = pb - np.asarray(posB)
    per_b = fvec.sum(axis=0)
    T = np.cross(lever, per_b).sum(axis=0)
    return F, T


# --------------------------------------------------------------------------
SCENE = """<mujoco model="magnobots">
  <option timestep="0.0002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom solref="0.002 1" solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"
          friction="{mu} 0.02 0.001"/>
{bodies}
  </worldbody>
</mujoco>
"""


def two_module_scene(mod, mu=0.9, gap=0.1e-3):
    """Two modules side by side, mating on their +x / -x faces.

    The collision shape is a sphere of radius r_face rather than a box.  The
    module is the intersection of three n-gon rings, so it is much closer to a
    sphere than to a cube, and for n = 8 the inscribed and circumscribed radii
    differ by only 8 per cent.  A sphere also rolls correctly, which a box
    would not, and rolling is the locomotion mode being studied.
    """
    r = mod.r_face
    d = np.diag(mod.inertia)
    body = """    <body name="{name}" pos="{x:.5f} 0 {z:.5f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{m:.6f}" diaginertia="{i0:.9f} {i1:.9f} {i2:.9f}"/>
      <geom type="sphere" size="{r:.5f}" rgba="{col}"
            friction="{mu} 0.02 0.001"/>
    </body>"""
    bodies = "\n".join([
        body.format(name="A", x=0.0, z=r, m=mod.mass, i0=d[0], i1=d[1],
                    i2=d[2], r=r, col="0.35 0.45 0.75 1", mu=mu),
        body.format(name="B", x=2 * r + gap, z=r, m=mod.mass, i0=d[0],
                    i1=d[1], i2=d[2], r=r, col="0.80 0.45 0.30 1", mu=mu),
    ])
    return SCENE.format(bodies=bodies, mu=mu)


def run_scenario(mod, spec, states_A, states_B, seconds=1.0, mu=0.9,
                 hold_A=True, record_every=50):
    """Simulate two modules with a fixed EPM state pattern.

    Returns a trace of B's pose so the caller can judge whether it held,
    separated, or rotated.
    """
    model = mujoco.MjModel.from_xml_string(two_module_scene(mod, mu))
    data = mujoco.MjData(model)
    bA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A")
    bB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B")
    qA = model.body_jntadr[bA]
    qB = model.body_jntadr[bB]

    n = int(seconds / model.opt.timestep)
    # populate xpos/xquat before the first wrench evaluation; without this the
    # first step sees both bodies at the origin and produces a singular force
    mujoco.mj_forward(model, data)
    trace = []
    for i in range(n):
        pA = data.xpos[bA].copy()
        pB = data.xpos[bB].copy()
        rA = data.xquat[bA].copy()
        rB = data.xquat[bB].copy()

        F, T = magnetic_wrenches(pA, rA, states_A, pB, rB, states_B, mod, spec)
        data.xfrc_applied[bB, :3] = F
        data.xfrc_applied[bB, 3:] = T
        data.xfrc_applied[bA, :3] = -F
        data.xfrc_applied[bA, 3:] = 0.0
        if hold_A:
            data.qvel[model.jnt_dofadr[qA]:model.jnt_dofadr[qA] + 6] = 0
            data.xfrc_applied[bA] = 0

        mujoco.mj_step(model, data)

        if i % record_every == 0:
            quat = data.xquat[bB]
            ang = 2 * np.degrees(np.arccos(np.clip(abs(quat[0]), 0, 1)))
            trace.append(dict(t=i * model.opt.timestep,
                              pos=data.xpos[bB].copy(),
                              sep=np.linalg.norm(data.xpos[bB] -
                                                 data.xpos[bA]) - 2 * mod.r_face,
                              angle=ang))
    return trace


