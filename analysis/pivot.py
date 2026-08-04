"""Does a module actually pivot onto its neighbour?

Stage 2 answers this statically: the magnetic work available over the roll is
compared with the gravitational barrier of lifting the centre of mass from the
face radius to the vertex radius.  That model assumes both driving faces
deliver their full force through the whole arc, which they cannot - the faces
separate as the module rotates and the force falls off steeply with gap.

This runs the manoeuvre in MuJoCo with the real polyhedron, so the module has
to get its centre of mass over a real edge with a real contact, and the
magnetic wrench is recomputed from the actual pose at every step.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dynamics import SCENE, pair_wrench  # noqa: E402
from module import ring_normals  # noqa: E402


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
    # deduplicate
    keep = []
    for p in P:
        if not any(np.linalg.norm(p - q) < 1e-9 for q in keep):
            keep.append(p)
    return np.array(keep)


def pivot_scene(mod, mu=0.9, gap=0.1e-3):
    """Module A fixed to the world, module B free, mating on A's +x face.

    A is a static body: the manoeuvre being tested is one module climbing onto
    a substrate, so the substrate must not recoil.
    """
    V = hull_vertices(mod.n_gon, mod.r_face)
    vtx = " ".join(f"{v:.6f}" for v in V.reshape(-1))
    d = np.diag(mod.inertia)
    r = mod.r_face
    asset = f'  <asset>\n    <mesh name="mod" vertex="{vtx}"/>\n  </asset>\n'
    bodies = f"""    <body name="A" pos="0 0 {r:.5f}">
      <geom type="mesh" mesh="mod" mass="{mod.mass:.6f}"
            rgba="0.35 0.45 0.75 1" friction="{mu} 0.02 0.001"/>
    </body>
    <body name="B" pos="{2*r+gap:.5f} 0 {r:.5f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mod.mass:.6f}"
                diaginertia="{d[0]:.9f} {d[1]:.9f} {d[2]:.9f}"/>
      <geom type="mesh" mesh="mod" rgba="0.80 0.45 0.30 1"
            friction="{mu} 0.02 0.001"/>
    </body>"""
    xml = SCENE.format(bodies=bodies, mu=mu)
    return xml.replace("  <worldbody>", asset + "  <worldbody>")


def face_index(mod, direction):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return int(np.argmax(mod.normals @ d))


def run_pivot(mod, spec, seconds=0.6, latch_time=0.05, mu=0.9,
              record_every=100, drive="repel"):
    """Latch, then command the pivot, and report what the module does.

    ``drive`` selects the actuation:

    * ``"repel"``  - the mating pair reverses.  The push acts at the centre
      height, and the floor contact at the leading bottom edge provides the
      pivot, so the module tips forward over that edge.
    * ``"reach"``  - the pair one step around the ring is energised to attract,
      pulling the module over.  This is the manoeuvre the static Stage 2 model
      implicitly assumes.
    * ``"none"``   - stay latched, as a control.

    Returns a trace with B's rotation about y (the pivot axis), its centre
    height and its horizontal position.
    """
    model = mujoco.MjModel.from_xml_string(pivot_scene(mod, mu))
    data = mujoco.MjData(model)
    bA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A")
    bB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B")

    c = np.cos(2 * np.pi / mod.n_gon)
    s = np.sin(2 * np.pi / mod.n_gon)
    ia = face_index(mod, [1, 0, 0])
    ib = face_index(mod, [-1, 0, 0])
    mate = [(ia, ib, "attract")]
    modes = {
        "repel": [(ia, ib, "repel")],
        "reach": [(face_index(mod, [c, 0, s]), ib, "attract"),
                  (ia, face_index(mod, [-c, 0, s]), "repel")],
        "none": mate,
    }
    active = modes[drive]

    n = int(seconds / model.opt.timestep)
    n_latch = int(latch_time / model.opt.timestep)
    mujoco.mj_forward(model, data)
    trace = []
    W_in = 0.0
    for i in range(n):
        pA, qA = data.xpos[bA].copy(), data.xquat[bA].copy()
        pB, qB = data.xpos[bB].copy(), data.xquat[bB].copy()
        pairs = mate if i < n_latch else active
        F, T = pair_wrench(pA, qA, pB, qB, mod, spec, pairs)
        data.xfrc_applied[bB, :3] = F
        data.xfrc_applied[bB, 3:] = T
        mujoco.mj_step(model, data)
        # work actually delivered by the magnetic wrench this step
        v = data.qvel[:3].copy()
        w = data.qvel[3:6].copy()
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, data.xquat[bB])
        W_in += float(F @ v + T @ (Rb.reshape(3, 3) @ w)) * model.opt.timestep
        if i % record_every == 0:
            q = data.xquat[bB]
            # signed rotation about y
            ang = np.degrees(2 * np.arctan2(q[2], q[0]))
            ke = 0.5 * mod.mass * float(v @ v) + \
                0.5 * float(w @ (np.diag(mod.inertia) * w))
            trace.append(dict(t=i * model.opt.timestep,
                              ang=ang,
                              z=float(data.xpos[bB][2]),
                              x=float(data.xpos[bB][0]),
                              ke=ke,
                              W_in=W_in,
                              pe=mod.mass * 9.81 * float(data.xpos[bB][2]),
                              F=float(np.linalg.norm(F)),
                              tau=float(T[1])))
    return trace


