"""Stage 4: every distinct rolling configuration, in MuJoCo.

A module has to move in more than one way, and gravity does not treat those
ways alike.  Four cases cover the distinct problems:

    horizontal          rolling along a floor.  One case is enough: every
                        horizontal direction is the same problem rotated, and
                        gravity is perpendicular to the travel throughout.

    vertical, bottom    climbing a wall starting from the base.  The module
                        has to lift its whole weight against gravity while
                        hanging off one face.  This is the hardest case and
                        the one that decides whether a structure can be built
                        upwards at all.

    vertical, side      traversing a wall sideways.  Gravity is perpendicular
                        to travel, as in the horizontal case, but the module
                        is hanging rather than resting, so the latch has to
                        carry the full weight for the whole manoeuvre instead
                        of the floor carrying it.

    vertical, top       coming over the top of a wall onto a ceiling, or down
                        the far side.  Gravity now HELPS the rotation and
                        hinders the latch: the risk is not failing to move
                        but failing to stop, so this case is scored on
                        whether the module overshoots past its target face.

Only run after switching has succeeded.  These are full dynamics runs and
they are the most expensive thing in the pipeline; there is no point
simulating the gait of a module whose coil cannot reverse its magnet.

The face-state rules come from ``face_states``: latching only on the six axis
faces, and a pivot driven by reversing the face-to-face pair together with the
trailing neighbour up to 90 degrees round.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dynamics import pair_wrench  # noqa: E402
from face_states import (active_pairs, face_index,  # noqa: E402
                         mate_partner, pivot_states)
from module import hull_vertices, pivot_angle  # noqa: E402

G = 9.81

# In every scene the joint is module A's +x face and gravity is -z.  The
# configurations then differ in the direction of TRAVEL relative to gravity,
# and in whether there is a floor carrying the weight:
#
#   name -> (travel direction, floor?, note)
CONFIGURATIONS = {
    "horizontal": (
        [0, 0, 1], True,
        "rolling over a neighbour on the ground; the floor carries the "
        "weight and gravity opposes only the lift over the edge"),
    "vertical_bottom": (
        [0, 0, 1], False,
        "climbing a wall from the base; nothing but the latch carries the "
        "weight and the roll must lift it"),
    "vertical_side": (
        [0, 1, 0], False,
        "traversing a wall sideways; gravity is across the travel, so the "
        "latch carries the weight throughout without helping or hindering "
        "the rotation"),
    "vertical_top": (
        [0, 0, -1], False,
        "coming back down a wall; gravity now assists the rotation and "
        "fights the latch, so the risk is overshooting rather than "
        "stalling"),
}


def _scene(mod, mu=0.9, gap=0.1e-3, with_floor=True):
    """Module A anchored, module B free, mated on A's +x face.

    A is static: the manoeuvre is one module climbing onto a substrate, so the
    substrate must not recoil.  The floor is present only for the horizontal
    case; on a wall the latch is the only thing holding the module up, which
    is the entire point of separating those cases.

    The two shells are placed in CONTACT, not ``gap`` apart.  The working air
    gap is a magnetic quantity - the distance between pole faces, which sit
    slightly inside the shell - and it does not mean the modules are floating.
    Separating them mechanically as well removed the contact, and with it the
    friction: the magnetic pull between two side-by-side modules acts along
    the joint, which is horizontal, so with no friction there was nothing at
    all resisting gravity and every wall case fell 2.7 metres.
    """
    V = hull_vertices(mod.n_gon, mod.r_face)
    vtx = " ".join(f"{v:.6f}" for v in V.reshape(-1))
    d = np.diag(mod.inertia)
    r = mod.r_face
    floor = (f'    <geom name="floor" type="plane" size="1 1 0.1" '
             f'rgba="0.85 0.85 0.87 1" friction="{mu} 0.02 0.001"/>\n'
             if with_floor else "")
    return f"""<mujoco model="magnobots">
  <option timestep="0.00005" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom solref="0.0005 1" solimp="0.98 0.999 0.0002"/>
  </default>
  <asset>
    <mesh name="mod" vertex="{vtx}"/>
  </asset>
  <worldbody>
    <light pos="0 0 1"/>
{floor}    <body name="A" pos="0 0 {r:.5f}">
      <geom type="mesh" mesh="mod" mass="{mod.mass:.6f}"
            rgba="0.35 0.45 0.75 1" friction="{mu} 0.02 0.001"/>
    </body>
    <body name="B" pos="{2*r - 2e-6:.6f} 0 {r:.5f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mod.mass:.6f}"
                diaginertia="{d[0]:.9f} {d[1]:.9f} {d[2]:.9f}"/>
      <geom type="mesh" mesh="mod" rgba="0.80 0.45 0.30 1"
            friction="{mu} 0.02 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass
class RollResult:
    name: str
    note: str
    drive: str
    target_deg: float
    peak_deg: float
    settled_deg: float
    steps: float
    completed: bool
    overshot: bool
    detached: bool
    max_sep: float = 0.0
    final_sep: float = 0.0
    t_complete: float = None
    trace: list = field(default_factory=list)

    def verdict(self):
        if self.detached:
            return "detached"
        if self.overshot:
            return "overshot"
        return "completed" if self.completed else "stalled"

    def summary(self):
        return (f"{self.name:<16} {self.drive:<14} {self.verdict():<10} "
                f"{self.settled_deg:6.1f} of {self.target_deg:.0f} deg, "
                f"gap {self.final_sep*1e3:6.2f} mm"
                + (f", {self.t_complete*1e3:.0f} ms" if self.t_complete
                   else ""))



def run_configuration(mod, spec, name, seconds=0.8, latch_time=0.05, mu=0.9,
                      record_every=40, drive="push_off"):
    """Latch, command the pivot, and report what the module does."""
    travel, with_floor, note = CONFIGURATIONS[name]

    model = mujoco.MjModel.from_xml_string(
        _scene(mod, mu, with_floor=with_floor))
    data = mujoco.MjData(model)
    bA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A")
    bB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B")

    ia = face_index(mod.normals, [1, 0, 0])
    ib = mate_partner(mod.normals, ia, [1, 0, 0])
    mate = [(ia, ib, "attract")]

    tv = np.asarray(travel, dtype=float)
    tv = tv - np.dot(tv, [1, 0, 0]) * np.array([1.0, 0.0, 0.0])
    if np.linalg.norm(tv) < 1e-9:
        tv = np.array([0.0, 0.0, 1.0])
    sa, sb, pairs = pivot_states(mod.normals, travel=tv, axis=[1, 0, 0],
                                 drive=drive)
    active = active_pairs(mod.normals, sa, sb, pairs)

    # The CATCH phase.  A controller does not hold the drive on after the step
    # is finished - it re-latches on the face the module has just landed on.
    # Leaving the push on for the rest of the run made every configuration
    # report "detached", because the module completed its 45 degree roll and
    # then kept being shoved.  The leading pair from the "reach" scheme is
    # exactly the pair that comes into contact at the end of the step.
    ca, cb, cpairs = pivot_states(mod.normals, travel=tv, axis=[1, 0, 0],
                                  drive="reach")
    catch = [(ka, kb, "attract") for ka, kb in cpairs[1:]] or mate

    n = int(seconds / model.opt.timestep)
    n_latch = int(latch_time / model.opt.timestep)
    mujoco.mj_forward(model, data)

    target = float(np.degrees(pivot_angle(mod.n_gon)))
    trace = []
    t_complete = None
    peak = 0.0
    max_sep = 0.0
    phase = "latch"
    for i in range(n):
        pA, qA = data.xpos[bA].copy(), data.xquat[bA].copy()
        pB, qB = data.xpos[bB].copy(), data.xquat[bB].copy()
        if i < n_latch:
            phase, pairs_now = "latch", mate
        elif t_complete is None:
            phase, pairs_now = "drive", active
        else:
            phase, pairs_now = "catch", catch
        F, T = pair_wrench(pA, qA, pB, qB, mod, spec, pairs_now)
        data.xfrc_applied[bB, :3] = F
        data.xfrc_applied[bB, 3:] = T
        mujoco.mj_step(model, data)

        q = data.xquat[bB]
        ang = abs(np.degrees(2 * np.arctan2(
            float(np.linalg.norm(q[1:])), float(q[0]))))
        if ang > 180:
            ang = 360.0 - ang
        peak = max(peak, ang)
        sep = float(np.linalg.norm(data.xpos[bB] - data.xpos[bA]) -
                    2 * mod.r_face)
        max_sep = max(max_sep, sep)
        if t_complete is None and ang >= target * 0.92:
            t_complete = i * model.opt.timestep
        if i % record_every == 0:
            trace.append(dict(
                t=i * model.opt.timestep, ang=ang, phase=phase,
                pos=[float(v) for v in data.xpos[bB]],
                quat=[float(v) for v in data.xquat[bB]],
                sep=sep,
                F=float(np.linalg.norm(F)), tau=float(np.linalg.norm(T))))

    settled = trace[-1]["ang"] if trace else 0.0
    sep = trace[-1]["sep"] if trace else 0.0
    return RollResult(
        name=name, note=note, drive=drive, target_deg=target, peak_deg=peak,
        settled_deg=settled, steps=settled / max(target, 1e-9),
        completed=bool(peak >= target * 0.9 and sep < 0.25 * mod.r_face),
        overshot=bool(peak > target * 1.6),
        detached=bool(sep > 0.25 * mod.r_face),
        max_sep=max_sep, final_sep=sep,
        t_complete=t_complete, trace=trace)


def run_all(mod, spec, seconds=0.8, which=None, drive="push_off", **kw):
    """Every distinct rolling configuration, under one drive scheme."""
    names = which or list(CONFIGURATIONS)
    return {n: run_configuration(mod, spec, n, seconds=seconds, drive=drive,
                                 **kw)
            for n in names}


def compare_drives(mod, spec, drives=("push_off", "trailing_only", "reach"),
                   **kw):
    """Every configuration under every drive scheme.

    Which scheme works is not obvious in advance and is not the same in every
    configuration, so the pipeline runs all of them rather than assuming one.
    """
    return {d: run_all(mod, spec, drive=d, **kw) for d in drives}


if __name__ == "__main__":
    import time

    from dynamics import make_spec
    from framework import Design, stage1_magnetics
    from module import build_module

    d = Design(material="LNGT44", d_mag=4.2e-3, l_mag=8.4e-3,
               r_face=19.4e-3, n_gon=8, t_steel=1.0e-3, r_clear=0.4e-3,
               wire_d=0.25e-3, n_layers=6, v_cap=120.0, c_cap=47e-6)
    print("=" * 74)
    print("ROLLING CONFIGURATIONS")
    print("=" * 74)
    mag = stage1_magnetics(d, fidelity="screen")
    mod = build_module(d)
    spec = make_spec(d, mag, fidelity="screen")
    print(f"\n  {mod.summary()}")
    print(f"  pivot target {np.degrees(pivot_angle(d.n_gon)):.1f} deg\n")
    t0 = time.time()
    for drive, results in compare_drives(mod, spec).items():
        for name, res in results.items():
            print(f"  {res.summary()}")
        print()
    print(f"  ({time.time()-t0:.0f} s for twelve runs)")
    print("""
  The horizontal case has a floor to pivot on.  The three vertical cases do
  not: the only thing holding the module to the wall is the latch, so a drive
  scheme that reverses the mating pair releases it.  That is a real result
  about the scheme rather than a limitation of the simulation, and it is why
  all three schemes are run instead of one being assumed.""")
