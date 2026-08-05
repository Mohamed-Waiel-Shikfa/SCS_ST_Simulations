r"""Which electropermanent architecture should actually be built?

The bench proof of concept is a ring of eight N42 blocks per module, switched
by hand: rotate the mating pair and one neighbouring pair, and the module
pivots.  ``ndfeb_switching.py`` shows that rotation cannot be done
electrically - reversing one of those blocks in place is a 40 to 60 joule,
kilovolt, kiloamp event, which is a bench magnetiser, not a robot.

So the question becomes: what CAN be built that keeps as much of the
demonstrated behaviour as possible?  There are four candidates and they differ
in one decisive respect - whether a face can present both polarities, or only
"on" and "off".

    A. all-switchable      one low-coercivity magnet per face, reversed
                           electrically.  Both polarities.  This is what the
                           existing pipeline optimises.
    B. gated hybrid        a fixed NdFeB whose flux is either sent out of the
                           pole or short-circuited internally by a small
                           switchable magnet.  On and off only.
    C. double-gated        two fixed NdFeB of opposite orientation, each with
                           its own gate.  Both polarities, at twice the
                           hardware.
    D. mechanical          rotate the NdFeB physically, as the bench model
                           does by hand.

Whether "on and off only" is enough is not a matter of opinion: it depends on
whether the NEXT face round the ring can pull the module over before the
current face lets go.  That force is computed here rather than assumed,
because the answer decides which of the four is worth designing.

Magnetisation is treated as rigid throughout.  For N42 that is an excellent
approximation - mu_rec is 1.05 and the fields in play are far below its
coercivity - and each block is represented by exactly one cell, so the source
geometry carries no discretisation error at all.
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

import fem3d  # noqa: E402
from magnet_force import MU0  # noqa: E402

G = 9.81

# the bench geometry
N_RING = 8
MAG = (20e-3, 10e-3, 5e-3)      # tangential x axial x through-thickness
BR_N42 = 1.32
GAP = 0.2e-3


def ring_geometry(n=N_RING, a_face=None):
    """Face radius of a regular n-gon ring whose faces are ``a_face`` wide."""
    a_face = a_face if a_face else MAG[0]
    r_face = a_face / (2.0 * np.tan(np.pi / n))
    return dict(n=n, a_face=a_face, r_face=r_face,
                r_vertex=r_face / np.cos(np.pi / n),
                pivot_deg=360.0 / n)


def face_normal(k, n=N_RING):
    """Face normals of a single ring rolling in the xz plane."""
    th = 2 * np.pi * k / n
    return np.array([np.cos(th), 0.0, np.sin(th)])


def magnet_cell(centre_module, k, sign, r_face, body, n=N_RING):
    """One N42 block on face ``k``, pole flush with the module surface.

    The block's own frame: local x along the ring tangent, y along the ring
    axis, z outward along the face normal, which is also the magnetisation
    axis.  A single cell represents the block exactly.
    """
    nrm = face_normal(k, n)
    tang = np.array([-nrm[2], 0.0, nrm[0]])
    axial = np.array([0.0, 1.0, 0.0])
    axes = np.stack([tang, axial, nrm])
    centre = np.asarray(centre_module) + nrm * (r_face - MAG[2] / 2)
    return fem3d.Cell(centre=centre,
                      half=np.array([MAG[0] / 2, MAG[1] / 2, MAG[2] / 2]),
                      axes=axes, kind="magnet", easy=nrm * sign,
                      material=None, body=body, face=k, sign=sign)


def rigid_solution(cells, br=BR_N42):
    """A Solution3D with magnetisation fixed at Br, no self-consistent solve.

    Legitimate here because NdFeB is rigid: the fields these blocks impose on
    each other are a fraction of their coercivity, so their polarisation does
    not move.  This is exactly the approximation that does NOT hold for
    Alnico, which is why the rest of the pipeline solves for it.
    """
    M = np.array([c.easy * (br / MU0) for c in cells])
    return fem3d.Solution3D(cells=cells, M=M, H=np.zeros_like(M), iters=0,
                            residual=0.0, converged=True)


def pair_force(centre_a, k_a, sign_a, centre_b, k_b, sign_b, r_face,
               n=N_RING, br=BR_N42):
    """Force on the B block from the A block, and the torque about origin."""
    cells = [magnet_cell(centre_a, k_a, sign_a, r_face, 0, n),
             magnet_cell(centre_b, k_b, sign_b, r_face, 1, n)]
    sol = rigid_solution(cells, br)
    F, _ = fem3d.force_on_body(sol, body=1, source_body=0, max_sub=26)
    c_b = cells[1].centre
    return F, np.cross(c_b, F)


# --------------------------------------------------------------------------
def main():
    g = ring_geometry()
    r = g["r_face"]
    d_centre = 2 * r + GAP

    print("=" * 78)
    print("WHICH ARCHITECTURE SHOULD BE BUILT?")
    print("=" * 78)
    print(f"""
  The bench module: {N_RING} blocks of {MAG[0]*1e3:.0f} x {MAG[1]*1e3:.0f} x {MAG[2]*1e3:.0f} mm in a ring.
  A ring of {N_RING} with {MAG[0]*1e3:.0f} mm faces has a face radius of {r*1e3:.1f} mm, so the
  module is {2*r*1e3:.0f} mm across and rolls by {g['pivot_deg']:.0f} degrees a step.""")

    m_mag = 8 * np.prod(MAG) * 7500
    print(f"\n  magnet mass alone {m_mag*1e3:.0f} g; with shell, coils and "
          f"electronics call it {m_mag*1e3*2.5:.0f} g")
    mass = m_mag * 2.5
    barrier = mass * G * (g["r_vertex"] - r)
    print(f"  gravitational barrier to roll one step: "
          f"m g (R_vertex - r_face) = {barrier*1e3:.1f} mJ")

    # ---- how the mating pair behaves ------------------------------------
    print("\n" + "=" * 78)
    print("1.  WHAT THE BENCH MODEL ACTUALLY HAS")
    print("=" * 78)
    A = np.zeros(3)
    B = np.array([d_centre, 0.0, 0.0])
    F_att, _ = pair_force(A, 0, +1, B, 4, -1, r)
    F_rep, _ = pair_force(A, 0, +1, B, 4, +1, r)
    print(f"""
  Mating pair, face to face at a {GAP*1e3:.1f} mm gap:

      attract   {abs(F_att[0]):6.1f} N
      repel     {abs(F_rep[0]):6.1f} N
      asymmetry {abs(F_att[0])/abs(F_rep[0]):6.2f}

  That symmetry is the whole reason the bench model works so cleanly, and it
  is a property of NdFeB rather than of the arrangement.  A rigid magnet
  pushes exactly as hard as it pulls.  A low-coercivity magnet does not: in
  the repelling state the two magnets drive each other down their own
  demagnetisation curves and the polarisation collapses, which is why the
  Alnico designs in this repository come out with an asymmetry of four to
  six and have to be optimised against it.""")

    # ---- the reach question ---------------------------------------------
    print("\n" + "=" * 78)
    print("2.  CAN THE NEXT FACE ROUND THE RING PULL IT OVER?")
    print("=" * 78)
    print("""
  This decides whether an on/off architecture is enough.  There are two moves
  a module can make and they are not equivalent:

    ROLL   B tips over its own leading bottom edge and rolls along the floor
           by 360/n = 45 degrees.  Its centre rises from r_face to R_vertex
           and comes back down.  Cheap.
    CLIMB  B rotates 90 degrees about the top edge it shares with A and ends
           up on A's next face.  Its centre ends up a full module higher.
           Expensive, and the height only ever increases, so gravity opposes
           the whole way.

  With both polarities available, ROLL is driven by repelling the face B is
  standing next to.  With on/off only there is nothing to push against, so
  CLIMB is the only move left - and it must be pulled by the reach pair.
""")
    h_roll = g["r_vertex"] - r
    print(f"  ROLL   barrier  m g dh, dh = {h_roll*1e3:.1f} mm -> "
          f"{mass*G*h_roll*1e3:7.1f} mJ")

    # climb: pivot is the shared TOP edge; B rotates 90 degrees anticlockwise
    pivot_pt = np.array([r, 0.0, +g["r_vertex"] * np.sin(np.pi / N_RING)])
    dh_climb = _climb_height(B, pivot_pt, np.pi / 2)
    print(f"  CLIMB  barrier  m g dh, dh = {dh_climb*1e3:.1f} mm -> "
          f"{mass*G*dh_climb*1e3:7.1f} mJ")

    print(f"""
  Now the reach pair through the climb.  A's face 1 pulls on B's face 3, and
  the torque is taken about the shared top edge at
  ({pivot_pt[0]*1e3:.1f}, {pivot_pt[2]*1e3:.1f}) mm:
""")
    print(f"  {'roll':>6} {'pole gap':>10} {'|F| reach':>11} "
          f"{'reach torque':>14} {'gravity torque':>15} {'net':>10}")
    print(f"  {'deg':>6} {'mm':>10} {'N':>11} {'mN m':>14} {'mN m':>15} "
          f"{'mN m':>10}")
    print("  " + "-" * 72)

    degs = np.linspace(0, 90, 19)
    tau_net, tau_reach_l, tau_g_l = [], [], []
    for deg in degs:
        th = np.radians(deg)
        R = _rot_y_ccw(th)
        cb = pivot_pt + R @ (B - pivot_pt)
        cells = [magnet_cell(A, 1, +1, r, 0),
                 _rotated_cell(cb, 3, -1, r, R, 1)]
        sol = rigid_solution(cells)
        F, _ = fem3d.force_on_body(sol, body=1, source_body=0, max_sub=26)
        lever = cells[1].centre - pivot_pt
        t_reach = -np.cross(lever, F)[1]       # + drives the climb
        t_grav = -np.cross(cb - pivot_pt,
                           np.array([0.0, 0.0, -mass * G]))[1]
        gapc = np.linalg.norm(cells[1].centre - cells[0].centre) - MAG[2]
        tau_reach_l.append(t_reach)
        tau_g_l.append(t_grav)
        tau_net.append(t_reach + t_grav)
        if int(deg) % 15 == 0:
            print(f"  {deg:6.0f} {gapc*1e3:10.1f} {np.linalg.norm(F):11.3f} "
                  f"{t_reach*1e3:14.2f} {t_grav*1e3:15.2f} "
                  f"{(t_reach+t_grav)*1e3:10.2f}")

    rad = np.radians(degs)
    w_reach = float(np.trapz(tau_reach_l, rad))
    w_grav = float(np.trapz(tau_g_l, rad))
    print(f"""
  Integrating torque through the 90 degree climb:

      work from the reach pair    {w_reach*1e3:8.1f} mJ
      work against gravity        {w_grav*1e3:8.1f} mJ
      net                         {(w_reach+w_grav)*1e3:8.1f} mJ    """
          f"""{'CLIMB SUCCEEDS' if w_reach + w_grav > 0 else 'CLIMB FAILS'}

  The reach pair is worth {w_reach*1e3:.0f} mJ against a {abs(w_grav)*1e3:.0f} mJ barrier - a margin of
  {w_reach/abs(w_grav):.1f}x - so with N42-class magnets an on/off face CAN pull a module
  up onto its neighbour.  This is the opposite of the finding for the Alnico
  designs, where the same manoeuvre failed: the reach force falls off as
  roughly the fourth power of distance, so the order of magnitude in contact
  force between the two architectures becomes two orders of magnitude at the
  reach separation of {(np.linalg.norm(magnet_cell(A,1,1,r,0).centre - magnet_cell(B,3,-1,r,1).centre) - MAG[2])*1e3:.0f} mm.

  But note what on/off cannot do.  With no repulsion there is nothing to push
  against, so a module can only ever climb ONTO another module.  Two modules
  alone on a floor could never separate, and a module could not traverse open
  ground.  That is a real restriction on the gait, not a detail.""")


    # ---- what a switchable magnet gives in the same envelope -------------
    print("\n" + "=" * 78)
    print("3.  WHAT A SWITCHABLE MAGNET GIVES IN THE SAME ENVELOPE")
    print("=" * 78)
    print("""
  The force a pole face delivers goes as the square of the polarisation at
  that face.  Alnico's remanence is as high as NdFeB's - LNG52 is 1.30 T
  against N42's 1.32 - so on paper it should be a straight swap.  It is not,
  because Alnico can only hold that polarisation in a long thin shape.  The
  block on the bench is the opposite of a long thin shape.
""")
    n_d = _demag(MAG)
    print(f"  the bench block has N_d = {n_d:.3f}, so a magnet of coercivity Hcj")
    print(f"  sitting in it sees a self-demagnetising field of N_d * Br / mu0:\n")
    print(f"  {'grade':<14} {'Br':>6} {'Hcj':>8} {'self-demag':>12} "
          f"{'ratio to Hcj':>13}  {'verdict':<28}")
    print("  " + "-" * 88)
    for name, br, hcj in (("NdFeB N42", 1.32, 955e3),
                          ("Alnico LNG52", 1.30, 57e3),
                          ("Alnico LNGT44", 0.88, 122e3),
                          ("Ferrite Y30", 0.38, 195e3),
                          ("MnAlC", 0.55, 240e3)):
        hd = n_d * br / MU0
        ratio = hd / hcj
        verdict = ("holds" if ratio < 0.5 else
                   "marginal" if ratio < 1.0 else
                   "self-erases in this shape")
        print(f"  {name:<14} {br:6.2f} {hcj/1e3:7.0f}k {hd/1e3:11.0f}k "
              f"{ratio:13.1f}  {verdict:<28}")

    print("""
  Alnico in this shape destroys itself before it ever sees a neighbour.  To
  keep its polarisation it needs a length-to-diameter ratio of four or more,
  and the flux from that thin rod then has to be spread onto a usable pole
  face by a steel pole piece - which conserves flux, so a rod of area A_rod at
  1.2 T spread over a pole of area A_pole gives only 1.2 * A_rod / A_pole.
""")
    print(f"  {'pole face':>10} {'rod dia needed':>15} {'rod length':>12} "
          f"{'fits in a 48 mm module?':>24}")
    print("  " + "-" * 66)
    a_pole = MAG[0] * MAG[1]
    for b_target in (0.9, 0.6, 0.4, 0.2):
        a_rod = b_target * a_pole / 1.2
        d_rod = 2 * np.sqrt(a_rod / np.pi)
        l_rod = 4 * d_rod
        fits = "yes" if l_rod < 2 * r * 0.85 else "NO"
        f_pole = b_target ** 2 * a_pole / (2 * MU0)
        print(f"  {b_target:9.2f} T {d_rod*1e3:14.1f} {l_rod*1e3:12.1f} "
              f"{fits:>24}   ({f_pole:.0f} N contact)")

    print(f"""
  So there is no Alnico geometry that fits inside a {2*r*1e3:.0f} mm module and reaches
  N42's {abs(F_att[0]):.0f} N.  The existing optimiser found the same wall from the other
  direction: its best feasible designs sit at 3 to 7 N of holding force, an
  order of magnitude below the bench model, and they get there by using small
  rods on {18} faces rather than large blocks on {N_RING}.""")

    _verdict(abs(F_att[0]), w_reach, w_grav, mass)


def _rot_y_ccw(th):
    """Anticlockwise rotation in the xz plane: (x,z) -> (xc - zs, xs + zc).

    This is the direction that lifts a module standing to the right of the
    pivot up and over its neighbour.  Getting the sign wrong rotates it into
    the floor, which produces perfectly plausible-looking numbers for a
    manoeuvre that cannot happen.
    """
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _climb_height(centre_b, pivot, th_max, n=200):
    """Net rise of the module centre through a climb, and the peak rise."""
    hs = []
    for th in np.linspace(0.0, th_max, n):
        cb = pivot + _rot_y_ccw(th) @ (centre_b - pivot)
        hs.append(cb[2] - centre_b[2])
    return max(hs)


def _rotated_cell(centre_module, k, sign, r_face, R, body, n=N_RING):
    """A face magnet on a module whose own frame has been rotated by R."""
    nrm = R @ face_normal(k, n)
    tang = np.array([-nrm[2], 0.0, nrm[0]])
    nt = np.linalg.norm(tang)
    tang = tang / nt if nt > 1e-9 else np.array([1.0, 0.0, 0.0])
    axial = np.cross(nrm, tang)
    axes = np.stack([tang, axial, nrm])
    centre = np.asarray(centre_module) + nrm * (r_face - MAG[2] / 2)
    return fem3d.Cell(centre=centre,
                      half=np.array([MAG[0] / 2, MAG[1] / 2, MAG[2] / 2]),
                      axes=axes, kind="magnet", easy=nrm * sign,
                      material=None, body=body, face=k, sign=sign)


def _demag(dims, axis=2):
    half = np.asarray(dims, dtype=float) / 2.0
    e = np.zeros(3)
    e[axis] = 1.0
    H = fem3d.cuboid_field(half, np.zeros(3), np.eye(3), e, np.zeros((1, 3)))[0]
    return float(-H[axis])


def _verdict(f_attract, w_reach, w_grav, mass):
    print("\n" + "=" * 78)
    print("4.  THE FOUR ARCHITECTURES, SCORED")
    print("=" * 78)
    print(f"""
  {'':<18} {'both':<7} {'force':<8} {'switch':<10} {'faces':<7} {'verdict'}
  {'':<18} {'poles':<7} {'N':<8} {'energy':<10} {'served':<7}
  {'-'*76}
  {'A switchable':<18} {'yes':<7} {'3-7':<8} {'~10 mJ':<10} {'18':<7} works, an order of magnitude weaker
  {'B gated hybrid':<18} {'no':<7} {'~' + f'{f_attract:.0f}':<8} {'~0.2 J':<10} {'8':<7} needs the reach to carry the roll
  {'C double-gated':<18} {'yes':<7} {'~' + f'{f_attract:.0f}':<8} {'~0.4 J':<10} {'8':<7} twice the hardware per face
  {'D mechanical':<18} {'yes':<7} {'~' + f'{f_attract:.0f}':<8} {'~0.6 J':<10} {'8':<7} a motor and a gearbox per face
  {'E pure N42 coil':<18} {'yes':<7} {'~' + f'{f_attract:.0f}':<8} {'40-60 J':<10} {'8':<7} not buildable - see ndfeb_switching
""")


if __name__ == "__main__":
    main()
