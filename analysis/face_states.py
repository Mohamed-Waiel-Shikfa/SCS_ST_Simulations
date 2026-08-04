"""Face-state rules: what each EPM is commanded to do, and when.

Two rules govern which faces are energised and how, and both are structural
rather than tuneable.

Latching
--------
Only the six AXIS faces may latch.  They are the faces shared by two of the
three rings, with normals along +-x, +-y, +-z, so two modules joined axis-face
to axis-face have parallel frames and the assembly stays on a cubic lattice.
The remaining 3n - 12 faces carry EPMs and are used to drive rolling, but a
module must never come to rest on one: a joint at an oblique angle would break
the lattice and no later move could recover it.

During a latch, every opposing pair presents opposite poles - both faces of
each pair are commanded to attract.

Pivoting
--------
To start a pivot in one direction, two pairs go to repel:

  * the pair that is directly face to face, which pushes the module off its
    seat;
  * the neighbouring pair on the far side, up to 90 degrees round from the
    direction of travel, which pushes the trailing edge up and over.

Everything else stays attracting, so the module is never fully released.  A
single repelling pair would push the module away bodily instead of rotating
it; it is the couple between the two that produces a pivot rather than a
shove.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from module import latch_faces, ring_normals  # noqa: E402

ATTRACT, REPEL, OFF = +1, -1, 0


def face_index(normals, direction):
    """Index of the face whose normal is closest to ``direction``."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return int(np.argmax(np.asarray(normals) @ d))


def is_latchable(normals, k):
    return k in latch_faces(normals)


def latch_states(normals, axis):
    """Commanded polarity of every face while latched along ``axis``.

    Returns ``(states_a, states_b)`` mapping face index to +1 or -1, using the
    convention that +1 presents a north pole outward.  The mating pair is
    given opposite signs so it attracts; every other modelled face keeps the
    module's default polarity.
    """
    ia = face_index(normals, axis)
    ib = face_index(normals, -np.asarray(axis, dtype=float))
    if not is_latchable(normals, ia):
        raise ValueError(
            f"face {ia} (normal {normals[ia]}) is not an axis face, so it "
            f"may not latch - only the six axis faces preserve the lattice")
    a = {k: +1 for k in range(len(normals))}
    b = {k: +1 for k in range(len(normals))}
    a[ia] = +1
    b[ib] = -1
    return a, b


def mate_partner(normals, k, axis):
    """The face on the neighbour that mates with face ``k`` on this module.

    Two modules joined along ``axis`` are mirror images across the joint
    plane, so the partner of a face with normal n is the REFLECTION of n
    through that plane, n - 2 (n . a) a - not -n.  For the mating faces
    themselves the two happen to coincide, which is why the error stayed
    hidden: it only shows up on the neighbouring pairs, where taking -n picks
    a face on the far side of the module that never comes near anything.
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    n = np.asarray(normals[k], dtype=float)
    return face_index(normals, n - 2.0 * np.dot(n, a) * a)


def pivot_states(normals, travel, axis, span_deg=90.0, drive="push_off"):
    """Commanded polarity while starting a pivot in direction ``travel``.

    ``axis`` is the direction from module A to module B, i.e. the latched
    joint.  ``travel`` is the direction the moving module is heading.

    ``drive`` selects the actuation scheme:

    * ``"push_off"`` - the specified scheme: the face-to-face pair reverses,
      and so does the neighbouring pair on the trailing side, up to
      ``span_deg`` round from the joint.  Both push, and the couple between
      them rotates the module over its leading edge.
    * ``"trailing_only"`` - only the trailing neighbour reverses; the mating
      pair stays attracting and acts as the hinge.  Included because
      ``push_off`` releases the module entirely, which matters when there is
      no floor underneath it.
    * ``"reach"`` - the mating pair reverses and the LEADING neighbour pair
      attracts, pulling the module over rather than pushing it.

    All three are simulated in ``rolling.py`` rather than one being assumed,
    because which of them works depends on whether gravity is helping.
    """
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    tv = np.asarray(travel, dtype=float)
    tv = tv - np.dot(tv, ax) * ax
    if np.linalg.norm(tv) < 1e-9:
        raise ValueError("travel direction must not be parallel to the joint")
    tv = tv / np.linalg.norm(tv)

    a, b = latch_states(normals, ax)
    ia = face_index(normals, ax)
    ib = mate_partner(normals, ia, ax)

    pairs = []
    if drive in ("push_off", "reach"):
        a[ia] = +1
        b[ib] = +1                       # like poles: repel
    else:
        a[ia] = +1
        b[ib] = -1                       # keep the latch: attract
    # The mating pair is ALWAYS in the returned list, whatever it is commanded
    # to do.  Leaving it out when it was not being reversed meant the latch
    # simply stopped being applied during the drive phase, and every wall case
    # fell off for a reason that had nothing to do with the physics.
    pairs.append((ia, ib))

    # neighbours of the joint, within span_deg, split by which side they are on
    cos_lim = np.cos(np.radians(span_deg))
    trail, lead = (None, -2.0), (None, -2.0)
    for k, nrm in enumerate(normals):
        if k == ia or float(np.dot(nrm, ax)) < cos_lim - 1e-9:
            continue
        d = float(np.dot(nrm, tv))
        if -d > trail[1]:
            trail = (k, -d)
        if d > lead[1]:
            lead = (k, d)

    if drive in ("push_off", "trailing_only") and trail[0] is not None \
            and trail[1] > 1e-6:
        ka = trail[0]
        kb = mate_partner(normals, ka, ax)
        a[ka] = +1
        b[kb] = +1                       # repel
        pairs.append((ka, kb))
    if drive == "reach" and lead[0] is not None and lead[1] > 1e-6:
        ka = lead[0]
        kb = mate_partner(normals, ka, ax)
        a[ka] = +1
        b[kb] = -1                       # attract
        pairs.append((ka, kb))

    return a, b, pairs


def active_pairs(normals, states_a, states_b, pairs):
    """Turn commanded polarities into (face_a, face_b, mode) triples."""
    out = []
    for ka, kb in pairs:
        same = states_a.get(ka, 0) * states_b.get(kb, 0) > 0
        out.append((ka, kb, "repel" if same else "attract"))
    return out


if __name__ == "__main__":
    print("=" * 74)
    print("FACE STATE RULES")
    print("=" * 74)
    for n in (8, 12, 16):
        N = ring_normals(n)
        lat = latch_faces(N)
        print(f"\n  n = {n}: {len(N)} faces, {len(lat)} latchable "
              f"(indices {lat})")
        for k in lat:
            print(f"      face {k:2d}  normal {np.round(N[k], 3)}")
        bad = [k for k in range(len(N)) if k not in lat][:3]
        print(f"      first non-latchable: {bad} (rolling only)")

        a, b, rep = pivot_states(N, travel=[0, 0, 1], axis=[1, 0, 0])
        print(f"      pivot along +z about the +x joint reverses "
              f"{len(rep)} pairs: {rep}")
        for ka, kb in rep:
            print(f"          A face {ka:2d} {np.round(N[ka],3)}  <->  "
                  f"B face {kb:2d} {np.round(N[kb],3)}")

    print("""
  Two pairs reverse, not one.  A single repelling pair pushes the module
  bodily away from its neighbour; it is the couple between the face-to-face
  pair and the trailing neighbour that turns that push into a rotation about
  the leading edge.""")
