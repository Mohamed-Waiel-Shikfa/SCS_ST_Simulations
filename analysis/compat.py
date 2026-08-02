"""Version compatibility shims.

These exist so the pipeline runs unchanged on a fresh machine - a CI runner or
a cloud agent - where the pinned versions in requirements.txt may not have been
honoured.  Silently crashing on the first Stage 2 call because a dependency
renamed a function is a bad way to lose an hour of compute.
"""

from __future__ import annotations

import numpy as np

# numpy removed ``trapz`` in 2.0 and renamed it ``trapezoid``; 1.x has only
# ``trapz``.  Nothing in this project needs anything else from the rename.
if hasattr(np, "trapezoid"):
    trapezoid = np.trapezoid
else:                                       # numpy < 2.0
    trapezoid = np.trapz


def check_environment(verbose=True):
    """Report the versions actually in use, and flag known-bad combinations."""
    import scipy
    import skfem

    info = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-fem": skfem.__version__,
    }
    try:
        import mujoco
        info["mujoco"] = mujoco.__version__
    except ImportError:
        info["mujoco"] = "not installed"

    problems = []
    if int(skfem.__version__.split(".")[0]) < 8:
        problems.append(
            "scikit-fem < 8: the condense/solve signature used by "
            "axisym_fem differs and the FEM will not assemble")
    if info["mujoco"] == "not installed":
        problems.append("mujoco missing: Stage 5 dynamics will not run "
                        "(the rest of the pipeline is unaffected)")

    if verbose:
        print("environment: " + ", ".join(f"{k} {v}" for k, v in info.items()))
        for p in problems:
            print(f"  WARNING: {p}")
    return info, problems


if __name__ == "__main__":
    check_environment()
