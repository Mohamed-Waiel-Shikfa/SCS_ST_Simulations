"""
Library-based Engine  (magpylib)
================================
Uses magpylib's analytical Cuboid model to compute the B-field and
derive clamping force independently, for cross-validation against
the circuit model in physics_engine.py.

magpylib uses the exact magnetostatic solution for uniformly
magnetised cuboids (Caciagli et al., J. Magn. Magn. Mater. 2018).
"""

import numpy as np

MU_0 = 4.0 * np.pi * 1e-7

try:
    import magpylib as magpy
    MAGPYLIB_AVAILABLE = True
except ImportError:
    MAGPYLIB_AVAILABLE = False


class MagpylibEngine:
    """Compute clamping force using magpylib's analytical cuboid fields."""

    def __init__(self):
        self.available = MAGPYLIB_AVAILABLE

    def compute(self, L, W, tm, tc, g_min, alpha_deg,
                magnet_material, cover_is_steel, is_attracting):
        """
        Parameters mirror the analytical engine (all in SI metres / degrees).
        Returns a dict with B_gap, F_total, F_kgf.
        """
        if not self.available:
            return {"error": "magpylib not installed",
                    "B_gap": 0, "F_total": 0, "F_kgf": 0}

        # Material parameters
        mat_props = {
            "AlNiCo 5":  {"Br": 1.27},
            "NdFeB N45": {"Br": 1.35},
        }
        Br = mat_props[magnet_material]["Br"]

        half_gap = g_min / 2.0
        alpha_rad = np.radians(alpha_deg)

        # Cover offset from magnet face to gap edge
        cover_offset = tc if not cover_is_steel else 0
        # For steel covers, the cover is ferromagnetic and part of the
        # magnetic circuit (negligible reluctance). For the magpylib
        # model we still place the magnet at the physical position but
        # the effective gap is just g_min.

        # ── Build magnet sources ──
        # Magnet dimensions: (length_x, width_y, thickness_z) in metres
        dim = (L, W, tm)

        # Polarisation direction: along z (vertical)
        # Top magnet: N-pole facing down (towards gap)
        #   → internal M points down → J = (0, 0, -Br)
        # Bottom magnet for ATTRACTING: S-pole facing up (towards gap)
        #   → same magnetisation direction → J = (0, 0, -Br)
        # Bottom magnet for REPELLING: N-pole facing up
        #   → opposite magnetisation → J = (0, 0, +Br)
        pol_top = (0, 0, -Br)
        pol_bot_z = -Br if is_attracting else Br
        pol_bot = (0, 0, pol_bot_z)

        # Positions (centre of each cuboid)
        # Top magnet centre at z = half_gap + tc + tm/2
        z_top = half_gap + tc + tm / 2.0
        z_bot = -(half_gap + tc + tm / 2.0)

        m_top = magpy.magnet.Cuboid(
            polarization=pol_top,
            dimension=dim,
            position=(0, 0, z_top),
        )
        m_bot = magpy.magnet.Cuboid(
            polarization=pol_bot,
            dimension=dim,
            position=(0, 0, z_bot),
        )

        # Apply wedge rotation (about x-axis through the inner corner)
        if alpha_deg > 0.1:
            from scipy.spatial.transform import Rotation as R
            # Rotate top magnet by +α/2 about x-axis, pivot at gap face
            pivot_top = np.array([0, 0, half_gap])
            r_top = R.from_euler('x', alpha_deg / 2, degrees=True)
            pos_rel = np.array(m_top.position) - pivot_top
            m_top.position = tuple(r_top.apply(pos_rel) + pivot_top)
            m_top.orientation = r_top

            pivot_bot = np.array([0, 0, -half_gap])
            r_bot = R.from_euler('x', -alpha_deg / 2, degrees=True)
            pos_rel = np.array(m_bot.position) - pivot_bot
            m_bot.position = tuple(r_bot.apply(pos_rel) + pivot_bot)
            m_bot.orientation = r_bot

        collection = magpy.Collection(m_top, m_bot)

        # ── Compute B-field at gap centre ──
        # Sample at a small offset from exact centre to avoid symmetry zeros
        # and average both sides for accuracy
        dz = g_min * 0.05
        B_upper = collection.getB((0, 0, dz))
        B_lower = collection.getB((0, 0, -dz))
        B_center = (B_upper + B_lower) / 2.0
        B_gap_z = abs(B_center[2])

        # ── Compute force via Maxwell stress tensor integration ──
        # Integrate Bz² / (2μ₀) over a plane in the gap (z = 0)
        Nx, Ny = 40, 40
        xs = np.linspace(-L / 2, L / 2, Nx)
        ys = np.linspace(-W / 2, W / 2, Ny)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        grid = np.array(
            [(x, y, 0) for x in xs for y in ys]
        )
        B_grid = collection.getB(grid)
        Bz = B_grid[:, 2]
        # Maxwell stress: σ_zz = Bz² / (2μ₀)  − (Bx²+By²) / (2μ₀)
        # For force normal to the surface, dominant term is Bz²
        Bx = B_grid[:, 0]
        By_arr = B_grid[:, 1]
        sigma_zz = (Bz**2 - Bx**2 - By_arr**2) / (2.0 * MU_0)
        F_total = np.sum(sigma_zz) * dx * dy
        F_total = abs(F_total)

        return {
            "B_gap":   float(B_gap_z),
            "F_total": float(F_total),
            "F_kgf":   float(F_total / 9.80665),
            "B_center_vec": B_center.tolist(),
        }
