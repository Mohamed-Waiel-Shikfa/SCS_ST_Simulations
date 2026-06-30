"""
Magnetic Circuit Physics Engine
================================
Implements the analytical model from Marchese, Asada & Rus (ICRA 2012)
adapted for a symmetric 5-layer sandwich:  Top Magnet → Top Cover → Air Gap → Bottom Cover → Bottom Magnet

Key equations (paper references):
  (1) Ampere's Law:         2·Hm·tm + Hg·lg_eff = 0
  (2) Flux Conservation:    Bm·Am = Bg·Ag
  (3) Gap Constitutive:     Bg = μ₀·Hg
  (4) Gap Permeance:        P  = μ₀·Ag_eff / lg_eff
  (5) Load Line:            Bm = −(Ag_eff/Am)·(2μ₀·tm/lg_eff)·Hm
  (6) Force (Maxwell):      F  = Bg²·Ag / (2μ₀)
      Paper uses F=Ag·Bg²/μ₀ because the EPMA has TWO gaps in parallel;
      our single-gap sandwich uses the standard ½ factor.
"""

import numpy as np
from scipy.interpolate import interp1d

MU_0 = 4.0 * np.pi * 1e-7  # Permeability of free space  [H/m]


# ──────────────────────────────────────────────────────────────
#  Material B-H Data  (2nd-quadrant demagnetization curves)
# ──────────────────────────────────────────────────────────────

def _alnico5_curve():
    """Return (H_array, B_array) for AlNiCo 5 in the 2nd quadrant.
    H values are negative (A/m), B values positive (T).
    Data points reconstructed from standard AlNiCo 5 datasheets.
    Br ≈ 1.27 T, Hcb ≈ 50 800 A/m.
    """
    H = np.array([
        0, -4000, -8000, -12000, -16000, -20000, -24000,
        -28000, -32000, -36000, -40000, -44000, -48000, -50800
    ], dtype=float)
    B = np.array([
        1.27, 1.265, 1.255, 1.24, 1.22, 1.19, 1.14,
        1.06, 0.94, 0.78, 0.56, 0.32, 0.12, 0.0
    ], dtype=float)
    return H, B


def _ndfeb_n45_curve():
    """Return (H_array, B_array) for NdFeB N45 in the 2nd quadrant.
    Nearly linear.  Br ≈ 1.35 T, Hcb ≈ 1 035 000 A/m.
    """
    H = np.linspace(0, -1_035_000, 200)
    Br = 1.35
    Hcb = -1_035_000.0
    B = Br * (1.0 - H / Hcb)          # linear demagnetisation
    B = np.clip(B, 0, None)
    return H, B


MATERIALS = {
    "AlNiCo 5": {
        "Br": 1.27,
        "Hcb": 50_800,
        "curve_fn": _alnico5_curve,
        "color": "#e8a838",
        "mu_rec": None,  # computed from curve
    },
    "NdFeB N45": {
        "Br": 1.35,
        "Hcb": 1_035_000,
        "curve_fn": _ndfeb_n45_curve,
        "color": "#4a9eff",
        "mu_rec": 1.35 / 1_035_000,  # ≈ 1.04 μ₀
    },
}


class MagneticCircuitEngine:
    """Analytical magnetic circuit solver for the 5-layer sandwich."""

    def __init__(self):
        # Geometry  (all stored in SI – metres)
        self.L  = 50e-3          # magnet length  [m]
        self.W  = 25e-3          # magnet width   [m]
        self.tm = 10e-3          # magnet thickness [m]
        self.tc = 1e-3           # cover thickness  [m]
        self.g_min = 2e-3        # minimum gap      [m]
        self.alpha_deg = 0.0     # wedge half-opening angle [°]

        # Material toggles
        self.magnet_material = "NdFeB N45"   # or "AlNiCo 5"
        self.cover_is_steel  = False         # True → ferromagnetic cover
        self.is_attracting   = True          # True → attract, False → repel

        # Caches filled by solve()
        self._results = {}

    # ── public setters (from UI, values arrive in mm / degrees) ──

    def set_geometry(self, L_mm, W_mm, tm_mm, tc_mm, g_min_mm, alpha_deg):
        self.L  = max(L_mm, 1) * 1e-3
        self.W  = max(W_mm, 1) * 1e-3
        self.tm = max(tm_mm, 0.5) * 1e-3
        self.tc = max(tc_mm, 0) * 1e-3
        self.g_min = max(g_min_mm, 0.1) * 1e-3
        self.alpha_deg = max(alpha_deg, 0)

    # ── geometry helpers ──

    @property
    def alpha_rad(self):
        return np.radians(self.alpha_deg)

    @property
    def Am(self):
        """Magnet pole-face area [m²]."""
        return self.L * self.W

    def effective_gap(self):
        """Effective magnetic gap length [m].
        Plastic/air cover adds to the gap; steel cover does not."""
        g = self.g_min
        if not self.cover_is_steel:
            g += 2.0 * self.tc
        return g

    def effective_gap_with_wedge(self):
        """Average effective gap accounting for wedge angle.
        Gap varies linearly: g(x) = g_eff + 2·x·sin(α/2), x ∈ [0, W].
        We return the value at the midpoint for the load-line,
        but force is computed by integration over strips."""
        g_eff = self.effective_gap()
        if self.alpha_deg <= 0:
            return g_eff
        return g_eff + self.W * np.sin(self.alpha_rad / 2.0)

    def Ag_eff_fringing(self, lg):
        """Effective gap area including Roters fringing correction.
        Each edge gains ≈ lg in effective width/length."""
        return (self.L + lg) * (self.W + lg)

    # ── core solver ──

    def solve(self):
        mat  = MATERIALS[self.magnet_material]
        H_bh, B_bh = mat["curve_fn"]()
        bh_interp = interp1d(H_bh, B_bh, kind='cubic',
                             bounds_error=False, fill_value=(mat["Br"], 0))

        lg_eff   = self.effective_gap_with_wedge()
        Ag_eff   = self.Ag_eff_fringing(lg_eff)
        Am       = self.Am

        # Load-line slope:  Bm = slope · Hm  (slope < 0 in 2nd quadrant)
        # From eqs (1)–(3):  Bm = −(Ag_eff/Am) · (2μ₀ tm / lg_eff) · Hm
        slope = -(Ag_eff / Am) * (2.0 * MU_0 * self.tm / lg_eff)

        # Permeance coefficient  PC = |slope| / μ₀
        PC = abs(slope) / MU_0

        # ── Operating point: intersection of load line with B-H curve ──
        # Load line: BL(H) = slope * H   (passes through origin)
        H_scan = np.linspace(0, H_bh[-1], 5000)
        BL = slope * H_scan
        B_curve = bh_interp(H_scan)

        diff = BL - B_curve
        # Find zero crossing
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            # Linear interpolation for precision
            frac = -diff[idx] / (diff[idx + 1] - diff[idx])
            Hm_op = H_scan[idx] + frac * (H_scan[idx + 1] - H_scan[idx])
            Bm_op = bh_interp(Hm_op)
        else:
            Hm_op = H_scan[0]
            Bm_op = mat["Br"]

        # ── Gap field from operating point ──
        # Eq (1):  Hg = −2·Hm·tm / lg_eff
        Hg = -2.0 * Hm_op * self.tm / lg_eff
        # Eq (3):  Bg = μ₀·Hg
        Bg = MU_0 * Hg

        # ── Force calculation with wedge integration ──
        if self.alpha_deg <= 1e-6:
            # Uniform gap: simple Maxwell stress
            F_total = Bg ** 2 * self.Am / (2.0 * MU_0)
        else:
            # Wedge: need strip integration.  Store Hm_op first so
            # _integrate_force_wedge can use it.
            self._Hm_op_cache = Hm_op
            F_total = self._integrate_force_wedge()

        if not self.is_attracting:
            F_total = abs(F_total)   # magnitude only; direction implicit

        # ── Permeance ──
        P_gap = MU_0 * Ag_eff / lg_eff

        # ── Store all results ──
        self._results = {
            "Hm_op":   float(Hm_op),
            "Bm_op":   float(Bm_op),
            "Hg":      float(Hg),
            "Bg":      float(Bg),
            "F_total": float(abs(F_total)),
            "F_top":   float(abs(F_total)),      # symmetric
            "F_bot":   float(abs(F_total)),
            "F_kgf":   float(abs(F_total) / 9.80665),
            "PC":      float(PC),
            "lg_eff":  float(lg_eff),
            "Ag_eff":  float(Ag_eff),
            "Am":      float(Am),
            "P_gap":   float(P_gap),
            "slope":   float(slope),
            # Curve data for plotting
            "H_bh":    H_bh,
            "B_bh":    B_bh,
            "H_loadline": np.array([0, H_bh[-1]]),
            "B_loadline": slope * np.array([0, H_bh[-1]]),
        }
        self._results["warnings"] = self._generate_warnings()
        return self._results

    def _integrate_force_wedge(self):
        """Integrate Maxwell stress over the face for wedge configurations.
        Divides the magnet width into strips, solves the operating point
        for each strip's local gap, and integrates the force density."""
        g_base = self.effective_gap()
        mat = MATERIALS[self.magnet_material]
        H_bh, B_bh = mat["curve_fn"]()
        bh_interp = interp1d(H_bh, B_bh, kind='cubic',
                             bounds_error=False, fill_value=(mat["Br"], 0))
        Am = self.Am

        N = 200
        x = np.linspace(0, self.W, N)
        g_x = g_base + 2.0 * x * np.sin(self.alpha_rad / 2.0)

        force_density = np.zeros(N)
        for i in range(N):
            lg_local = g_x[i]
            Ag_local = self.Ag_eff_fringing(lg_local)
            slope_local = -(Ag_local / Am) * (2.0 * MU_0 * self.tm / lg_local)
            H_scan = np.linspace(0, H_bh[-1], 2000)
            BL = slope_local * H_scan
            B_c = bh_interp(H_scan)
            diff = BL - B_c
            sc = np.where(np.diff(np.sign(diff)))[0]
            if len(sc) > 0:
                idx = sc[0]
                frac = -diff[idx] / (diff[idx + 1] - diff[idx])
                Hm_local = H_scan[idx] + frac * (H_scan[idx + 1] - H_scan[idx])
            else:
                Hm_local = 0
            Hg_local = -2.0 * Hm_local * self.tm / lg_local
            Bg_local = MU_0 * Hg_local
            force_density[i] = Bg_local ** 2 / (2.0 * MU_0)

        # Integrate: ∫ f(x) dx over width, then multiply by length L
        return float(np.trapz(force_density, x) * self.L)

    def _generate_warnings(self):
        warnings = []
        r = self._results
        mat = MATERIALS[self.magnet_material]

        # Check if operating near knee (AlNiCo risk)
        if self.magnet_material == "AlNiCo 5":
            if abs(r["Hm_op"]) > 0.6 * mat["Hcb"]:
                warnings.append(
                    "⚠ Operating point is near the AlNiCo knee. "
                    "Risk of irreversible demagnetization!"
                )
            warnings.append(
                "ℹ AlNiCo 5 has low coercivity — the circuit model "
                "captures B-H demagnetization that magpylib ignores. "
                "Analytical force may be more realistic for AlNiCo."
            )

        # Check permeance coefficient
        if r["PC"] < 1.0:
            warnings.append(
                "⚠ Permeance coefficient PC < 1. The magnet is "
                "under-loaded — consider reducing the gap."
            )

        # Repelling mode warning
        if not self.is_attracting:
            warnings.append(
                "⚠ Repelling mode: circuit model uses attracting-mode "
                "magnitude (symmetric approximation). magpylib result "
                "is more reliable for repulsion."
            )
            if self.magnet_material == "AlNiCo 5":
                warnings.append(
                    "⚠ Mutual demagnetization risk! Opposing field "
                    "can permanently demagnetize AlNiCo."
                )

        # Very large gap
        if r["lg_eff"] > 2 * self.tm:
            warnings.append(
                "⚠ Effective gap exceeds 2× magnet thickness. "
                "Significant flux leakage expected; model accuracy degrades."
            )

        # Steel cover note
        if self.cover_is_steel:
            warnings.append(
                "ℹ Steel covers act as flux keepers — the circuit model "
                "benefits but magpylib does not model soft iron, so its "
                "result is unchanged by this toggle."
            )

        # Model comparison note
        if self.magnet_material == "NdFeB N45":
            warnings.append(
                "ℹ Circuit model assumes no leakage (upper bound). "
                "magpylib uses exact cuboid fields (more realistic for "
                "open circuits without steel keepers)."
            )

        # Sticking risk
        if self.is_attracting and r["F_total"] > 50:
            warnings.append(
                f"⚠ Clamping force ({r['F_total']:.0f} N / "
                f"{r['F_kgf']:.1f} kgf) may make manual separation difficult."
            )

        if not warnings:
            warnings.append("✓ No warnings. Operating conditions are nominal.")

        return warnings

    def get_field_map(self, nx=80, ny=80, extent_mm=None):
        """Compute a 2D magnetic field map (Bx, By, |B|) over a grid.
        Uses magpylib for accurate cuboid-magnet fields when available,
        otherwise falls back to a simplified analytical model.
        """
        r = self._results
        if not r:
            self.solve()
            r = self._results

        half_gap = self.g_min / 2.0
        tm = self.tm
        tc = self.tc
        W  = self.W
        L  = self.L
        mat = MATERIALS[self.magnet_material]
        Br = mat["Br"]

        total_h = 2 * (tm + tc + half_gap)
        margin = total_h * 0.35
        x_range = (-margin, W + margin)
        y_range = (-(total_h / 2 + margin), total_h / 2 + margin)

        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)

        try:
            import magpylib as magpy
            return self._field_map_magpylib(
                X, Y, magpy, Br, half_gap, tm, tc, W, L)
        except ImportError:
            return self._field_map_analytical(
                X, Y, Br, half_gap, tm, tc, W, r)

    def _field_map_magpylib(self, X, Y, magpy, Br, half_gap, tm, tc, W, L):
        """Use magpylib cuboid sources for the 2D field map (xz-plane)."""
        z_top = half_gap + tc + tm / 2.0
        z_bot = -(half_gap + tc + tm / 2.0)
        pol_top = (0, 0, -Br)
        pol_bot_z = -Br if self.is_attracting else Br
        pol_bot = (0, 0, pol_bot_z)

        # magpylib uses (x,y,z).  Our 2D view is the xz-plane (y=0).
        # Map our X → magpylib x (centred), Y → magpylib z.
        m_top = magpy.magnet.Cuboid(
            polarization=pol_top,
            dimension=(L, W, tm),
            position=(0, 0, z_top),
        )
        m_bot = magpy.magnet.Cuboid(
            polarization=pol_bot,
            dimension=(L, W, tm),
            position=(0, 0, z_bot),
        )
        coll = magpy.Collection(m_top, m_bot)

        # Build observer grid: (x_obs=0, y_obs = our_X - W/2, z_obs = our_Y)
        pts = np.column_stack([
            np.zeros(X.size),
            X.ravel() - W / 2,
            Y.ravel(),
        ])
        B_vec = coll.getB(pts)          # shape (N, 3)
        Bx_flat = B_vec[:, 1]           # our "horizontal" is magpylib y
        By_flat = B_vec[:, 2]           # our "vertical" is magpylib z
        Bx = Bx_flat.reshape(X.shape)
        By = By_flat.reshape(X.shape)
        B_mag = np.sqrt(Bx**2 + By**2)

        return X * 1e3, Y * 1e3, Bx, By, B_mag

    def _field_map_analytical(self, X, Y, Br, half_gap, tm, tc, W, r):
        """Simplified exponential-decay model (fallback)."""
        Bg = abs(r["Bg"])
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)

        y_top_face = half_gap + tc
        y_bot_face = -(half_gap + tc)

        for iy in range(X.shape[0]):
            for ix in range(X.shape[1]):
                px, py = X[iy, ix], Y[iy, ix]
                in_width = 0 <= px <= W
                d_top = abs(py - y_top_face)
                d_bot = abs(py + y_bot_face)

                if in_width and y_bot_face <= py <= y_top_face:
                    By[iy, ix] = -Bg
                else:
                    scale = tm * 0.6
                    decay_t = Bg * np.exp(-d_top / scale)
                    decay_b = Bg * np.exp(-d_bot / scale)
                    By[iy, ix] = -(decay_t + decay_b) * (
                        1.0 if in_width else 0.3)

        B_mag = np.sqrt(Bx**2 + By**2)
        return X * 1e3, Y * 1e3, Bx, By, B_mag
