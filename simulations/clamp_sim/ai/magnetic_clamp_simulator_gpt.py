#!/usr/bin/env python3
"""Magnetic sandwich clamping simulator.

Tkinter + Matplotlib GUI for:
- material demagnetization curve vs. load line
- operating point intersection
- clamping force estimates
- twist overlap reduction
- wedge-gap permeance and force integration
- repelling/attracting interaction warnings

This is an engineering model intended for exploration and visualization,
not a finite-element solver.
"""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle


MU0 = 4.0 * math.pi * 1e-7
STEEL_MUR = 2000.0
KGF_PER_NEWTON = 1.0 / 9.80665

MATERIALS: Dict[str, Dict[str, float]] = {
    "AlNiCo 5": {
        "Br": 1.20,
        "mu_d": 1.8218e-5,  # given by prompt
        "desc": "Low coercivity electropermanent material",
    },
    "NdFeB N45": {
        "Br": 1.35,
        "mu_d": MU0 * 1.05,  # prompt: B = Br + mu_r mu0 H
        "desc": "High coercivity rare-earth magnet",
    },
}

GAP_MEDIA = {
    "Plastic/Air": MU0,
    "Steel": MU0 * STEEL_MUR,
}


def fmt(x: float, ndigits: int = 6) -> str:
    if abs(x) >= 1e4 or (0 < abs(x) < 1e-3):
        return f"{x:.{ndigits}e}"
    return f"{x:.{ndigits}f}".rstrip("0").rstrip(".")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -------------------- Convex polygon overlap (twist) --------------------

def rectangle_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float) -> List[Tuple[float, float]]:
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    out = []
    for x, y in pts:
        out.append((cx + x*ca - y*sa, cy + x*sa + y*ca))
    return out


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _inside(p, a, b) -> bool:
    (x, y) = p
    (x1, y1) = a
    (x2, y2) = b
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0


def _intersection(s, e, cp1, cp2):
    x1, y1 = s
    x2, y2 = e
    x3, y3 = cp1
    x4, y4 = cp2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-15:
        return e
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / den
    return (px, py)


def convex_polygon_clip(subject_polygon: List[Tuple[float, float]], clip_polygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    output = subject_polygon[:]
    cp1 = clip_polygon[-1]
    for cp2 in clip_polygon:
        input_list = output[:]
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if _inside(e, cp1, cp2):
                if not _inside(s, cp1, cp2):
                    output.append(_intersection(s, e, cp1, cp2))
                output.append(e)
            elif _inside(s, cp1, cp2):
                output.append(_intersection(s, e, cp1, cp2))
            s = e
        cp1 = cp2
    return output


def overlap_area_rectangles(len_top, wid_top, len_bottom, wid_bottom, twist_deg) -> float:
    top = rectangle_polygon(0.0, 0.0, len_top, wid_top, twist_deg)
    bottom = rectangle_polygon(0.0, 0.0, len_bottom, wid_bottom, 0.0)
    inter = convex_polygon_clip(top, bottom)
    return polygon_area(inter)


@dataclass
class ModelResult:
    material: str
    gap_material: str
    polarity: str
    length_top: float
    width_top: float
    thickness_top: float
    length_bottom: float
    width_bottom: float
    thickness_bottom: float
    gap_length: float
    twist_angle_deg: float
    wedge_angle_deg: float

    area_top: float
    area_bottom: float
    area_magnet: float
    overlap_area: float
    active_gap_area: float
    effective_gap_length: float
    equivalent_gap_length: float
    gap_permeance: float
    gap_reluctance: float
    dR_dl0: float

    Br: float
    mu_d: float
    mu_gap: float
    load_line_k: float
    H_op: float
    Bm_op: float
    Bg_op: float

    force_attract_N: float
    force_signed_N: float
    force_factor: float
    force_top_N: float
    force_bottom_N: float
    force_kgf: float
    warnings: List[str]


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        self._window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        def _resize_window(event):
            canvas.itemconfigure(self._window, width=event.width)

        canvas.bind("<Configure>", _resize_window)


class MagneticClampApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Magnetic Sandwich Clamp Simulator")
        self.root.geometry("1800x1050")
        self.root.minsize(1450, 900)
        self._update_job = None

        self._build_variables()
        self._build_layout()
        self._attach_traces()
        self.update_all()

    def _build_variables(self):
        self.var_material = tk.StringVar(value="AlNiCo 5")
        self.var_gap_material = tk.StringVar(value="Plastic/Air")
        self.var_polarity = tk.StringVar(value="Attracting")

        self.var_len_top = tk.StringVar(value="50")
        self.var_wid_top = tk.StringVar(value="40")
        self.var_thk_top = tk.StringVar(value="10")
        self.var_len_bottom = tk.StringVar(value="50")
        self.var_wid_bottom = tk.StringVar(value="40")
        self.var_thk_bottom = tk.StringVar(value="10")
        self.var_gap_length = tk.StringVar(value="3")
        self.var_twist = tk.StringVar(value="0")
        self.var_wedge = tk.StringVar(value="0")

        self.var_show_gradient = tk.IntVar(value=1)
        self.var_show_vectors = tk.IntVar(value=1)

    def _build_layout(self):
        self.root.grid_rowconfigure(0, weight=1)
        for c in range(3):
            self.root.grid_columnconfigure(c, weight=1, uniform="panels")

        self.panel_left = ttk.Frame(self.root, padding=8)
        self.panel_center = ttk.Frame(self.root, padding=8)
        self.panel_right = ttk.Frame(self.root, padding=8)
        self.panel_left.grid(row=0, column=0, sticky="nsew")
        self.panel_center.grid(row=0, column=1, sticky="nsew")
        self.panel_right.grid(row=0, column=2, sticky="nsew")

        for panel in (self.panel_left, self.panel_center, self.panel_right):
            panel.grid_rowconfigure(0, weight=1, uniform="half")
            panel.grid_rowconfigure(1, weight=1, uniform="half")
            panel.grid_columnconfigure(0, weight=1)

        self.left_top = ttk.LabelFrame(self.panel_left, text="Inputs", padding=8)
        self.left_bottom = ttk.LabelFrame(self.panel_left, text="General Equations", padding=8)
        self.left_top.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.left_bottom.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        self.center_top = ttk.LabelFrame(self.panel_center, text="B-H Graph", padding=8)
        self.center_bottom = ttk.LabelFrame(self.panel_center, text="Numerical Equations", padding=8)
        self.center_top.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.center_bottom.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        self.right_top = ttk.LabelFrame(self.panel_right, text="Setup Visualization", padding=8)
        self.right_bottom = ttk.LabelFrame(self.panel_right, text="Results & Analytics", padding=8)
        self.right_top.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.right_bottom.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        self._build_inputs_panel()
        self._build_equations_panel()
        self._build_graph_panel()
        self._build_visual_panel()
        self._build_results_panel()

    def _attach_traces(self):
        tracked = [
            self.var_material, self.var_gap_material, self.var_polarity,
            self.var_len_top, self.var_wid_top, self.var_thk_top,
            self.var_len_bottom, self.var_wid_bottom, self.var_thk_bottom,
            self.var_gap_length, self.var_twist, self.var_wedge,
            self.var_show_gradient, self.var_show_vectors,
        ]
        for var in tracked:
            var.trace_add("write", lambda *_: self.schedule_update())

    def _build_inputs_panel(self):
        sf = ScrollableFrame(self.left_top)
        sf.pack(fill="both", expand=True)
        frm = sf.scrollable_frame
        frm.grid_columnconfigure(1, weight=1)

        def add_row(row: int, label: str, widget):
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
            widget.grid(row=row, column=1, sticky="ew", pady=3)

        r = 0
        add_row(r, "Material", ttk.Combobox(frm, textvariable=self.var_material, values=list(MATERIALS.keys()), state="readonly"))
        r += 1
        add_row(r, "Gap Material", ttk.Combobox(frm, textvariable=self.var_gap_material, values=list(GAP_MEDIA.keys()), state="readonly"))
        r += 1
        add_row(r, "Polarity", ttk.Combobox(frm, textvariable=self.var_polarity, values=["Attracting", "Repelling"], state="readonly"))
        r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1
        ttk.Label(frm, text="Top magnet geometry (mm)", font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))
        r += 1
        for label, var in [("Length", self.var_len_top), ("Width", self.var_wid_top), ("Thickness", self.var_thk_top)]:
            add_row(r, f"Top {label}", ttk.Entry(frm, textvariable=var, width=14))
            r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1
        ttk.Label(frm, text="Bottom magnet geometry (mm)", font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))
        r += 1
        for label, var in [("Length", self.var_len_bottom), ("Width", self.var_wid_bottom), ("Thickness", self.var_thk_bottom)]:
            add_row(r, f"Bottom {label}", ttk.Entry(frm, textvariable=var, width=14))
            r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1
        ttk.Label(frm, text="Gap / angle controls", font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 4))
        r += 1

        add_row(r, "Gap Length (mm)", ttk.Entry(frm, textvariable=self.var_gap_length, width=14))
        r += 1
        self.gap_slider = ttk.Scale(frm, from_=0.1, to=30.0, orient="horizontal", command=self._sync_gap_slider)
        self.gap_slider.set(float(self.var_gap_length.get()))
        add_row(r, "Gap Slider", self.gap_slider)
        r += 1

        add_row(r, "Twist Angle (°)", ttk.Entry(frm, textvariable=self.var_twist, width=14))
        r += 1
        self.twist_slider = ttk.Scale(frm, from_=0.0, to=90.0, orient="horizontal", command=self._sync_twist_slider)
        self.twist_slider.set(float(self.var_twist.get()))
        add_row(r, "Twist Slider", self.twist_slider)
        r += 1

        add_row(r, "Wedge Angle (°)", ttk.Entry(frm, textvariable=self.var_wedge, width=14))
        r += 1
        self.wedge_slider = ttk.Scale(frm, from_=0.0, to=90.0, orient="horizontal", command=self._sync_wedge_slider)
        self.wedge_slider.set(float(self.var_wedge.get()))
        add_row(r, "Wedge Slider", self.wedge_slider)
        r += 1

        ttk.Separator(frm, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1
        ttk.Checkbutton(frm, text="Show Magnetic Gradient", variable=self.var_show_gradient).grid(row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1
        ttk.Checkbutton(frm, text="Show Force Vectors", variable=self.var_show_vectors).grid(row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1

        ttk.Label(
            frm,
            text="All geometry values are interpreted in millimetres.\nThe model updates live as you edit values.",
            justify="left",
            foreground="#444",
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _sync_gap_slider(self, val):
        try:
            self.var_gap_length.set(f"{float(val):.4f}")
        except Exception:
            pass

    def _sync_twist_slider(self, val):
        try:
            self.var_twist.set(f"{float(val):.4f}")
        except Exception:
            pass

    def _sync_wedge_slider(self, val):
        try:
            self.var_wedge.set(f"{float(val):.4f}")
        except Exception:
            pass

    def _build_equations_panel(self):
        self.raw_eq_text = tk.Text(self.left_bottom, wrap="word", height=10, font=("Consolas", 10))
        self.raw_eq_text.pack(fill="both", expand=True)
        self.raw_eq_text.configure(state="disabled")

        self.num_eq_text = tk.Text(self.center_bottom, wrap="word", height=10, font=("Consolas", 10))
        self.num_eq_text.pack(fill="both", expand=True)
        self.num_eq_text.configure(state="disabled")

    def _build_graph_panel(self):
        self.fig_bh = Figure(figsize=(5.5, 4.0), dpi=100)
        self.ax_bh = self.fig_bh.add_subplot(111)
        self.canvas_bh = FigureCanvasTkAgg(self.fig_bh, master=self.center_top)
        self.canvas_bh.get_tk_widget().pack(fill="both", expand=True)

    def _build_visual_panel(self):
        controls = ttk.Frame(self.right_top)
        controls.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(controls, text="Show Magnetic Gradient", variable=self.var_show_gradient).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(controls, text="Show Force Vectors", variable=self.var_show_vectors).pack(side="left")

        self.fig_setup = Figure(figsize=(5.5, 4.0), dpi=100)
        self.ax_setup = self.fig_setup.add_subplot(111)
        self.canvas_setup = FigureCanvasTkAgg(self.fig_setup, master=self.right_top)
        self.canvas_setup.get_tk_widget().pack(fill="both", expand=True)

    def _build_results_panel(self):
        self.results_grid = ttk.Frame(self.right_bottom)
        self.results_grid.pack(fill="x", expand=False)

        self.lbl_force = self._add_result_row(self.results_grid, 0, "Total Clamping Force", "0 N")
        self.lbl_force_kg = self._add_result_row(self.results_grid, 1, "Equivalent in kgf", "0 kg")
        self.lbl_top = self._add_result_row(self.results_grid, 2, "Force felt by top magnet", "0 N")
        self.lbl_bottom = self._add_result_row(self.results_grid, 3, "Force felt by bottom magnet", "0 N")
        self.lbl_bg = self._add_result_row(self.results_grid, 4, "Operating Flux Density B_g", "0 T")
        self.lbl_hop = self._add_result_row(self.results_grid, 5, "Operating Field H_m", "0 A/m")
        self.lbl_op = self._add_result_row(self.results_grid, 6, "Operating Point", "(0, 0)")

        ttk.Separator(self.right_bottom, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(self.right_bottom, text="Engineering Warnings", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.warning_text = tk.Text(self.right_bottom, wrap="word", height=8, font=("Consolas", 10))
        self.warning_text.pack(fill="both", expand=True, pady=(4, 0))
        self.warning_text.configure(state="disabled")

    def _add_result_row(self, parent, row: int, title: str, initial: str):
        frm = ttk.Frame(parent)
        frm.grid(row=row, column=0, sticky="ew", pady=2)
        parent.grid_columnconfigure(0, weight=1)
        ttk.Label(frm, text=title + ":", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        lbl = ttk.Label(frm, text=initial, font=("TkDefaultFont", 13))
        lbl.pack(side="right")
        return lbl

    # ---------------------------- model computation ---------------------------

    def schedule_update(self):
        if self._update_job is not None:
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(80, self.update_all)

    def _read_float(self, var: tk.StringVar, default: float) -> float:
        try:
            return float(var.get())
        except Exception:
            return default

    def _gather_inputs(self) -> Dict[str, float | str]:
        return {
            "material": self.var_material.get(),
            "gap_material": self.var_gap_material.get(),
            "polarity": self.var_polarity.get(),
            "lt": max(1e-6, self._read_float(self.var_len_top, 50.0) * 1e-3),
            "wt": max(1e-6, self._read_float(self.var_wid_top, 40.0) * 1e-3),
            "tt": max(1e-6, self._read_float(self.var_thk_top, 10.0) * 1e-3),
            "lb": max(1e-6, self._read_float(self.var_len_bottom, 50.0) * 1e-3),
            "wb": max(1e-6, self._read_float(self.var_wid_bottom, 40.0) * 1e-3),
            "tb": max(1e-6, self._read_float(self.var_thk_bottom, 10.0) * 1e-3),
            "gap": max(1e-6, self._read_float(self.var_gap_length, 3.0) * 1e-3),
            "twist": clamp(self._read_float(self.var_twist, 0.0), 0.0, 90.0),
            "wedge": clamp(self._read_float(self.var_wedge, 0.0), 0.0, 90.0),
        }

    def _equivalent_wedge_length(self, W: float, L: float, l0: float, alpha_deg: float) -> Tuple[float, float, float, float]:
        """Return (P_gap, R_gap, equivalent_gap_length, dR/dl0)."""
        if alpha_deg <= 1e-9:
            A = W * L
            P = MU0 * A / max(l0, 1e-12)
            R = 1.0 / P
            dR = 1.0 / (MU0 * A + 1e-30)
            return P, R, l0, dR

        a = math.radians(alpha_deg)
        ta = max(1e-12, math.tan(a))
        l0 = max(l0, 1e-12)
        P = (MU0 * W / ta) * math.log(1.0 + (L * ta / l0))
        R = 1.0 / max(P, 1e-30)
        A = W * L
        l_eff = MU0 * A * R

        # dP/dl0 from the wedge permeance integral
        dP_dl0 = -(MU0 * W * L) / (l0 * (l0 + L * ta))
        dR_dl0 = -dP_dl0 / max(P * P, 1e-30)
        return P, R, l_eff, dR_dl0

    def compute_model(self) -> ModelResult:
        cfg = self._gather_inputs()
        material = cfg["material"]
        gap_material = cfg["gap_material"]
        polarity = cfg["polarity"]

        lt, wt, tt = cfg["lt"], cfg["wt"], cfg["tt"]
        lb, wb, tb = cfg["lb"], cfg["wb"], cfg["tb"]
        gap = cfg["gap"]
        twist = cfg["twist"]
        wedge = cfg["wedge"]

        mat = MATERIALS[material]
        Br = mat["Br"]
        mu_d = mat["mu_d"]
        mu_gap = GAP_MEDIA[gap_material]

        area_top = lt * wt
        area_bottom = lb * wb
        area_magnet = math.sqrt(area_top * area_bottom)

        overlap = overlap_area_rectangles(lt, wt, lb, wb, twist)
        if overlap <= 1e-12:
            overlap = min(area_top, area_bottom)

        W_eff = min(wt, wb)
        L_eff = min(lt, lb)
        projected_area = W_eff * L_eff
        active_gap_area = overlap
        if wedge > 0:
            active_gap_area = max(1e-12, projected_area * math.cos(math.radians(wedge)))

        if active_gap_area <= 1e-15:
            active_gap_area = max(1e-15, projected_area)

        if wedge > 0:
            P_gap, R_gap, l_eff_gap, dR_dl0 = self._equivalent_wedge_length(W_eff, L_eff, gap, wedge)
            equivalent_gap_length = l_eff_gap
        else:
            P_gap = mu_gap * active_gap_area / max(gap, 1e-12)
            R_gap = 1.0 / max(P_gap, 1e-30)
            equivalent_gap_length = gap
            dR_dl0 = 1.0 / (mu_gap * active_gap_area + 1e-30)

        l_m = 0.5 * (tt + tb)
        load_line_k = mu_gap * (active_gap_area / max(area_magnet, 1e-15)) * (l_m / max(equivalent_gap_length, 1e-15))

        H_op = -Br / max(mu_d + load_line_k, 1e-30)
        Bm_op = Br + mu_d * H_op
        Bg_op = Bm_op * (area_magnet / max(active_gap_area, 1e-15))

        if wedge > 0:
            phi = Bg_op * active_gap_area
            force_attract_N = 0.5 * phi * phi * dR_dl0
        else:
            force_attract_N = (active_gap_area * Bg_op * Bg_op) / (2.0 * MU0)

        if polarity == "Attracting":
            force_factor = 1.0
            force_signed_N = force_attract_N
        else:
            force_factor = 0.65 if gap_material == "Plastic/Air" else 0.10
            if material == "AlNiCo 5" and gap_material == "Steel":
                force_factor *= 0.5
            if wedge > 0:
                force_factor *= max(0.25, math.cos(math.radians(wedge)))
            if twist > 0:
                force_factor *= max(0.15, math.cos(math.radians(twist)))
            force_signed_N = -force_attract_N * force_factor

        force_top_N = force_signed_N
        force_bottom_N = force_signed_N
        force_kgf = abs(force_signed_N) * KGF_PER_NEWTON

        warnings: List[str] = []
        if material == "AlNiCo 5" and abs(H_op) > 2.0e4:
            warnings.append("AlNiCo 5 is operating at high demagnetizing stress; recoil margin is low.")
        if material == "AlNiCo 5" and wedge >= 10.0:
            warnings.append("High wedge angle with AlNiCo 5 increases the risk of local demagnetization.")
        if gap_material == "Steel" and polarity == "Repelling":
            warnings.append("Repelling through steel can create strong crosstalk and mutual demagnetization.")
        if gap_material == "Steel" and polarity == "Repelling" and material == "AlNiCo 5":
            warnings.append("WARNING: Mutual demagnetization risk is especially high for AlNiCo in this mode.")
        if polarity == "Repelling" and abs(force_signed_N) < 0.25 * force_attract_N:
            warnings.append("Snap/Stick risk: the repelling coupling is much weaker than the underlying attractive bias.")
        if wedge >= 20.0:
            warnings.append("Large wedge angle causes a strong force gradient and nonuniform pressure concentration.")
        if twist >= 45.0:
            warnings.append("Large twist angle substantially reduces overlap area and force transfer.")
        if gap < 0.5e-3:
            warnings.append("Very small gap can make the model optimistic; real assemblies will be limited by contact, saturation, and tolerances.")
        if gap_material == "Steel" and polarity == "Attracting":
            warnings.append("Steel separator acts like a magnetic short circuit; the magnets operate very close to Br.")
        if not warnings:
            warnings.append("No major engineering warning triggered for the current parameter set.")

        return ModelResult(
            material=material,
            gap_material=gap_material,
            polarity=polarity,
            length_top=lt,
            width_top=wt,
            thickness_top=tt,
            length_bottom=lb,
            width_bottom=wb,
            thickness_bottom=tb,
            gap_length=gap,
            twist_angle_deg=twist,
            wedge_angle_deg=wedge,
            area_top=area_top,
            area_bottom=area_bottom,
            area_magnet=area_magnet,
            overlap_area=overlap,
            active_gap_area=active_gap_area,
            effective_gap_length=gap,
            equivalent_gap_length=equivalent_gap_length,
            gap_permeance=P_gap,
            gap_reluctance=R_gap,
            dR_dl0=dR_dl0,
            Br=Br,
            mu_d=mu_d,
            mu_gap=mu_gap,
            load_line_k=load_line_k,
            H_op=H_op,
            Bm_op=Bm_op,
            Bg_op=Bg_op,
            force_attract_N=force_attract_N,
            force_signed_N=force_signed_N,
            force_factor=force_factor,
            force_top_N=force_top_N,
            force_bottom_N=force_bottom_N,
            force_kgf=force_kgf,
            warnings=warnings,
        )

    # ----------------------- equation text / results text -----------------------

    def _raw_equations_text(self) -> str:
        return (
            "Raw symbolic equations used by the model\n\n"
            "1) Demagnetization curve\n"
            "   AlNiCo 5:   B_m = B_r + μ_rec H_m\n"
            "   NdFeB N45:  B_m = B_r + μ0 μ_r H_m\n\n"
            "2) Magnetic-circuit load line\n"
            "   H_m l_m + H_g l_g,eq = 0\n"
            "   B_g = (A_m / A_g) B_m\n"
            "   H_g = B_g / (μ0 μ_r,gap)\n"
            "   => B_m = - μ_gap (A_g / A_m) (l_m / l_g,eq) H_m\n\n"
            "3) Operating point\n"
            "   B_r + μ_d H_m = -k H_m\n"
            "   H_m = -B_r / (μ_d + k)\n"
            "   B_m = B_r + μ_d H_m\n"
            "   B_g = B_m (A_m / A_g)\n\n"
            "4) Force\n"
            "   Face-to-face:  F = A_g B_g^2 / (2 μ0)\n"
            "   Wedge gap:     P_gap = (μ0 W / tan α) ln(1 + (L tan α / l0))\n"
            "                  R_gap = 1 / P_gap\n"
            "                  F = (1/2) Φ^2 dR_gap/dl0\n"
            "   Twist overlap reduces A_g via polygon intersection.\n\n"
            "5) Interaction modes\n"
            "   Attracting:  use the clamping force directly.\n"
            "   Repelling:    apply a reduced coupling factor for crosstalk through steel.\n"
        )

    def _numeric_equations_text(self, m: ModelResult) -> str:
        mu_label = "μ_rec" if m.material == "AlNiCo 5" else "μ0 μ_r"
        force_block = (
            f"F_attract = A_g B_g² / (2 μ0) = {fmt(m.force_attract_N, 8)} N\n"
            if m.wedge_angle_deg <= 1e-9
            else (
                f"P_gap = {fmt(m.gap_permeance, 8)} Wb/A\n"
                f"R_gap = {fmt(m.gap_reluctance, 8)} A/Wb\n"
                f"dR_gap/dl0 = {fmt(m.dR_dl0, 8)} A/(Wb·m)\n"
                f"F = 1/2 · Φ² · dR_gap/dl0 = {fmt(m.force_attract_N, 8)} N\n"
            )
        )

        return (
            "Same equations with current numbers substituted\n\n"
            f"Material = {m.material}\n"
            f"B_r = {fmt(m.Br, 6)} T\n"
            f"{mu_label} = {fmt(m.mu_d, 8)} T·m/A\n"
            f"A_top = {fmt(m.area_top, 8)} m²\n"
            f"A_bottom = {fmt(m.area_bottom, 8)} m²\n"
            f"A_m = sqrt(A_top A_bottom) = {fmt(m.area_magnet, 8)} m²\n"
            f"A_g = {fmt(m.active_gap_area, 8)} m²\n"
            f"l_m = {(0.5 * (m.thickness_top + m.thickness_bottom)):.8f} m\n"
            f"l_g,eq = {fmt(m.equivalent_gap_length, 8)} m\n"
            f"μ_gap = {fmt(m.mu_gap, 8)} H/m\n"
            f"k = μ_gap (A_g / A_m) (l_m / l_g,eq) = {fmt(m.load_line_k, 8)}\n"
            f"H_m = -B_r / (μ_d + k) = {fmt(m.H_op, 8)} A/m\n"
            f"B_m = B_r + μ_d H_m = {fmt(m.Bm_op, 8)} T\n"
            f"B_g = B_m (A_m / A_g) = {fmt(m.Bg_op, 8)} T\n"
            f"{force_block}"
            f"Force factor = {fmt(m.force_factor, 8)}\n"
            f"Signed force = {fmt(m.force_signed_N, 8)} N\n"
            f"kgf = |F| / 9.80665 = {fmt(m.force_kgf, 8)} kgf\n"
        )

    # ---------------------------- plotting helpers ----------------------------

    def update_bh_graph(self, m: ModelResult):
        ax = self.ax_bh
        ax.clear()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

        H_op = m.H_op
        H_min = min(H_op * 1.8, -1.0)
        H = np.linspace(H_min, 0.0, 400)
        B_demag = m.Br + m.mu_d * H
        B_load = -m.load_line_k * H

        ax.plot(H, B_demag, label="Demagnetization curve", linewidth=2)
        ax.plot(H, B_load, label="Load line", linewidth=2)
        ax.scatter([m.H_op], [m.Bm_op], s=60, zorder=5, label="Operating point")
        ax.annotate(
            f"OP\nH={fmt(m.H_op,4)}\nB={fmt(m.Bm_op,4)}",
            (m.H_op, m.Bm_op),
            textcoords="offset points",
            xytext=(12, 12),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666", alpha=0.85),
        )

        ax.set_title("2nd-quadrant B-H intersection")
        ax.set_xlabel("H_m (A/m)")
        ax.set_ylabel("B_m (T)")
        ax.legend(loc="best")
        self.fig_bh.tight_layout()
        self.canvas_bh.draw_idle()

    def update_setup_visual(self, m: ModelResult):
        ax = self.ax_setup
        ax.clear()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

        lt = m.length_top * 1e3
        wt = m.width_top * 1e3
        tt = m.thickness_top * 1e3
        lb = m.length_bottom * 1e3
        wb = m.width_bottom * 1e3
        tb = m.thickness_bottom * 1e3
        gap = m.gap_length * 1e3
        wedge = math.radians(m.wedge_angle_deg)
        twist = m.twist_angle_deg

        L = min(lt, lb)
        W = min(wt, wb)
        g0 = gap
        g1 = gap + L * math.tan(wedge)
        y_top_max = tb + max(g0, g1) + tt

        bottom_rect = Rectangle((0, 0), L, tb, facecolor="#8ecae6", edgecolor="#1d3557", alpha=0.55)
        ax.add_patch(bottom_rect)
        ax.text(L * 0.02, tb * 0.5, "Bottom magnet", fontsize=9, va="center")

        if m.wedge_angle_deg <= 1e-6:
            gap_rect = Rectangle((0, tb), L, gap, facecolor="#ffb703", edgecolor="#fb8500", alpha=0.25)
            ax.add_patch(gap_rect)
            top_poly = [(0, tb + gap), (L, tb + gap), (L, tb + gap + tt), (0, tb + gap + tt)]
        else:
            gap_poly = [(0, tb), (L, tb), (L, tb + g1), (0, tb + g0)]
            ax.add_patch(Polygon(gap_poly, closed=True, facecolor="#ffb703", edgecolor="#fb8500", alpha=0.22))
            top_poly = [(0, tb + g0), (L, tb + g1), (L, tb + g1 + tt), (0, tb + g0 + tt)]

        top_patch = Polygon(top_poly, closed=True, facecolor="#90be6d", edgecolor="#2a9d8f", alpha=0.55)
        ax.add_patch(top_patch)
        ax.text(L * 0.02, max(tb + g0, tb + g1) + tt * 0.5, "Top magnet", fontsize=9, va="center")

        if self.var_show_gradient.get():
            x = np.linspace(0, L, 140)
            y = np.linspace(tb, y_top_max, 90)
            X, Y = np.meshgrid(x, y)
            gap_profile = gap + X * math.tan(wedge)
            inside = (Y >= tb) & (Y <= tb + gap_profile)
            local_gap = np.clip(gap_profile, 1e-6, None)
            Z = np.where(inside, 1.0 / local_gap, np.nan)
            ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", alpha=0.35)

        if self.var_show_vectors.get():
            arrow_len = max(tt, tb, gap) * 0.85
            if m.polarity == "Attracting":
                top_dir = -arrow_len
                bottom_dir = arrow_len
            else:
                top_dir = arrow_len
                bottom_dir = -arrow_len

            ax.annotate("", xy=(L * 0.5, tb + gap / 2 + tt + top_dir), xytext=(L * 0.5, tb + gap / 2 + tt),
                        arrowprops=dict(arrowstyle="->", lw=2))
            ax.annotate("", xy=(L * 0.5, tb / 2 + bottom_dir), xytext=(L * 0.5, tb / 2),
                        arrowprops=dict(arrowstyle="->", lw=2))
            ax.text(L * 0.52, tb + gap / 2 + tt + top_dir, "F", fontsize=11)
            ax.text(L * 0.52, tb / 2 + bottom_dir, "F", fontsize=11)

        ax.text(L * 0.65, tb * 0.1, f"Gap = {fmt(gap,2)} mm", fontsize=9)
        ax.text(L * 0.65, tb + gap + tt + 0.2, f"Twist = {fmt(twist,2)}°", fontsize=9)
        ax.text(L * 0.65, tb + gap * 0.5, f"Wedge = {fmt(m.wedge_angle_deg,2)}°", fontsize=9)

        inset = ax.inset_axes([0.64, 0.58, 0.33, 0.34])
        inset.set_aspect("equal", adjustable="box")
        inset.set_title("Twist overlap", fontsize=8)
        inset.grid(True, alpha=0.2)

        bottom = rectangle_polygon(0.0, 0.0, L, W, 0.0)
        top = rectangle_polygon(0.0, 0.0, lt if lt > 0 else L, wt if wt > 0 else W, twist)
        inter = convex_polygon_clip(top, bottom)

        inset.add_patch(Polygon(bottom, closed=True, facecolor="#8ecae6", edgecolor="#1d3557", alpha=0.25))
        inset.add_patch(Polygon(top, closed=True, facecolor="#90be6d", edgecolor="#2a9d8f", alpha=0.30))
        if len(inter) >= 3:
            inset.add_patch(Polygon(inter, closed=True, facecolor="#f4a261", edgecolor="#e76f51", alpha=0.65))
        inset.set_xlim(-max(L, lt) * 0.8, max(L, lt) * 0.8)
        inset.set_ylim(-max(W, wt) * 0.8, max(W, wt) * 0.8)
        inset.tick_params(labelsize=6)

        ax.set_xlim(-5, L + 5)
        ax.set_ylim(-tb * 0.25, y_top_max + 5)
        ax.set_xlabel("Length direction (mm)")
        ax.set_ylabel("Stack direction (mm)")
        ax.set_title("Sandwich geometry schematic")
        self.fig_setup.tight_layout()
        self.canvas_setup.draw_idle()

    # ------------------------------- update loop -------------------------------

    def update_all(self):
        self._update_job = None
        m = self.compute_model()

        self._set_text(self.raw_eq_text, self._raw_equations_text())
        self._set_text(self.num_eq_text, self._numeric_equations_text(m))
        self.update_bh_graph(m)
        self.update_setup_visual(m)

        self.lbl_force.configure(text=f"{abs(m.force_signed_N):.3f} N")
        self.lbl_force_kg.configure(text=f"{abs(m.force_signed_N) * KGF_PER_NEWTON:.3f} kgf")
        self.lbl_top.configure(text=self._force_text(m.force_top_N))
        self.lbl_bottom.configure(text=self._force_text(m.force_bottom_N))
        self.lbl_bg.configure(text=f"{m.Bg_op:.4f} T")
        self.lbl_hop.configure(text=f"{m.H_op:.3f} A/m")
        self.lbl_op.configure(text=f"({m.H_op:.3f}, {m.Bm_op:.4f})")

        self._set_text(self.warning_text, "\n".join(f"• {w}" for w in m.warnings))

    def _force_text(self, f: float) -> str:
        if self.var_polarity.get() == "Repelling":
            sign = "↑" if f < 0 else "↓"
        else:
            sign = "↓" if f >= 0 else "↑"
        return f"{f:+.3f} N  {sign}"

    def _set_text(self, widget: tk.Text, content: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    MagneticClampApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
