"""
Magnetic Clamp Simulator — UI Application
==========================================
A customtkinter + matplotlib GUI for simulating magnetic clamping
forces in a symmetric 5-layer sandwich configuration.

  Panel 1 (Left):   Parameters & Algebraic Equations
  Panel 2 (Centre): B-H Curve & Numerical Equations
  Panel 3 (Right):  Setup Visualisation & Results

Run:  python main.py
"""

import sys
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from physics_engine import MagneticCircuitEngine, MATERIALS, MU_0
from library_engine import MagpylibEngine
import plot_manager as pm

# ── Theme ──
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Catppuccin Mocha palette
C = {
    "base":     "#1e1e2e",
    "mantle":   "#181825",
    "crust":    "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "text":     "#cdd6f4",
    "subtext":  "#a6adc8",
    "blue":     "#89b4fa",
    "green":    "#a6e3a1",
    "red":      "#f38ba8",
    "peach":    "#fab387",
    "mauve":    "#cba6f7",
    "teal":     "#94e2d5",
    "yellow":   "#f9e2af",
}

FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_LABEL = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 10)
FONT_MONO  = ("Consolas", 11)
FONT_RESULT = ("Consolas", 12, "bold")


class ToggleSwitch(ctk.CTkFrame):
    """A labelled toggle switch with two mutually-exclusive options."""

    def __init__(self, master, label, option_a, option_b,
                 command=None, default_b=False, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self._command = command
        self._var = ctk.BooleanVar(value=default_b)
        self._option_a = option_a
        self._option_b = option_b

        self._lbl = ctk.CTkLabel(self, text=label, font=FONT_LABEL,
                                 text_color=C["subtext"])
        self._lbl.pack(side="top", anchor="w", pady=(0, 2))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x")

        self._lbl_a = ctk.CTkLabel(row, text=option_a, font=FONT_SMALL,
                                   text_color=C["text"], width=80)
        self._lbl_a.pack(side="left")

        self._switch = ctk.CTkSwitch(
            row, text="", variable=self._var,
            onvalue=True, offvalue=False,
            command=self._on_toggle,
            width=48,
            button_color=C["blue"],
            progress_color=C["mauve"],
        )
        self._switch.pack(side="left", padx=6)

        self._lbl_b = ctk.CTkLabel(row, text=option_b, font=FONT_SMALL,
                                   text_color=C["text"], width=80)
        self._lbl_b.pack(side="left")

    def _on_toggle(self):
        # Highlight active label
        if self._var.get():
            self._lbl_a.configure(text_color=C["subtext"])
            self._lbl_b.configure(text_color=C["yellow"])
        else:
            self._lbl_a.configure(text_color=C["yellow"])
            self._lbl_b.configure(text_color=C["subtext"])
        if self._command:
            self._command()

    @property
    def value(self):
        return self._option_b if self._var.get() else self._option_a

    @property
    def is_b(self):
        return self._var.get()


class SliderEntry(ctk.CTkFrame):
    """A slider with a synchronised numeric entry box."""

    def __init__(self, master, label, unit, from_, to, default,
                 resolution=0.1, command=None, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self._command = command
        self._from = from_
        self._to = to
        self._resolution = resolution

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x")

        ctk.CTkLabel(header, text=f"{label} ({unit})", font=FONT_LABEL,
                     text_color=C["subtext"]).pack(side="left")

        self._var = ctk.DoubleVar(value=default)

        self._entry = ctk.CTkEntry(
            header, width=70, font=FONT_MONO,
            textvariable=self._var,
            fg_color=C["surface0"], border_color=C["surface1"],
            text_color=C["text"],
        )
        self._entry.pack(side="right")
        self._entry.bind("<Return>", self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

        self._slider = ctk.CTkSlider(
            self, from_=from_, to=to,
            number_of_steps=int((to - from_) / resolution),
            variable=self._var,
            command=self._on_slide,
            button_color=C["blue"],
            progress_color=C["mauve"],
            fg_color=C["surface0"],
        )
        self._slider.pack(fill="x", pady=(2, 4))

    def _on_slide(self, val):
        rounded = round(float(val) / self._resolution) * self._resolution
        self._var.set(round(rounded, 6))
        if self._command:
            self._command()

    def _on_entry(self, event=None):
        try:
            v = float(self._var.get())
            v = max(self._from, min(self._to, v))
            self._var.set(round(v, 6))
            self._slider.set(v)
        except (ValueError, TypeError):
            pass
        if self._command:
            self._command()

    @property
    def value(self):
        return float(self._var.get())


class MagneticClampApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Magnetic Clamp Simulator — 5-Layer Sandwich")
        self.configure(fg_color=C["base"])
        self.geometry("1720x960")
        self.minsize(1400, 800)

        self.engine = MagneticCircuitEngine()
        self.lib_engine = MagpylibEngine()

        self._figures = {}   # name → (fig, canvas)
        self._build_layout()
        self._schedule_update()  # initial solve

    # ────────────────────────────────────────────────────────────
    #  Layout
    # ────────────────────────────────────────────────────────────

    def _build_layout(self):
        # Three equal panels
        self.grid_columnconfigure((0, 1, 2), weight=1, uniform="panel")
        self.grid_rowconfigure(0, weight=1)

        self._panel1 = ctk.CTkFrame(self, fg_color=C["mantle"], corner_radius=10)
        self._panel2 = ctk.CTkFrame(self, fg_color=C["mantle"], corner_radius=10)
        self._panel3 = ctk.CTkFrame(self, fg_color=C["mantle"], corner_radius=10)

        self._panel1.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        self._panel2.grid(row=0, column=1, sticky="nsew", padx=3, pady=6)
        self._panel3.grid(row=0, column=2, sticky="nsew", padx=(3, 6), pady=6)

        for p in (self._panel1, self._panel2, self._panel3):
            p.grid_rowconfigure(0, weight=1)
            p.grid_rowconfigure(1, weight=1)
            p.grid_columnconfigure(0, weight=1)

        self._build_panel1()
        self._build_panel2()
        self._build_panel3()

    # ── Panel 1: Parameters & Algebra ──

    def _build_panel1(self):
        top = ctk.CTkScrollableFrame(self._panel1, fg_color=C["crust"],
                                     corner_radius=8,
                                     label_text="  ⚙  Parameters",
                                     label_font=FONT_TITLE,
                                     label_fg_color=C["surface0"],
                                     label_text_color=C["blue"])
        top.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))

        cmd = self._schedule_update

        self.sl_L  = SliderEntry(top, "Magnet Length", "mm", 5, 200, 50,
                                 resolution=1, command=cmd)
        self.sl_W  = SliderEntry(top, "Magnet Width", "mm", 5, 100, 25,
                                 resolution=1, command=cmd)
        self.sl_tm = SliderEntry(top, "Magnet Thickness", "mm", 1, 80, 10,
                                 resolution=0.5, command=cmd)
        self.sl_tc = SliderEntry(top, "Cover Thickness", "mm", 0, 20, 1,
                                 resolution=0.1, command=cmd)
        self.sl_g  = SliderEntry(top, "Min Gap Length", "mm", 0.1, 50, 2,
                                 resolution=0.1, command=cmd)
        self.sl_a  = SliderEntry(top, "Wedge Angle α", "°", 0, 45, 0,
                                 resolution=0.5, command=cmd)

        for s in (self.sl_L, self.sl_W, self.sl_tm,
                  self.sl_tc, self.sl_g, self.sl_a):
            s.pack(fill="x", padx=10, pady=3)

        sep = ctk.CTkFrame(top, height=2, fg_color=C["surface1"])
        sep.pack(fill="x", padx=10, pady=8)

        self.tgl_mat = ToggleSwitch(top, "Magnet Material",
                                    "AlNiCo 5", "NdFeB N45",
                                    command=cmd, default_b=True)
        self.tgl_mat.pack(fill="x", padx=10, pady=4)

        self.tgl_cover = ToggleSwitch(top, "Cover Material",
                                      "Plastic/Air", "Steel",
                                      command=cmd, default_b=False)
        self.tgl_cover.pack(fill="x", padx=10, pady=4)

        self.tgl_pol = ToggleSwitch(top, "Interaction Mode",
                                    "Attract", "Repel",
                                    command=cmd, default_b=False)
        self.tgl_pol.pack(fill="x", padx=10, pady=4)

        # Bottom: algebraic equations
        bot = ctk.CTkFrame(self._panel1, fg_color=C["crust"], corner_radius=8)
        bot.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
        lbl = ctk.CTkLabel(bot, text="  📐  General Equations",
                           font=FONT_TITLE, text_color=C["blue"])
        lbl.pack(anchor="w", padx=8, pady=(6, 0))
        self._eq_alg_frame = bot

    # ── Panel 2: B-H Curve & Numerical Equations ──

    def _build_panel2(self):
        top = ctk.CTkFrame(self._panel2, fg_color=C["crust"], corner_radius=8)
        top.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))
        lbl = ctk.CTkLabel(top, text="  📈  B-H Curve & Operating Point",
                           font=FONT_TITLE, text_color=C["blue"])
        lbl.pack(anchor="w", padx=8, pady=(6, 0))
        self._bh_frame = top

        bot = ctk.CTkFrame(self._panel2, fg_color=C["crust"], corner_radius=8)
        bot.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
        lbl2 = ctk.CTkLabel(bot, text="  🔢  Numerical Substitution",
                            font=FONT_TITLE, text_color=C["blue"])
        lbl2.pack(anchor="w", padx=8, pady=(6, 0))
        self._eq_num_frame = bot

    # ── Panel 3: Visualisation & Results ──

    def _build_panel3(self):
        top = ctk.CTkFrame(self._panel3, fg_color=C["crust"], corner_radius=8)
        top.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 2))

        top_header = ctk.CTkFrame(top, fg_color="transparent")
        top_header.pack(fill="x", padx=8, pady=(6, 0))
        ctk.CTkLabel(top_header, text="  🧲  Setup Visualisation",
                     font=FONT_TITLE, text_color=C["blue"]).pack(side="left")

        # Toggles for gradient / vectors
        self._show_gradient = ctk.BooleanVar(value=True)
        self._show_vectors  = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(top_header, text="Gradient", font=FONT_SMALL,
                      variable=self._show_gradient,
                      command=self._schedule_update,
                      button_color=C["teal"],
                      text_color=C["subtext"]).pack(side="right", padx=4)
        ctk.CTkSwitch(top_header, text="Vectors", font=FONT_SMALL,
                      variable=self._show_vectors,
                      command=self._schedule_update,
                      button_color=C["teal"],
                      text_color=C["subtext"]).pack(side="right", padx=4)

        self._vis_frame = top

        # Bottom: Results
        bot = ctk.CTkFrame(self._panel3, fg_color=C["crust"], corner_radius=8)
        bot.grid(row=1, column=0, sticky="nsew", padx=4, pady=(2, 4))
        ctk.CTkLabel(bot, text="  📊  Results",
                     font=FONT_TITLE, text_color=C["blue"]
                     ).pack(anchor="w", padx=8, pady=(6, 0))

        results_grid = ctk.CTkFrame(bot, fg_color="transparent")
        results_grid.pack(fill="x", padx=12, pady=4)
        results_grid.grid_columnconfigure(0, weight=2)
        results_grid.grid_columnconfigure(1, weight=1)
        results_grid.grid_columnconfigure(2, weight=1)

        # Headers
        for col, txt in [(0, "Quantity"), (1, "Analytical"), (2, "magpylib")]:
            ctk.CTkLabel(results_grid, text=txt, font=("Segoe UI", 10, "bold"),
                         text_color=C["subtext"]).grid(
                row=0, column=col, sticky="w", padx=6, pady=2)

        self._result_labels = {}
        rows = [
            ("F_total",  "Total Clamping Force",    "0 N"),
            ("F_kgf",    "Equivalent in kgf",       "0 kg"),
            ("F_top",    "Force felt by top magnet", "0 N"),
            ("F_bot",    "Force felt by bottom magnet", "0 N"),
            ("Bg",       "Operating Flux Density Bᵍ", "0 T"),
            ("Hm",       "Operating Field Hₘ",       "0 A/m"),
            ("OP",       "Operating Point",          "(0, 0)"),
        ]
        for i, (key, label, default) in enumerate(rows, start=1):
            self._add_result_row(results_grid, i, key, label, default)

        # Warnings box
        ctk.CTkLabel(bot, text="Physics Warnings", font=FONT_LABEL,
                     text_color=C["peach"]).pack(anchor="w", padx=12, pady=(8, 0))
        self._warnings_box = ctk.CTkTextbox(
            bot, height=80, font=FONT_SMALL,
            fg_color=C["surface0"], text_color=C["yellow"],
            border_color=C["surface1"], border_width=1,
            wrap="word",
        )
        self._warnings_box.pack(fill="x", padx=12, pady=(2, 8))

    def _add_result_row(self, parent, row, key, label, default):
        ctk.CTkLabel(parent, text=label, font=FONT_LABEL,
                     text_color=C["text"], anchor="w").grid(
            row=row, column=0, sticky="w", padx=6, pady=3)

        lbl_ana = ctk.CTkLabel(parent, text=default, font=FONT_RESULT,
                               text_color=C["green"], anchor="e")
        lbl_ana.grid(row=row, column=1, sticky="e", padx=6, pady=3)

        lbl_lib = ctk.CTkLabel(parent, text="—", font=FONT_RESULT,
                               text_color=C["teal"], anchor="e")
        lbl_lib.grid(row=row, column=2, sticky="e", padx=6, pady=3)

        self._result_labels[key] = (lbl_ana, lbl_lib)

    # ────────────────────────────────────────────────────────────
    #  Update / solve
    # ────────────────────────────────────────────────────────────

    def _schedule_update(self, *_):
        """Debounce: delay solving to avoid lag during slider drag."""
        if hasattr(self, '_update_id'):
            self.after_cancel(self._update_id)
        self._update_id = self.after(120, self._do_update)

    def _do_update(self):
        # ── 1. Push UI values into the engine ──
        self.engine.set_geometry(
            L_mm=self.sl_L.value,
            W_mm=self.sl_W.value,
            tm_mm=self.sl_tm.value,
            tc_mm=self.sl_tc.value,
            g_min_mm=self.sl_g.value,
            alpha_deg=self.sl_a.value,
        )
        self.engine.magnet_material = self.tgl_mat.value
        self.engine.cover_is_steel  = self.tgl_cover.is_b
        self.engine.is_attracting   = not self.tgl_pol.is_b

        # ── 2. Solve analytical model ──
        results = self.engine.solve()

        # ── 3. Solve magpylib model ──
        lib_results = self.lib_engine.compute(
            L=self.engine.L, W=self.engine.W, tm=self.engine.tm,
            tc=self.engine.tc, g_min=self.engine.g_min,
            alpha_deg=self.engine.alpha_deg,
            magnet_material=self.engine.magnet_material,
            cover_is_steel=self.engine.cover_is_steel,
            is_attracting=self.engine.is_attracting,
        )

        # ── 4. Update all plots ──
        self._update_bh(results)
        self._update_equations_algebraic()
        self._update_equations_numerical(results)
        self._update_visualisation(results)
        self._update_results(results, lib_results)

    # ── Plot helpers ──

    def _embed_figure(self, parent, name, fig):
        """Embed a matplotlib figure in a ctk frame, replacing any prior."""
        if name in self._figures:
            old_fig, old_canvas = self._figures[name]
            old_canvas.get_tk_widget().destroy()
            plt.close(old_fig)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._figures[name] = (fig, canvas)

    def _update_bh(self, results):
        fig = pm.create_bh_figure(results, self.engine.magnet_material)
        self._embed_figure(self._bh_frame, "bh", fig)

    def _update_equations_algebraic(self):
        fig = pm.create_equations_algebraic(self.engine)
        self._embed_figure(self._eq_alg_frame, "eq_alg", fig)

    def _update_equations_numerical(self, results):
        fig = pm.create_equations_numerical(self.engine, results)
        self._embed_figure(self._eq_num_frame, "eq_num", fig)

    def _update_visualisation(self, results):
        fig = pm.create_setup_figure(
            self.engine, results,
            show_gradient=self._show_gradient.get(),
            show_vectors=self._show_vectors.get(),
        )
        self._embed_figure(self._vis_frame, "vis", fig)

    def _update_results(self, results, lib_results):
        r = results
        lr = lib_results

        def fmt_ana(key):
            if key == "F_total": return f"{r['F_total']:.2f} N"
            if key == "F_kgf":  return f"{r['F_kgf']:.2f} kgf"
            if key == "F_top":  return f"{r['F_top']:.2f} N"
            if key == "F_bot":  return f"{r['F_bot']:.2f} N"
            if key == "Bg":     return f"{r['Bg']:.4f} T"
            if key == "Hm":     return f"{r['Hm_op']:.0f} A/m"
            if key == "OP":     return f"({r['Hm_op']/1e3:.1f} kA/m, {r['Bm_op']:.3f} T)"
            return "—"

        def fmt_lib(key):
            if "error" in lr:
                return "N/A"
            if key == "F_total": return f"{lr['F_total']:.2f} N"
            if key == "F_kgf":  return f"{lr['F_kgf']:.2f} kgf"
            if key == "F_top":  return f"{lr['F_total']:.2f} N"
            if key == "F_bot":  return f"{lr['F_total']:.2f} N"
            if key == "Bg":     return f"{lr['B_gap']:.4f} T"
            if key in ("Hm", "OP"):  return "—"
            return "—"

        for key, (lbl_a, lbl_l) in self._result_labels.items():
            lbl_a.configure(text=fmt_ana(key))
            lbl_l.configure(text=fmt_lib(key))

        # Warnings
        self._warnings_box.configure(state="normal")
        self._warnings_box.delete("1.0", "end")
        for w in r.get("warnings", []):
            self._warnings_box.insert("end", w + "\n")
        self._warnings_box.configure(state="disabled")
