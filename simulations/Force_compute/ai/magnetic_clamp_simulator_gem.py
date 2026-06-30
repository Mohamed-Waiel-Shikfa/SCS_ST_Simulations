import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon

# --- Physics Constants ---
MU_0 = 4 * math.pi * 1e-7

MATERIALS = {
    "NdFeB (N45)": {"Br": 1.35, "mu_r": 1.05, "type": "hard"},
    "AlNiCo 5": {"Br": 1.2, "mu_rec": 1.8218e-5, "type": "epm"}
}

GAP_MATERIALS = {
    "Plastic/Air": 1.0,
    "Steel": 2000.0
}

class MagneticSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Magnetic Circuit Clamping Simulator")
        self.root.geometry("1600x900")

        # Configure strict 3-panel vertical layout
        for i in range(3):
            self.root.columnconfigure(i, weight=1, uniform="col")
        self.root.rowconfigure(0, weight=1)

        self.create_panels()
        self.setup_ui()
        self.update_simulation() # Initial run

    def create_panels(self):
        # Panel 1: Left
        self.p1 = ttk.Frame(self.root, borderwidth=2, relief="groove")
        self.p1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.p1.rowconfigure(0, weight=1, uniform="row")
        self.p1.rowconfigure(1, weight=1, uniform="row")

        # Panel 2: Center
        self.p2 = ttk.Frame(self.root, borderwidth=2, relief="groove")
        self.p2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.p2.rowconfigure(0, weight=1, uniform="row")
        self.p2.rowconfigure(1, weight=1, uniform="row")

        # Panel 3: Right
        self.p3 = ttk.Frame(self.root, borderwidth=2, relief="groove")
        self.p3.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.p3.rowconfigure(0, weight=1, uniform="row")
        self.p3.rowconfigure(1, weight=1, uniform="row")

        # Top/Bottom sub-frames
        self.p1_top = ttk.LabelFrame(self.p1, text="Inputs & Configurations")
        self.p1_bot = ttk.LabelFrame(self.p1, text="General Equations")
        self.p1_top.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.p1_bot.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.p2_top = ttk.LabelFrame(self.p2, text="B-H Demagnetization Curve")
        self.p2_bot = ttk.LabelFrame(self.p2, text="Numerical Equations (Step-by-Step)")
        self.p2_top.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.p2_bot.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.p3_top = ttk.LabelFrame(self.p3, text="Setup Visualization")
        self.p3_bot = ttk.LabelFrame(self.p3, text="Results & Analytics")
        self.p3_top.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.p3_bot.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def setup_ui(self):
        # --- Panel 1 Top: Inputs ---
        self.vars = {
            "mat": tk.StringVar(value="NdFeB (N45)"),
            "L": tk.DoubleVar(value=50.0),
            "W": tk.DoubleVar(value=25.0),
            "t1": tk.DoubleVar(value=10.0),
            "t2": tk.DoubleVar(value=10.0),
            "lg": tk.DoubleVar(value=2.0),
            "gap_mat": tk.StringVar(value="Plastic/Air"),
            "twist": tk.DoubleVar(value=0.0),
            "wedge": tk.DoubleVar(value=0.0),
            "polarity": tk.StringVar(value="Attracting")
        }

        row = 0
        def add_slider(parent, label, var, from_, to_):
            nonlocal row
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            scale = ttk.Scale(parent, from_=from_, to=to_, variable=var, command=lambda e: self.update_simulation())
            scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            val_label = ttk.Label(parent, text="")
            val_label.grid(row=row, column=2, sticky="w")

            # Sync label with slider
            def update_lbl(*args):
                val_label.config(text=f"{var.get():.1f}")
            var.trace_add("write", update_lbl)
            update_lbl()
            row += 1

        ttk.Label(self.p1_top, text="Core Material:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(self.p1_top, textvariable=self.vars["mat"], values=list(MATERIALS.keys()), state="readonly").grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        self.vars["mat"].trace_add("write", lambda *args: self.update_simulation())
        row += 1

        add_slider(self.p1_top, "Length (mm):", self.vars["L"], 10, 100)
        add_slider(self.p1_top, "Width (mm):", self.vars["W"], 10, 100)
        add_slider(self.p1_top, "Top Mag Thick (mm):", self.vars["t1"], 1, 50)
        add_slider(self.p1_top, "Bot Mag Thick (mm):", self.vars["t2"], 1, 50)
        add_slider(self.p1_top, "Gap Length lg (mm):", self.vars["lg"], 0.1, 50)

        ttk.Label(self.p1_top, text="Gap Material:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(self.p1_top, textvariable=self.vars["gap_mat"], values=list(GAP_MATERIALS.keys()), state="readonly").grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        self.vars["gap_mat"].trace_add("write", lambda *args: self.update_simulation())
        row += 1

        add_slider(self.p1_top, "Twist Angle (°):", self.vars["twist"], 0, 90)
        add_slider(self.p1_top, "Wedge Angle (°):", self.vars["wedge"], 0, 45) # Max 45 to prevent absurd geometries

        ttk.Label(self.p1_top, text="Polarity:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(self.p1_top, textvariable=self.vars["polarity"], values=["Attracting", "Repelling"], state="readonly").grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        self.vars["polarity"].trace_add("write", lambda *args: self.update_simulation())

        self.p1_top.columnconfigure(1, weight=1)

        # --- Panel 1 Bottom: General Equations ---
        self.gen_eq_text = tk.Text(self.p1_bot, wrap="word", bg="#f0f0f0", font=("Consolas", 11))
        self.gen_eq_text.pack(fill="both", expand=True, padx=5, pady=5)

        equations = (
            "--- Demagnetization Curve ---\n"
            "Bm = Br + \u03bc_rec * Hm\n\n"
            "--- Permeance & Load Line ---\n"
            "P_gap (wedge) = (\u03bc_0 * \u03bc_r * W / tan(\u03b1)) * ln(1 + (L*tan(\u03b1))/lg)\n"
            "R_gap = 1 / P_gap\n"
            "Pc_slope = - lm / (Am * \u03bc_0 * R_gap)\n\n"
            "--- Operating Point Intersection ---\n"
            "Hm = Br / (Pc_slope * \u03bc_0 - \u03bc_rec)\n"
            "Bm = Pc_slope * \u03bc_0 * Hm\n\n"
            "--- Clamping Force ---\n"
            "F = \u222B (B(x)\u00B2 / 2\u03bc_0) dA  * Area_Overlap_Factor\n"
        )
        self.gen_eq_text.insert("1.0", equations)
        self.gen_eq_text.config(state="disabled")

        # --- Panel 2 Top: B-H Graph ---
        self.fig_bh, self.ax_bh = plt.subplots(figsize=(4, 3))
        self.canvas_bh = FigureCanvasTkAgg(self.fig_bh, master=self.p2_top)
        self.canvas_bh.get_tk_widget().pack(fill="both", expand=True)

        # --- Panel 2 Bottom: Numerical Equations ---
        self.num_eq_text = tk.Text(self.p2_bot, wrap="word", bg="#ffffff", font=("Consolas", 11))
        self.num_eq_text.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Panel 3 Top: Visualization ---
        self.vis_controls = ttk.Frame(self.p3_top)
        self.vis_controls.pack(fill="x")
        self.show_grad = tk.BooleanVar(value=True)
        self.show_force = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.vis_controls, text="Show Magnetic Gradient", variable=self.show_grad, command=self.update_simulation).pack(side="left", padx=5)
        ttk.Checkbutton(self.vis_controls, text="Show Force Vectors", variable=self.show_force, command=self.update_simulation).pack(side="left", padx=5)

        self.fig_vis, self.ax_vis = plt.subplots(figsize=(4, 3))
        self.canvas_vis = FigureCanvasTkAgg(self.fig_vis, master=self.p3_top)
        self.canvas_vis.get_tk_widget().pack(fill="both", expand=True)

        # --- Panel 3 Bottom: Results ---
        self.lbl_force_n = ttk.Label(self.p3_bot, text="Total Force: 0.00 N", font=("Arial", 16, "bold"))
        self.lbl_force_n.pack(anchor="w", padx=10, pady=(10,0))
        self.lbl_force_kg = ttk.Label(self.p3_bot, text="Force Equivalent: 0.00 kg", font=("Arial", 12))
        self.lbl_force_kg.pack(anchor="w", padx=10)
        self.lbl_bg = ttk.Label(self.p3_bot, text="Operating Bg: 0.000 T", font=("Arial", 12))
        self.lbl_bg.pack(anchor="w", padx=10, pady=(0, 10))

        self.txt_warnings = tk.Text(self.p3_bot, height=6, wrap="word", bg="#ffe6e6", fg="#990000", font=("Arial", 10, "bold"))
        self.txt_warnings.pack(fill="x", padx=10, pady=5)

    def calculate_physics(self):
        # Retrieve and convert inputs to SI units
        L = self.vars["L"].get() / 1000.0
        W = self.vars["W"].get() / 1000.0
        t1 = self.vars["t1"].get() / 1000.0
        t2 = self.vars["t2"].get() / 1000.0
        lg = self.vars["lg"].get() / 1000.0
        twist_rad = math.radians(self.vars["twist"].get())
        wedge_rad = math.radians(self.vars["wedge"].get())

        mat_prop = MATERIALS[self.vars["mat"].get()]
        Br = mat_prop["Br"]
        if "mu_rec" in mat_prop:
            mu_rec = mat_prop["mu_rec"]
        else:
            mu_rec = mat_prop["mu_r"] * MU_0

        mu_r_gap = GAP_MATERIALS[self.vars["gap_mat"].get()]
        is_repelling = self.vars["polarity"].get() == "Repelling"

        lm = t1 + t2
        Am = L * W

        # Twist Angle Area Approximation (Linearly scales effective overlapping area)
        A_min = min(L, W)**2
        Ag = Am - (Am - A_min) * math.sin(twist_rad)
        area_factor = Ag / Am

        # Reluctance & Permeance
        if wedge_rad < 1e-5: # Parallel faces
            R_gap = lg / (MU_0 * mu_r_gap * Ag)
        else:
            P_gap = (MU_0 * mu_r_gap * W / math.tan(wedge_rad)) * math.log(1 + (L * math.tan(wedge_rad))/lg)
            R_gap = 1.0 / (P_gap * area_factor)

        # Repelling flux short circuit adjustment
        if is_repelling and mu_r_gap > 1.0:
            R_gap = R_gap * 0.05 # Steel severely shorts the repelling field

        # Load Line Slope (Pc)
        Pc_slope = -lm / (Am * MU_0 * R_gap)

        # Intersection Point (Operating Point)
        Hm = Br / (Pc_slope * MU_0 - mu_rec)
        Bm = Pc_slope * MU_0 * Hm

        # Clamping Force Integration
        if wedge_rad < 1e-5:
            Bg = Bm * (Am / Ag)
            F = (Bg**2 * Ag) / (2 * MU_0)
        else:
            MMF = -Hm * lm
            F_integral = ((MMF * MU_0)**2 * W * L) / (2 * MU_0 * lg * (lg + L * math.tan(wedge_rad)))
            F = F_integral * area_factor

        if is_repelling:
            F = -F

        return {
            "L": L, "W": W, "t1": t1, "t2": t2, "lg": lg, "wedge": wedge_rad,
            "Br": Br, "mu_rec": mu_rec, "Hm": Hm, "Bm": Bm, "Pc": Pc_slope,
            "F": F, "Bg": Bm*(Am/Ag), "R_gap": R_gap
        }

    def update_simulation(self):
        try:
            res = self.calculate_physics()
        except Exception as e:
            return # Skip if transient invalid state

        # 1. Update Numerical Equations Panel
        self.num_eq_text.config(state="normal")
        self.num_eq_text.delete("1.0", tk.END)
        step_text = (
            f"--- Material Properties ---\n"
            f"Br = {res['Br']:.2f} T\n"
            f"\u03bc_rec = {res['mu_rec']:.2e} T\u00B7m/A\n\n"
            f"--- Magnetic Circuit Analysis ---\n"
            f"R_gap = {res['R_gap']:.2e} A\u00B7t/Wb\n"
            f"Pc_slope = {res['Pc']:.2f}\n"
            f"Hm = {res['Br']:.2f} / ({res['Pc']:.2f} * \u03bc_0 - {res['mu_rec']:.2e}) = {res['Hm']:.0f} A/m\n"
            f"Bm = {res['Pc']:.2f} * \u03bc_0 * {res['Hm']:.0f} = {res['Bm']:.4f} T\n\n"
            f"--- Force Output ---\n"
            f"Total Force F = {res['F']:.2f} N\n"
        )
        self.num_eq_text.insert("1.0", step_text)
        self.num_eq_text.config(state="disabled")

        # 2. Update B-H Plot
        self.ax_bh.clear()
        self.ax_bh.set_title("2nd Quadrant B-H Curve")
        self.ax_bh.set_xlabel("Field Strength H (A/m)")
        self.ax_bh.set_ylabel("Flux Density B (T)")
        self.ax_bh.grid(True, linestyle="--", alpha=0.6)

        H_c = -res['Br'] / res['mu_rec']
        h_vals = np.linspace(H_c * 1.1, 0, 100)
        b_vals = res['Br'] + res['mu_rec'] * h_vals

        self.ax_bh.plot(h_vals, b_vals, 'b-', label="Demagnetization Curve")
        self.ax_bh.plot([0, res['Hm']*1.2], [0, res['Pc'] * MU_0 * res['Hm']*1.2], 'r--', label="Load Line")
        self.ax_bh.plot(res['Hm'], res['Bm'], 'go', markersize=8, label="Operating Point")

        self.ax_bh.set_xlim(H_c * 1.1, max(-H_c*0.1, 1000))
        self.ax_bh.set_ylim(0, res['Br'] * 1.2)
        self.ax_bh.legend(loc="lower right", fontsize=8)
        self.canvas_bh.draw()

        # 3. Update Visualization Plot (Side View)
        self.ax_vis.clear()
        self.ax_vis.set_title("2D Side-Profile Geometry")
        self.ax_vis.axis('equal')
        self.ax_vis.axis('off')

        L = res['L']
        lg = res['lg']
        t1 = res['t1']
        t2 = res['t2']
        wedge = res['wedge']

        # Coordinates
        bot_mag = np.array([[0, -t2], [L, -t2], [L, 0], [0, 0]])
        y_gap_end = lg + L * math.tan(wedge)
        gap_poly = np.array([[0, 0], [L, 0], [L, y_gap_end], [0, lg]])
        top_mag = np.array([[0, lg], [L, y_gap_end], [L, y_gap_end + t1], [0, lg + t1]])

        # Draw Polygons
        self.ax_vis.add_patch(Polygon(bot_mag, closed=True, facecolor='#a0a0a0', edgecolor='black'))
        self.ax_vis.add_patch(Polygon(top_mag, closed=True, facecolor='#606060', edgecolor='black'))

        # Gradient
        if self.show_grad.get():
            cmap = plt.get_cmap('plasma')
            # Normalize color intensity based on Bg
            intensity = min(abs(res['Bg']) / 1.5, 1.0)
            self.ax_vis.add_patch(Polygon(gap_poly, closed=True, facecolor=cmap(intensity), alpha=0.5))
        else:
            self.ax_vis.add_patch(Polygon(gap_poly, closed=True, facecolor='none', edgecolor='blue', linestyle=':'))

        # Force Vectors
        if self.show_force.get() and abs(res['F']) > 0.1:
            arrow_dir = -1 if res['F'] > 0 else 1 # Attract (down) vs Repel (up)
            center_x = L / 2
            top_center_y = lg + (L/2)*math.tan(wedge) + t1/2
            self.ax_vis.arrow(center_x, top_center_y, 0, arrow_dir * (t1/1.5), head_width=L/15, head_length=t1/4, fc='red', ec='red')

        self.ax_vis.autoscale(enable=True)
        self.canvas_vis.draw()

        # 4. Update Results & Analytics
        force_n = res['F']
        force_kg = force_n / 9.81
        self.lbl_force_n.config(text=f"Total Force: {force_n:,.2f} N")
        self.lbl_force_kg.config(text=f"Force Equivalent: {force_kg:,.2f} kg")
        self.lbl_bg.config(text=f"Operating Bg: {res['Bg']:.4f} T")

        # Warnings Logic
        warnings = []
        if self.vars["polarity"].get() == "Repelling" and self.vars["gap_mat"].get() == "Steel":
            warnings.append("WARNING: Repelling through steel without active cancellation causes catastrophic magnetic flux shorting.")
        if res['Hm'] < (-res['Br']/res['mu_rec']) * 0.8 and "AlNiCo" in self.vars["mat"].get():
            warnings.append("CRITICAL: High demagnetization stress on AlNiCo core. Magnet is operating near its coercive limit.")
        if self.vars["polarity"].get() == "Repelling" and force_n > -5:
            warnings.append("WARNING: Snap/Stick Risk. Repelling force is extremely weak; magnets may overpower and violently snap together if perturbed.")

        self.txt_warnings.delete("1.0", tk.END)
        if warnings:
            self.txt_warnings.insert("1.0", "\n\n".join(warnings))
        else:
            self.txt_warnings.insert("1.0", "System Operating Nominally. No severe magnetic stresses detected.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MagneticSimulatorApp(root)
    root.mainloop()
