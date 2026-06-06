import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from matplotlib import cm

# ==========================================
# 1. Constants & Material Properties
# ==========================================
mu0 = 4 * np.pi * 1e-7
g = 9.81
alnico_density = 7300  # kg/m^3
Hc = -59000            # Coercivity (A/m)

# Alnico 5-7 Demagnetization Curve
H_data = np.array([-59000, -56000, -52000, -45000, -30000, -15000, 0])
B_data = np.array([0.0,    0.50,   0.90,   1.15,   1.28,   1.33,   1.35])
demag_curve = CubicSpline(H_data, B_data)

# ==========================================
# 2. Vectorized Physics Engine
# ==========================================
def calc_force(am, lm, lg):
    """Calculates holding force. Assumes square magnet face for fringing."""
    # Assuming square magnet face: wim = hem = sqrt(am)
    wim = np.sqrt(am)
    wim_eff = wim + lg
    ag_eff = wim_eff ** 2  # Effective gap area

    Pt = mu0 * (ag_eff / lg)
    m_load = (-2 * lm * Pt) / am

    def intersection_eq(Hm):
        return (m_load * Hm) - demag_curve(Hm)

    try:
        res = root_scalar(intersection_eq, bracket=[Hc, 0], method='brentq')
        Hm_intersect = res.root
    except ValueError:
        Hm_intersect = 0  # Fallback if solver escapes bounds

    Bm_intersect = m_load * Hm_intersect
    bg = Bm_intersect * (am / ag_eff)
    return (ag_eff * (bg ** 2)) / (2 * mu0)

# Vectorize the function so it can evaluate entire 2D grids at once very quickly
v_calc_force = np.vectorize(calc_force)

def calc_weight(am, lm):
    """Calculates weight force of a single magnet."""
    return alnico_density * am * lm * g

# ==========================================
# 3. Parameter Ranges (in meters)
# ==========================================
# Am: 1 mm^2 to 5 cm^2
am_min, am_max, am_init = 1e-6, 500e-6, 100e-6
# Lm: 0.1 mm to 15 mm
lm_min, lm_max, lm_init = 0.1e-3, 15e-3, 5e-3
# Lg: 0.05 mm to 2 mm
lg_min, lg_max, lg_init = 0.05e-3, 2e-3, 0.5e-3

# Grid resolution (lower = faster slider response, higher = prettier surface)
grid_res = 25

# ==========================================
# 4. Figure & UI Setup
# ==========================================
fig = plt.figure(figsize=(16, 10))
fig.canvas.manager.set_window_title('Magnetic Circuit Parameter Explorer')

# Adjust layout to make room for sliders at the bottom
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.92, wspace=0.2, hspace=0.2)

# Create the 4 subplots
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

# --- Helper function to format axes ---
def format_axis(ax, title, xlabel, ylabel, zlabel='Force (N)'):
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=25, azim=-125)

# ==========================================
# 5. Plotting Functions
# ==========================================
# Note: Matplotlib 3D doesn't easily update Z-data on the fly.
# The standard trick for 3D interactive plots is to clear() the axis and redraw.

# Plot 1: X=lm, Y=lg, Z=Force (Slider=am)
LM1, LG1 = np.meshgrid(np.linspace(lm_min, lm_max, grid_res),
                       np.linspace(lg_min, lg_max, grid_res))
def draw_plot1(am_val):
    ax1.clear()
    Z = v_calc_force(am_val, LM1, LG1)
    ax1.plot_surface(LM1*1000, LG1*1000, Z, cmap=cm.viridis, alpha=0.9)
    format_axis(ax1, f'Force vs (lm, lg) | Area = {am_val*1e6:.0f} mm²', 'lm (mm)', 'lg (mm)')

# Plot 2: X=am, Y=lg, Z=Force (Slider=lm)
AM2, LG2 = np.meshgrid(np.linspace(am_min, am_max, grid_res),
                       np.linspace(lg_min, lg_max, grid_res))
def draw_plot2(lm_val):
    ax2.clear()
    Z = v_calc_force(AM2, lm_val, LG2)
    ax2.plot_surface(AM2*1e6, LG2*1000, Z, cmap=cm.plasma, alpha=0.9)
    format_axis(ax2, f'Force vs (Am, lg) | Length = {lm_val*1000:.1f} mm', 'Am (mm²)', 'lg (mm)')

# Plot 3: X=am, Y=lm, Z=Force (Slider=lg)
AM3, LM3 = np.meshgrid(np.linspace(am_min, am_max, grid_res),
                       np.linspace(lm_min, lm_max, grid_res))
def draw_plot3(lg_val):
    ax3.clear()
    Z = v_calc_force(AM3, LM3, lg_val)
    ax3.plot_surface(AM3*1e6, LM3*1000, Z, cmap=cm.inferno, alpha=0.9)
    format_axis(ax3, f'Force vs (Am, lm) | Gap = {lg_val*1000:.2f} mm', 'Am (mm²)', 'lm (mm)')

# Plot 4: X=am, Y=lm, Z=Weight (No slider needed)
AM4, LM4 = np.meshgrid(np.linspace(am_min, am_max, grid_res),
                       np.linspace(lm_min, lm_max, grid_res))
def draw_plot4():
    ax4.clear()
    Z_weight = calc_weight(AM4, LM4)
    ax4.plot_surface(AM4*1e6, LM4*1000, Z_weight, color='grey', alpha=0.7, edgecolor='k', linewidth=0.2)
    format_axis(ax4, 'Static Magnet Weight vs (Am, lm)', 'Am (mm²)', 'lm (mm)', 'Weight (N)')

# Draw initial states
draw_plot1(am_init)
draw_plot2(lm_init)
draw_plot3(lg_init)
draw_plot4()

# ==========================================
# 6. Sliders & Callbacks
# ==========================================
# Slider positions (left, bottom, width, height)
ax_slider_am = plt.axes([0.15, 0.08, 0.25, 0.02])
ax_slider_lm = plt.axes([0.55, 0.08, 0.25, 0.02])
ax_slider_lg = plt.axes([0.35, 0.03, 0.25, 0.02])

# Create Sliders (display values mapped to mm for readability)
slider_am = Slider(ax_slider_am, 'Area (Am) [mm²]', am_min*1e6, am_max*1e6, valinit=am_init*1e6)
slider_lm = Slider(ax_slider_lm, 'Length (lm) [mm]', lm_min*1000, lm_max*1000, valinit=lm_init*1000)
slider_lg = Slider(ax_slider_lg, 'Gap (lg) [mm]', lg_min*1000, lg_max*1000, valinit=lg_init*1000)

# Callback functions mapping mm slider values back to meters for the physics engine
def update_am(val):
    draw_plot1(val * 1e-6)
    fig.canvas.draw_idle()

def update_lm(val):
    draw_plot2(val * 1e-3)
    fig.canvas.draw_idle()

def update_lg(val):
    draw_plot3(val * 1e-3)
    fig.canvas.draw_idle()

# Attach callbacks
slider_am.on_changed(update_am)
slider_lm.on_changed(update_lm)
slider_lg.on_changed(update_lg)

# ==========================================
# 7. Blender-Style Navigation Hack
# ==========================================
def apply_blender_navigation(fig, axes_list):
    """Injects Scroll-Wheel Zooming into Matplotlib 3D axes"""
    def on_scroll(event):
        if event.inaxes not in axes_list:
            return

        ax = event.inaxes
        # Blender uses scroll UP to zoom IN.
        # We scale the viewing box by 15% per scroll click.
        scale_factor = 0.85 if event.step > 0 else 1.15

        # Get current 3D limits
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()

        # Calculate the center of the current view
        x_mid, y_mid, z_mid = np.mean(xlim), np.mean(ylim), np.mean(zlim)

        # Calculate the new narrowed/widened ranges
        dx = (xlim[1] - xlim[0]) * scale_factor / 2
        dy = (ylim[1] - ylim[0]) * scale_factor / 2
        dz = (zlim[1] - zlim[0]) * scale_factor / 2

        # Apply the new zoomed limits
        ax.set_xlim3d([x_mid - dx, x_mid + dx])
        ax.set_ylim3d([y_mid - dy, y_mid + dy])
        ax.set_zlim3d([z_mid - dz, z_mid + dz])

        # Redraw the canvas
        fig.canvas.draw_idle()

    # Attach the event listener to the figure
    fig.canvas.mpl_connect('scroll_event', on_scroll)

# Apply the navigation hack to all 4 of your 3D plots
apply_blender_navigation(fig, [ax1, ax2, ax3, ax4])

# Now show the plot
plt.show()
