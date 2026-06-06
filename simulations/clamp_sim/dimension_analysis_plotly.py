import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

# ==========================================
# 1. Constants & Physics Engine
# ==========================================
mu0 = 4 * np.pi * 1e-7
g = 9.81
alnico_density = 7300
Hc = -59000

H_data = np.array([-59000, -56000, -52000, -45000, -30000, -15000, 0])
B_data = np.array([0.0,    0.50,   0.90,   1.15,   1.28,   1.33,   1.35])
demag_curve = CubicSpline(H_data, B_data)

def calc_force(am, lm, lg):
    wim = np.sqrt(am)
    ag_eff = (wim + lg) ** 2

    Pt = mu0 * (ag_eff / lg)
    m_load = (-2 * lm * Pt) / am

    def intersection_eq(Hm):
        return (m_load * Hm) - demag_curve(Hm)

    try:
        res = root_scalar(intersection_eq, bracket=[Hc, 0], method='brentq')
        Hm_intersect = res.root
    except ValueError:
        Hm_intersect = 0

    Bm_intersect = m_load * Hm_intersect
    bg = Bm_intersect * (am / ag_eff)
    return (ag_eff * (bg ** 2)) / (2 * mu0)

v_calc_force = np.vectorize(calc_force)

def calc_weight(am, lm):
    return alnico_density * am * lm * g

# ==========================================
# 2. Parameter Ranges & Grid Setup
# ==========================================
am_min, am_max, am_init = 10, 400, 100        # mm^2
lm_min, lm_max, lm_init = 0.5, 15.0, 5.0      # mm
lg_min, lg_max, lg_init = 0.05, 2.0, 0.5      # mm

grid_res = 50
FORCE_CAP = 150.0  # Cap the force spike near zero gap

# Pre-generate the grids
LM_grid, LG_grid = np.meshgrid(np.linspace(lm_min, lm_max, grid_res), np.linspace(lg_min, lg_max, grid_res))
AM_grid, LG_grid_2 = np.meshgrid(np.linspace(am_min, am_max, grid_res), np.linspace(lg_min, lg_max, grid_res))
AM_grid_3, LM_grid_3 = np.meshgrid(np.linspace(am_min, am_max, grid_res), np.linspace(lm_min, lm_max, grid_res))

# ==========================================
# 3. Dash App Initialization
# ==========================================
app = Dash(__name__)
app.title = "Magnetic Circuit Explorer"

# Helper to format Plotly 3D scenes
def get_scene_layout(xaxis_title, yaxis_title, zaxis_title='Force (N)'):
    return dict(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        zaxis_title=zaxis_title,
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        aspectratio=dict(x=1, y=1, z=0.8)
    )

# --- UI Layout ---
app.layout = html.Div(style={'font-family': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f8f9fa'}, children=[
    html.H1("Magnetic Circuit Dimension Analysis", style={'textAlign': 'center', 'color': '#343a40'}),

    # Sliders Control Panel
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}, children=[
        html.Div(style={'width': '30%'}, children=[
            html.Label(id='am-label', style={'fontWeight': 'bold'}),
            dcc.Slider(id='slider-am', min=am_min, max=am_max, step=5, value=am_init, marks={i: f"{i}" for i in range(0, 401, 100)})
        ]),
        html.Div(style={'width': '30%'}, children=[
            html.Label(id='lm-label', style={'fontWeight': 'bold'}),
            dcc.Slider(id='slider-lm', min=lm_min, max=lm_max, step=0.5, value=lm_init, marks={i: f"{i}" for i in range(0, 16, 5)})
        ]),
        html.Div(style={'width': '30%'}, children=[
            html.Label(id='lg-label', style={'fontWeight': 'bold'}),
            dcc.Slider(id='slider-lg', min=lg_min, max=lg_max, step=0.05, value=lg_init, marks={i: f"{i}" for i in range(0, 3, 1)})
        ]),
    ]),

    # 2x2 Graph Grid
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'marginTop': '20px'}, children=[
        html.Div(dcc.Graph(id='graph-1', style={'height': '600px'}), style={'width': '50%'}),
        html.Div(dcc.Graph(id='graph-2', style={'height': '600px'}), style={'width': '50%'}),
        html.Div(dcc.Graph(id='graph-3', style={'height': '600px'}), style={'width': '50%'}),
        html.Div(dcc.Graph(id='graph-4', style={'height': '600px'}), style={'width': '50%'}),
    ])
])

# ==========================================
# 4. Interactive Callbacks
# ==========================================

@app.callback(
    [Output('graph-1', 'figure'), Output('am-label', 'children')],
    Input('slider-am', 'value')
)
def update_plot1(am_val):
    Z = np.clip(v_calc_force(am_val * 1e-6, LM_grid * 1e-3, LG_grid * 1e-3), 0, FORCE_CAP)
    fig = go.Figure(data=[go.Surface(z=Z, x=LM_grid, y=LG_grid, colorscale='Turbo')])
    fig.update_layout(title=f'Force vs (Length, Gap) [Area = {am_val} mm²]', scene=get_scene_layout('Length lm (mm)', 'Gap lg (mm)'))
    return fig, f"Constant Area (Am): {am_val} mm²"

@app.callback(
    [Output('graph-2', 'figure'), Output('lm-label', 'children')],
    Input('slider-lm', 'value')
)
def update_plot2(lm_val):
    Z = np.clip(v_calc_force(AM_grid * 1e-6, lm_val * 1e-3, LG_grid_2 * 1e-3), 0, FORCE_CAP)
    fig = go.Figure(data=[go.Surface(z=Z, x=AM_grid, y=LG_grid_2, colorscale='Magma')])
    fig.update_layout(title=f'Force vs (Area, Gap) [Length = {lm_val} mm]', scene=get_scene_layout('Area Am (mm²)', 'Gap lg (mm)'))
    return fig, f"Constant Length (lm): {lm_val} mm"

@app.callback(
    [Output('graph-3', 'figure'), Output('lg-label', 'children')],
    Input('slider-lg', 'value')
)
def update_plot3(lg_val):
    Z = np.clip(v_calc_force(AM_grid_3 * 1e-6, LM_grid_3 * 1e-3, lg_val * 1e-3), 0, FORCE_CAP)
    fig = go.Figure(data=[go.Surface(z=Z, x=AM_grid_3, y=LM_grid_3, colorscale='Viridis')])
    fig.update_layout(title=f'Force vs (Area, Length) [Gap = {lg_val} mm]', scene=get_scene_layout('Area Am (mm²)', 'Length lm (mm)'))
    return fig, f"Constant Gap (lg): {lg_val} mm"

@app.callback(
    Output('graph-4', 'figure'),
    Input('slider-am', 'value') # Just triggering once on load
)
def update_plot4(_):
    Z = calc_weight(AM_grid_3 * 1e-6, LM_grid_3 * 1e-3)
    fig = go.Figure(data=[go.Surface(z=Z, x=AM_grid_3, y=LM_grid_3, colorscale='Ocean')])
    fig.update_layout(title='Static Weight vs (Area, Length)', scene=get_scene_layout('Area Am (mm²)', 'Length lm (mm)', 'Weight (N)'))
    return fig

# ==========================================
# 5. Run Server
# ==========================================
if __name__ == '__main__':
    print("\nStarting Dashboard! Open http://127.0.0.1:8050 in your web browser.\n")
    app.run(debug=True)
