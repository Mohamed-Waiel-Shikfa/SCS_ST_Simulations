import math
from functools import cache
from linecache import cache

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

# =====================================================================
# 0. USER CONFIGURATIONS
# =====================================================================
# Area (Am) in mm²
AM_MIN = 100.0
AM_MAX = 900.0
AM_STEP = 5.0
AM_INIT = 100.0

# Length (lm) in mm
LM_MIN = 0.5
LM_MAX = 50.0
LM_STEP = 0.5
LM_INIT = 5.0

# Gap (lg) in mm
LG_MIN = 0.05
LG_MAX = 10.0
LG_STEP = 0.05
LG_INIT = 0.5

GRID_RESOLUTION = 60

# ==========================================
# 1. Constants & Physics Engine
# ==========================================
mu0 = 4 * np.pi * 1e-7
Hc = -65000
FORCE_CAP = 10_000.0

H_alnico_q2 = np.array(
    [
        -59500,
        -59000,
        -57500,
        -56000,
        -54000,
        -52000,
        -49000,
        -45000,
        -38000,
        -30000,
        -20000,
        -10000,
        0,
    ]
)
J_alnico_q2 = np.array(
    [
        0.0,
        0.074,
        0.305,
        0.570,
        0.800,
        0.965,
        1.100,
        1.207,
        1.285,
        1.318,
        1.338,
        1.346,
        1.35,
    ]
)

demag_curve = CubicSpline(H_alnico_q2, J_alnico_q2)

# ---------------------------------------------------------
# OPTIMIZATION: Pre-compute Intersection Lookup Table
# ---------------------------------------------------------
# Instead of running root_scalar thousands of times per callback,
# we map out the relationship between m_load and Hm once at startup.
Hm_lookup = np.linspace(Hc, -1e-6, 10000)
J_lookup = demag_curve(Hm_lookup)
m_load_lookup = J_lookup / Hm_lookup

# np.interp requires the x-axis (m_load) to be strictly increasing
sort_idx = np.argsort(m_load_lookup)
m_load_sorted = m_load_lookup[sort_idx]
Hm_sorted = Hm_lookup[sort_idx]


def fast_calc_force(area, lm, lg):
    """Fully vectorized physical calculation."""
    # Ensure inputs are treated as numpy arrays for grid operations
    area = np.asarray(area)
    lm = np.asarray(lm)
    lg = np.asarray(lg)

    perimeter = np.sqrt(area) * 4.0

    P_main = (mu0 * area) / lg
    P_edge = mu0 * 0.26 * perimeter
    P_corner = 4.0 * 0.077 * mu0 * lg
    Pt = P_main + P_edge + P_corner

    m_load = (-2.0 * lm * Pt) / area

    # Instant vectorized lookup replacing root_scalar
    Hm_intersect = np.interp(m_load, m_load_sorted, Hm_sorted)

    Bm_intersect = m_load * Hm_intersect
    A_roters = (Pt * lg) / mu0
    Bg = Bm_intersect * (area / A_roters)

    return (A_roters * (Bg**2)) / (2.0 * mu0)


# ==========================================
# 2. Variable Configurations Dictionary
# ==========================================
VAR_CONFIG = {
    "am": {
        "min": AM_MIN,
        "max": AM_MAX,
        "step": AM_STEP,
        "default": AM_INIT,
        "label": "Area Am (mm²)",
        "scale": 1e-6,
    },
    "lm": {
        "min": LM_MIN,
        "max": LM_MAX,
        "step": LM_STEP,
        "default": LM_INIT,
        "label": "Length lm (mm)",
        "scale": 1e-3,
    },
    "lg": {
        "min": LG_MIN,
        "max": LG_MAX,
        "step": LG_STEP,
        "default": LG_INIT,
        "label": "Gap lg (mm)",
        "scale": 1e-3,
    },
}

# ==========================================
# 3. Dash App Layout
# ==========================================
app = Dash(__name__)
app.title = "Magnetic Circuit Explorer"

# Styling constants
BOX_STYLE = {
    "backgroundColor": "white",
    "padding": "25px",
    "borderRadius": "10px",
    "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
    "marginBottom": "20px",
}

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
        "padding": "20px",
    },
    children=[
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "gap": "20px",
                "maxWidth": "2000px",
                "margin": "0 auto",
            },
            children=[
                # ==========================================
                # LEFT COLUMN (Controls)
                # ==========================================
                html.Div(
                    style={
                        "width": "35%",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                    children=[
                        # Section: Axis Settings
                        html.Div(
                            style=BOX_STYLE,
                            children=[
                                html.H3(
                                    "Axis Configuration",
                                    style={
                                        "marginTop": "0",
                                        "color": "#343a40",
                                        "marginBottom": "15px",
                                    },
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "row",
                                        "alignItems": "flex-end",
                                        "gap": "10px",
                                    },
                                    children=[
                                        html.Div(
                                            style={"flex": "1"},
                                            children=[
                                                html.Label(
                                                    "X Axis:",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "fontSize": "14px",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="dropdown-x",
                                                    options=[
                                                        {
                                                            "label": VAR_CONFIG[k][
                                                                "label"
                                                            ],
                                                            "value": k,
                                                        }
                                                        for k in VAR_CONFIG
                                                    ],
                                                    value="lm",
                                                    clearable=False,
                                                    style={"marginTop": "5px"},
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={"flex": "1"},
                                            children=[
                                                html.Label(
                                                    "Y Axis:",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "fontSize": "14px",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id="dropdown-y",
                                                    # Options populated dynamically by callback to include 'Force'
                                                    value="lg",
                                                    clearable=False,
                                                    style={"marginTop": "5px"},
                                                ),
                                            ],
                                        ),
                                        html.Button(
                                            "Switch View",
                                            id="btn-toggle-view",
                                            n_clicks=0,
                                            style={
                                                "height": "36px",
                                                "padding": "0 15px",
                                                "fontSize": "13px",
                                                "fontWeight": "bold",
                                                "cursor": "pointer",
                                                "backgroundColor": "#007bff",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "4px",
                                                "whiteSpace": "nowrap",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Section: Sliders
                        html.Div(
                            style=BOX_STYLE,
                            children=[
                                html.H3(
                                    "Fixed Constraints",
                                    style={
                                        "marginTop": "0",
                                        "color": "#343a40",
                                        "marginBottom": "20px",
                                    },
                                ),
                                html.Div(
                                    id="slider-1-container",
                                    children=[
                                        html.Label(
                                            id="slider-1-label",
                                            style={
                                                "fontWeight": "bold",
                                                "color": "#495057",
                                            },
                                        ),
                                        dcc.Slider(
                                            id="dynamic-slider-1",
                                            updatemode="drag",
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="slider-2-container",
                                    style={"marginTop": "30px"},
                                    children=[
                                        html.Label(
                                            id="slider-2-label",
                                            style={
                                                "fontWeight": "bold",
                                                "color": "#495057",
                                            },
                                        ),
                                        dcc.Slider(
                                            id="dynamic-slider-2",
                                            updatemode="drag",
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        # Section: Exact Calculator
                        html.Div(
                            style=BOX_STYLE,
                            children=[
                                html.H3(
                                    "Exact Force Calculator",
                                    style={"marginTop": "0", "color": "#343a40"},
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "gap": "15px",
                                        "alignItems": "flex-end",
                                    },
                                    children=[
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Area (mm²)",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "fontSize": "13px",
                                                    },
                                                ),
                                                dcc.Input(
                                                    id="calc-am",
                                                    type="number",
                                                    value=AM_INIT,
                                                    step=0.1,
                                                    style={
                                                        "width": "100%",
                                                        "padding": "6px",
                                                        "marginTop": "4px",
                                                    },
                                                ),
                                            ],
                                            style={"width": "calc(33% - 10px)"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Length (mm)",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "fontSize": "13px",
                                                    },
                                                ),
                                                dcc.Input(
                                                    id="calc-lm",
                                                    type="number",
                                                    value=LM_INIT,
                                                    step=0.1,
                                                    style={
                                                        "width": "100%",
                                                        "padding": "6px",
                                                        "marginTop": "4px",
                                                    },
                                                ),
                                            ],
                                            style={"width": "calc(33% - 10px)"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Gap (mm)",
                                                    style={
                                                        "fontWeight": "bold",
                                                        "fontSize": "13px",
                                                    },
                                                ),
                                                dcc.Input(
                                                    id="calc-lg",
                                                    type="number",
                                                    value=LG_INIT,
                                                    step=0.01,
                                                    style={
                                                        "width": "100%",
                                                        "padding": "6px",
                                                        "marginTop": "4px",
                                                    },
                                                ),
                                            ],
                                            style={"width": "calc(33% - 10px)"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="calc-output",
                                    style={
                                        "fontSize": "22px",
                                        "fontWeight": "bold",
                                        "color": "#28a745",
                                        "textAlign": "center",
                                        "marginTop": "20px",
                                        "padding": "15px",
                                        "backgroundColor": "#e9ecef",
                                        "borderRadius": "5px",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                # ==========================================
                # RIGHT COLUMN (Title & Graph)
                # ==========================================
                html.Div(
                    style={
                        "width": "65%",
                        "backgroundColor": "white",
                        "padding": "25px",
                        "borderRadius": "10px",
                        "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                    },
                    children=[
                        html.H1(
                            "Magnetic Circuit Dimension Analysis",
                            style={
                                "textAlign": "center",
                                "color": "#343a40",
                                "marginTop": "0",
                            },
                        ),
                        dcc.Graph(
                            id="main-graph", style={"height": "750px", "width": "100%"}
                        ),
                    ],
                ),
            ],
        ),
        # Hidden Data Stores
        dcc.Store(id="slider-1-var", data="am"),
        dcc.Store(id="slider-2-var", data=""),
    ],
)

# ==========================================
# 4. Interactive Callbacks
# ==========================================


# -- Axis Options Update --
@app.callback(
    [Output("dropdown-y", "options"), Output("dropdown-y", "value")],
    [Input("dropdown-x", "value")],
    [State("dropdown-y", "value")],
)
def adjust_y_options(x_val, current_y_val):
    # Base variables excluding the selected X
    filtered = [
        {"label": VAR_CONFIG[k]["label"], "value": k} for k in VAR_CONFIG if k != x_val
    ]
    # Inject "Force" into the Y-axis options
    filtered.append({"label": "Force (N)", "value": "force"})

    fallback = current_y_val if current_y_val != x_val else filtered[0]["value"]
    return filtered, fallback


# -- Sliders Setup --
@app.callback(
    [
        Output("slider-1-container", "style"),
        Output("dynamic-slider-1", "min"),
        Output("dynamic-slider-1", "max"),
        Output("dynamic-slider-1", "step"),
        Output("dynamic-slider-1", "value"),
        Output("slider-1-var", "data"),
        Output("slider-2-container", "style"),
        Output("dynamic-slider-2", "min"),
        Output("dynamic-slider-2", "max"),
        Output("dynamic-slider-2", "step"),
        Output("dynamic-slider-2", "value"),
        Output("slider-2-var", "data"),
        Output("btn-toggle-view", "style"),
    ],
    [Input("dropdown-x", "value"), Input("dropdown-y", "value")],
    [
        State("dynamic-slider-1", "value"),
        State("slider-1-var", "data"),
        State("dynamic-slider-2", "value"),
        State("slider-2-var", "data"),
    ],
)
def manage_sliders(x_var, y_var, val1, var1, val2, var2):
    base_btn_style = {
        "height": "36px",
        "padding": "0 15px",
        "fontSize": "13px",
        "fontWeight": "bold",
        "cursor": "pointer",
        "backgroundColor": "#007bff",
        "color": "white",
        "border": "none",
        "borderRadius": "4px",
        "whiteSpace": "nowrap",
        "transition": "all 0.2s",
    }

    if y_var == "force":
        # TWO Sliders active
        remaining = [v for v in VAR_CONFIG if v != x_var]
        v1, v2 = remaining[0], remaining[1]

        c1, c2 = VAR_CONFIG[v1], VAR_CONFIG[v2]

        # Persist values smoothly if variables shift around
        new_val1 = val1 if var1 == v1 else (val2 if var2 == v1 else c1["default"])
        new_val2 = val2 if var2 == v2 else (val1 if var1 == v2 else c2["default"])

        # Hide toggle button for 2D curve plot
        btn_style = {**base_btn_style, "display": "none"}

        return (
            {"display": "block"},
            c1["min"],
            c1["max"],
            c1["step"],
            new_val1,
            v1,
            {"display": "block", "marginTop": "30px"},
            c2["min"],
            c2["max"],
            c2["step"],
            new_val2,
            v2,
            btn_style,
        )
    else:
        # ONE Slider active
        remaining = [v for v in VAR_CONFIG if v not in (x_var, y_var)]
        v1 = remaining[0]
        c1 = VAR_CONFIG[v1]

        new_val1 = val1 if var1 == v1 else (val2 if var2 == v1 else c1["default"])

        # Show toggle button
        btn_style = {**base_btn_style, "display": "block"}

        return (
            {"display": "block"},
            c1["min"],
            c1["max"],
            c1["step"],
            new_val1,
            v1,
            {"display": "none"},
            0,
            1,
            0.1,
            0,
            "",
            btn_style,
        )


# -- Slider Labels --
@app.callback(
    Output("slider-1-label", "children"),
    [Input("dynamic-slider-1", "value")],
    [State("slider-1-var", "data")],
)
def update_label1(val, var):
    if not var:
        return dash.no_update
    return f"Adjust Constraint: {VAR_CONFIG[var]['label']} = {val}"


@app.callback(
    Output("slider-2-label", "children"),
    [Input("dynamic-slider-2", "value")],
    [State("slider-2-var", "data")],
)
def update_label2(val, var):
    if not var:
        return dash.no_update
    return f"Adjust Constraint: {VAR_CONFIG[var]['label']} = {val}"


# -- Main Graph Renderer --
@app.callback(
    [
        Output("main-graph", "figure"),
        Output("main-graph", "config"),
        Output("btn-toggle-view", "children"),
    ],
    [
        Input("dropdown-x", "value"),
        Input("dropdown-y", "value"),
        Input("dynamic-slider-1", "value"),
        Input("dynamic-slider-2", "value"),
        Input("btn-toggle-view", "n_clicks"),
    ],
    [State("slider-1-var", "data"), State("slider-2-var", "data")],
)
def render_analysis(x_var, y_var, val1, val2, n_clicks, var1, var2):
    if not x_var or not y_var:
        return go.Figure(), {}, "Toggle View"

    cfg_x = VAR_CONFIG[x_var]
    x_space = np.linspace(cfg_x["min"], cfg_x["max"], GRID_RESOLUTION)
    fig = go.Figure()

    # ==========================
    # 2D Curve Plot (Force vs X)
    # ==========================
    if y_var == "force":
        if val1 is None or val2 is None:
            return go.Figure(), {}, "Toggle View Mode"

        def map_inputs_1d(target):
            if target == x_var:
                return x_space * VAR_CONFIG[target]["scale"]
            if target == var1:
                return val1 * VAR_CONFIG[target]["scale"]
            if target == var2:
                return val2 * VAR_CONFIG[target]["scale"]

        Z = np.clip(
            fast_calc_force(
                map_inputs_1d("am"), map_inputs_1d("lm"), map_inputs_1d("lg")
            ),
            0,
            FORCE_CAP,
        )

        fig.add_trace(
            go.Scatter(
                x=x_space,
                y=Z,
                mode="lines",
                line=dict(color="#d62728", width=3),
                fill="tozeroy",
                fillcolor="rgba(214, 39, 40, 0.1)",
            )
        )
        fig.update_layout(
            xaxis_title=cfg_x["label"],
            yaxis_title="Force (N)",
            title=f"<b>Force Profile Over {cfg_x['label']}</b><br><sup>Constants: {VAR_CONFIG[var1]['label']} = {val1} | {VAR_CONFIG[var2]['label']} = {val2}</sup>",
            template="plotly_white",
            margin=dict(l=40, r=40, t=70, b=40),
        )

        # Dynamic filename for 2D curve (includes both constants)
        filename = f"force_vs_{x_var}_{var1}_{val1}_{var2}_{val2}_1D"
        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
                "height": 700,
                "width": 1100,
                "scale": 1,
            },
            "displaylogo": False,
        }

        return fig, config, "Switch View"

    # ==========================
    # 3D Surface / Heatmap
    # ==========================
    else:
        if val1 is None:
            return go.Figure(), {}, "Switch View"

        cfg_y = VAR_CONFIG[y_var]
        y_space = np.linspace(cfg_y["min"], cfg_y["max"], GRID_RESOLUTION)
        X_grid, Y_grid = np.meshgrid(x_space, y_space)

        def map_inputs_2d(target):
            if target == x_var:
                return X_grid * VAR_CONFIG[target]["scale"]
            if target == y_var:
                return Y_grid * VAR_CONFIG[target]["scale"]
            if target == var1:
                return val1 * VAR_CONFIG[target]["scale"]

        Z = np.clip(
            fast_calc_force(
                map_inputs_2d("am"), map_inputs_2d("lm"), map_inputs_2d("lg")
            ),
            0,
            FORCE_CAP,
        )

        view_type = "2D" if (n_clicks % 2 == 1) else "3D"
        btn_label = "Switch to 3D View" if view_type == "2D" else "Switch to 2D Heatmap"

        if view_type == "3D":
            fig.add_trace(go.Surface(z=Z, x=x_space, y=y_space, colorscale="Turbo"))
            fig.update_layout(
                scene=dict(
                    xaxis_title=cfg_x["label"],
                    yaxis_title=cfg_y["label"],
                    zaxis_title="Force (N)",
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=1.3)),
                    aspectratio=dict(x=1, y=1, z=0.75),
                )
            )
        else:
            fig.add_trace(
                go.Heatmap(
                    z=Z,
                    x=x_space,
                    y=y_space,
                    colorscale="Turbo",
                    zsmooth="best",
                    colorbar=dict(title="Force (N)"),
                )
            )
            fig.update_layout(
                xaxis_title=cfg_x["label"],
                yaxis_title=cfg_y["label"],
                template="plotly_white",
            )

        fig.update_layout(
            title=f"<b>Magnetic Force Field Response Array</b><br><sup>Constant Variable: {VAR_CONFIG[var1]['label']} = {val1}</sup>",
            title_x=0.5,
            margin=dict(l=40, r=40, t=70, b=40),
        )

        # Dynamic filename for 3D/Heatmap (includes the single constant and view type)
        filename = f"{x_var}_vs_{y_var}_{var1}_{val1}_{view_type}"
        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
                "height": 700,
                "width": 1100,
                "scale": 1,
            },
            "displaylogo": False,
        }

        return fig, config, btn_label


# -- Exact Calculator Update --
@app.callback(
    Output("calc-output", "children"),
    [Input("calc-am", "value"), Input("calc-lm", "value"), Input("calc-lg", "value")],
)
def update_exact_calculator(am_val, lm_val, lg_val):
    if None in (am_val, lm_val, lg_val):
        return "--- N"
    try:
        force = fast_calc_force(
            am_val * VAR_CONFIG["am"]["scale"],
            lm_val * VAR_CONFIG["lm"]["scale"],
            lg_val * VAR_CONFIG["lg"]["scale"],
        )
        return f"{force:,.2f} N"
    except Exception:
        return "Error"


if __name__ == "__main__":
    print("\nStarting Interactive Workspace! Go to http://127.0.0.1:8050\n")
    app.run(debug=True)
