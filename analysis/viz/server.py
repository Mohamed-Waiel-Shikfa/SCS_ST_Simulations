"""Backend for the design explorer.

Every number this serves comes from calling the real pipeline - the same
``framework.evaluate`` path the optimiser used.  Nothing is recomputed with a
simplified formula for display, because a visualisation that quietly disagrees
with the thing it visualises is worse than none.

Two consequences shape the design:

* A full-fidelity evaluation takes about a minute, so evaluations run in a
  background thread and the client polls.  Screening runs go through the same
  path, so the two can never be confused.
* Every returned value carries its fidelity.  Screening has a measured 4.2 %
  median force error (screening_study.py), which is fine for ranking and not
  fine for quoting, and the UI has to be able to say which it is showing.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import threading
import traceback
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

import plots  # noqa: E402
from driver import select_driver  # noqa: E402
from framework import (CUBE_MAX, HOLD_MIN, MARGIN_LIMIT,  # noqa: E402
                       MATERIALS, PIVOT_MIN, Design, prescreen,
                       score, stage1_magnetics, stage2_mechanics,
                       stage3_switching)
from module import build_module, face_count, pivot_angle  # noqa: E402
from module import ring_normals  # noqa: E402

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="")

DESIGN_KEYS = ("material", "circuit", "n_gon", "r_face", "d_mag", "l_mag",
               "t_steel", "r_clear", "gap", "wire_d", "v_cap", "c_cap")
CAT_KEYS = ("material", "circuit")
INT_KEYS = ("n_gon",)

# The bounds the optimiser actually searched, so the UI can show where a
# config sits relative to the explored space rather than an arbitrary range.
BOUNDS = {
    "r_face": (8e-3, 24e-3), "d_mag": (1e-3, 20e-3), "l_mag": (1e-3, 20e-3),
    "t_steel": (0.3e-3, 4e-3), "r_clear": (0.0, 3e-3),
    "gap": (0.05e-3, 0.4e-3), "wire_d": (0.15e-3, 0.6e-3),
    "v_cap": (20.0, 200.0), "c_cap": (4.7e-6, 220e-6),
}

LIMITS = dict(margin=MARGIN_LIMIT, hold_ratio=HOLD_MIN,
              pivot_ratio=PIVOT_MIN, bounding_cube=CUBE_MAX)

_jobs = {}
_jobs_lock = threading.Lock()


def parse_design(payload):
    kw = {}
    for k in DESIGN_KEYS:
        if k not in payload:
            raise ValueError(f"missing field {k}")
        v = payload[k]
        if k in CAT_KEYS:
            kw[k] = str(v)
        elif k in INT_KEYS:
            kw[k] = int(float(v))
        else:
            kw[k] = float(v)
    if kw["material"] not in MATERIALS:
        raise ValueError(f"unknown material {kw['material']}")
    if kw["circuit"] not in ("none", "potcore"):
        raise ValueError(f"unknown circuit {kw['circuit']}")
    if kw["n_gon"] < 8 or (kw["n_gon"] - 8) % 4:
        raise ValueError("n_gon must be 8 + 4k (8, 12, 16, 20)")
    if kw["d_mag"] <= 0 or kw["l_mag"] <= 0 or kw["r_face"] <= 0:
        raise ValueError("dimensions must be positive")
    return Design(**kw)


def clean(o):
    """JSON-safe: NaN and infinity are not valid JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    return o


# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/meta")
def meta():
    """Everything the UI needs to build its forms and label its axes."""
    mats = []
    for name, m in MATERIALS.items():
        fam = ("Alnico" if name.startswith("LNG")
               else "FeCrCo" if name.startswith("FeCrCo")
               else "Ferrite" if "errite" in name else "other")
        mats.append(dict(name=name, Br=m["Br"], Hcj=m["Hcj"],
                         Hcb=m.get("Hcb"), mu_rec=m.get("mu_rec"),
                         source=m.get("src", "?"), family=fam))
    mats.sort(key=lambda m: (m["family"], -m["Hcj"]))
    return jsonify(clean(dict(
        materials=mats, bounds=BOUNDS, limits=LIMITS,
        design_keys=list(DESIGN_KEYS),
        n_gon_options=[dict(n=n, faces=face_count(n),
                            pivot_deg=float(np.degrees(pivot_angle(n))))
                       for n in (8, 12, 16, 20)],
        notes=dict(
            screen="Screening mesh h = min(D,L)/6. Measured against full "
                   "fidelity over 24 designs: median 4.2 % force error, "
                   "worst 21 %, Spearman rank correlation 0.992. Good for "
                   "ranking, not for quoting.",
            normal="Full fidelity, h = min(D,L)/16. Slower by roughly 10x. "
                   "This is what any believed number must come from.",
            pivot="Energy bound: magnetic work available over the roll "
                  "divided by the gravitational barrier of lifting the "
                  "centre of mass to the vertex radius. Necessary, not "
                  "sufficient - MuJoCo is the arbiter.",
            asymmetry="F_attract / F_repel. A material effect, not geometry: "
                      "a rigid magnet pair gives 1.014.",
            margin="Worst |H|/Hcj over both states. Above ~0.8 the magnet is "
                   "being driven towards irreversible loss in service.",
        ))))


@app.route("/api/matrix/files")
def matrix_files():
    return jsonify(sorted(p.name for p in ANALYSIS.glob("*.csv")))


@app.route("/api/matrix")
def matrix():
    """The design matrix, straight from the optimiser's CSV."""
    which = request.args.get("file", "ga_front.csv")
    if "/" in which or "\\" in which or not which.endswith(".csv"):
        return jsonify(error="bad file name"), 400
    path = ANALYSIS / which
    if not path.exists():
        return jsonify(error=f"no such file: {which}"), 404

    text_cols = {"material", "circuit", "fidelity", "drv_cap", "drv_mosfet",
                 "violations"}
    rows, num = [], set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                if v is None or v == "":
                    out[k] = None
                elif k in text_cols:
                    out[k] = v
                elif k == "feasible":
                    out[k] = (v == "True")
                else:
                    try:
                        f = float(v)
                        out[k] = None if (np.isnan(f) or np.isinf(f)) else f
                        num.add(k)
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return jsonify(clean(dict(rows=rows, numeric=sorted(num),
                              file=which, n=len(rows))))


# --------------------------------------------------------------------------
def _run_dynamics(dsg, mag, mod):
    from dynamics import make_spec, run_scenario
    from pivot import run_pivot

    spec = make_spec(dsg, mag, fidelity="screen")
    ix_p = int(np.argmax(mod.normals @ np.array([1.0, 0, 0])))
    ix_m = int(np.argmin(mod.normals @ np.array([1.0, 0, 0])))

    latch = run_scenario(mod, spec, [(ix_p, ix_m, "attract")], seconds=0.25)
    repel = run_scenario(mod, spec, [(ix_p, ix_m, "repel")], seconds=0.25)
    piv = run_pivot(mod, spec, seconds=0.6, drive="repel")

    tgt = float(np.degrees(pivot_angle(dsg.n_gon)))
    settled = abs(piv[-1]["ang"])
    t_step = next((t["t"] for t in piv if abs(t["ang"]) >= tgt), None)
    return dict(
        latch=dict(t_ms=[t["t"] * 1e3 for t in latch],
                   sep_mm=[t["sep"] * 1e3 for t in latch],
                   held=bool(latch[-1]["sep"] < 2e-3),
                   final_sep_mm=latch[-1]["sep"] * 1e3),
        repel=dict(t_ms=[t["t"] * 1e3 for t in repel],
                   sep_mm=[t["sep"] * 1e3 for t in repel],
                   moved_mm=(repel[-1]["sep"] - repel[0]["sep"]) * 1e3),
        pivot=dict(t_ms=[t["t"] * 1e3 for t in piv],
                   ang_deg=[t["ang"] for t in piv],
                   z_mm=[t["z"] * 1e3 for t in piv],
                   x_mm=[t["x"] * 1e3 for t in piv],
                   target_deg=tgt, settled_deg=settled,
                   peak_deg=max(abs(t["ang"]) for t in piv),
                   steps=settled / tgt,
                   t_one_step_ms=(t_step * 1e3 if t_step else None)),
    )


def _evaluate_job(job_id, dsg, fidelity, want_dynamics):
    def note(msg, pct):
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(stage=msg, progress=pct)

    try:
        note("stage 3: switching and driver", 5)
        sw = stage3_switching(dsg)
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"],
                            n_faces=dsg.n_faces)
        ok, why = prescreen(dsg, sw, drv)

        note("stage 1: magnetics FEM (both states)", 15)
        mag = stage1_magnetics(dsg, fidelity=fidelity)

        note("stage 4: module geometry and inertia", 55)
        mod = build_module(dsg, drv if drv.feasible else None)

        note("stage 2: mechanics and pivot work", 62)
        mech = stage2_mechanics(dsg, mag, mod=mod, fidelity=fidelity)

        note("scoring", 80)
        sc = score(dsg, mag, mech, sw, drv)

        res = dict(
            design={k: getattr(dsg, k) for k in DESIGN_KEYS},
            fidelity=fidelity,
            derived=dict(n_faces=dsg.n_faces, a_face=dsg.a_face,
                         bounding_cube=dsg.bounding_cube,
                         r_vertex=dsg.r_vertex,
                         pivot_deg=float(np.degrees(pivot_angle(dsg.n_gon))),
                         ld_ratio=dsg.l_mag / dsg.d_mag),
            prescreen=dict(passed=bool(ok), reasons=list(why)),
            magnetics=dict(mag),
            mechanics={k: v for k, v in mech.items() if k != "module"},
            switching=dict(sw),
            driver=dict(feasible=bool(drv.feasible), topology=drv.topology,
                        cap_name=drv.cap_name, n_caps=drv.n_caps,
                        mosfet_name=drv.mosfet_name, n_fets=drv.n_fets,
                        charger_name=drv.charger_name, v_bank=drv.v_bank,
                        c_bank=drv.c_bank, e_bank=drv.e_bank,
                        i_peak=drv.i_peak, mass=drv.mass, volume=drv.volume,
                        price=drv.price, recharge_s=drv.recharge_s,
                        notes=drv.notes),
            module=dict(mass=mod.mass, parts=dict(mod.parts),
                        fits=bool(mod.fits), free_volume=mod.free_volume,
                        inertia=np.diag(mod.inertia).tolist(),
                        n_faces=mod.n_faces),
            verdict=dict(feasible=bool(sc["feasible"]), scalar=sc["scalar"],
                         violations=list(sc["violations"])),
            limits=LIMITS,
        )

        if want_dynamics and drv.feasible:
            note("stage 5: MuJoCo (latch, repel, pivot)", 85)
            res["dynamics"] = _run_dynamics(dsg, mag, mod)

        with _jobs_lock:
            _jobs[job_id].update(state="done", progress=100,
                                 stage="complete", result=clean(res))
    except Exception as exc:                                # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id].update(state="error", error=str(exc),
                                 trace=traceback.format_exc())


@app.route("/api/evaluate", methods=["POST"])
def start_evaluate():
    payload = request.get_json(force=True)
    try:
        dsg = parse_design(payload)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    fidelity = payload.get("fidelity", "screen")
    if fidelity not in ("screen", "normal"):
        return jsonify(error="fidelity must be 'screen' or 'normal'"), 400

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = dict(state="running", progress=0, stage="queued",
                             fidelity=fidelity)
        # keep the job table from growing without bound in a long session
        if len(_jobs) > 64:
            for k in [k for k, v in list(_jobs.items())
                      if v.get("state") != "running"][:32]:
                _jobs.pop(k, None)
    threading.Thread(target=_evaluate_job,
                     args=(job_id, dsg, fidelity,
                           bool(payload.get("dynamics", False))),
                     daemon=True).start()
    return jsonify(job=job_id)


@app.route("/api/job/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if j is None:
        return jsonify(error="no such job"), 404
    return jsonify(clean(j))


# --------------------------------------------------------------------------
@app.route("/api/geometry")
def geometry():
    """Hull vertices and face normals for the 3D module view."""
    from pivot import hull_vertices

    n = int(float(request.args.get("n_gon", 8)))
    r = float(request.args.get("r_face", 19.4e-3))
    V = hull_vertices(n, r)
    N = ring_normals(n)
    return jsonify(clean(dict(
        vertices=V.tolist(), normals=N.tolist(), n_gon=n, r_face=r,
        n_faces=len(N), a_face=2 * r * np.tan(np.pi / n),
        r_vertex=r / np.cos(np.pi / n),
        d_mag=float(request.args.get("d_mag", 4.2e-3)),
        pivot_deg=float(np.degrees(pivot_angle(n))),
        bounding_cube=2 * r)))


# --------------------------------------------------------------------------
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plots.plt.close(fig)
    buf.seek(0)
    resp = app.response_class(buf.read(), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/plot/<kind>")
def plot(kind):
    args = request.args
    try:
        if kind == "experiment":
            return _png(plots.experiment_validation())
        if kind == "demag":
            names = [s for s in args.get("materials", "").split(",") if s]
            return _png(plots.demag_curves(
                names or ["LNG37", "LNGT44", "LNGT72"],
                marks=json.loads(args.get("marks", "[]"))))

        dsg = parse_design({k: args[k] for k in DESIGN_KEYS})
        fid = args.get("fidelity", "screen")
        if kind == "field":
            return _png(plots.field_map(dsg, state=args.get("state",
                                                            "attract"),
                                        fidelity=fid))
        if kind == "force_gap":
            return _png(plots.force_vs_gap(dsg, fidelity=fid))
        if kind == "pivot":
            return _png(plots.pivot_work_plot(dsg, fidelity=fid))
        if kind == "pulse":
            return _png(plots.pulse_waveform(dsg))
        return jsonify(error=f"unknown plot: {kind}"), 404
    except KeyError as exc:
        return jsonify(error=f"missing query parameter: {exc}"), 400
    except Exception as exc:                                # noqa: BLE001
        return jsonify(error=str(exc), trace=traceback.format_exc()), 500


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5173)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    # The experiment fit is a two-minute grid search that depends on nothing
    # the user can change, so start it now rather than making the first person
    # to open the Validation tab wait for it.
    plots.warm_experiment_fit()
    print(f"design explorer:  http://{a.host}:{a.port}", flush=True)
    app.run(host=a.host, port=a.port, threaded=True, debug=False)
