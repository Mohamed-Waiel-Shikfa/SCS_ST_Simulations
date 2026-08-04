"""Backend for the design explorer.

Every number this serves comes from calling the real pipeline - the same
``framework.evaluate`` path the optimiser used.  Nothing is recomputed with a
simplified formula for display, because a visualisation that quietly disagrees
with the thing it visualises is worse than none.

Three consequences shape the design:

* A full-fidelity evaluation takes about a minute, so evaluations run in a
  background thread and the client polls.  Screening runs go through the same
  path, so the two can never be confused.
* Every returned value carries its fidelity.  Screening has a measured 4.2 %
  median force error, which is fine for ranking and not fine for quoting, and
  the UI has to be able to say which it is showing.
* The stage endpoints are served in PIPELINE ORDER - module, magnetics,
  switching, mechanics - and each takes the previous stage's output rather
  than recomputing it.  The UI cannot show a mechanics result that was not
  built on the magnetics result displayed above it.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ROOT / "simulations" / "Force_compute" / "python"))

import plots  # noqa: E402
from driver import select_driver  # noqa: E402
from framework import (CUBE_MAX, HOLD_MIN, MARGIN_LIMIT,  # noqa: E402
                       PIVOT_MIN, Design, epm_outer_radius, prescreen, score,
                       stage1_magnetics, stage2_switching, stage3_mechanics)
from materials import MATERIALS, switching_class  # noqa: E402
from module import (build_module, face_count, hull_vertices,  # noqa: E402
                    latch_faces, pivot_angle, ring_normals)
from param_info import PARAM_INFO  # noqa: E402

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="")

DESIGN_KEYS = ("material", "circuit", "n_gon", "r_face", "d_mag", "l_mag",
               "t_steel", "r_clear", "gap", "wire_d", "n_layers",
               "v_cap", "c_cap", "pulse_mode", "f_pulse", "duty", "n_pulses")
CAT_KEYS = ("material", "circuit", "pulse_mode")
INT_KEYS = ("n_gon", "n_layers", "n_pulses")

BOUNDS = {
    "r_face": (8e-3, 24e-3), "d_mag": (1e-3, 20e-3), "l_mag": (1e-3, 20e-3),
    "t_steel": (0.3e-3, 4e-3), "r_clear": (0.0, 3e-3),
    "gap": (0.05e-3, 0.4e-3), "wire_d": (0.10e-3, 0.6e-3),
    "n_layers": (1, 16), "v_cap": (20.0, 300.0), "c_cap": (4.7e-6, 220e-6),
    "f_pulse": (2e3, 120e3), "duty": (0.10, 0.90), "n_pulses": (1, 12),
}

LIMITS = dict(margin=MARGIN_LIMIT, hold_ratio=HOLD_MIN,
              pivot_ratio=PIVOT_MIN, bounding_cube=CUBE_MAX)

STAGES = [
    dict(id="module", n=0, title="Module",
         blurb="The physical assembly. Everything downstream is measured "
               "against this geometry, so it is built first."),
    dict(id="magnetics", n=1, title="Magnetics",
         blurb="Attraction, repulsion, demagnetisation margin, and the "
               "effective demagnetising factor of the real magnetic "
               "circuit."),
    dict(id="driver", n=2, title="Driver",
         blurb="The transient switching circuit, driven with the inductance "
               "and field-per-ampere the field solve measured."),
    dict(id="mechanics", n=3, title="Mechanics",
         blurb="Latching, holding and every rolling configuration. Runs only "
               "if switching succeeded."),
]

_jobs = {}
_jobs_lock = threading.Lock()
_field_cache = {}
_field_lock = threading.Lock()


def parse_design(payload):
    kw = {}
    for k in DESIGN_KEYS:
        if k not in payload:
            continue
        v = payload[k]
        if v is None or v == "":
            continue
        if k in CAT_KEYS:
            kw[k] = str(v)
        elif k in INT_KEYS:
            kw[k] = int(float(v))
        else:
            kw[k] = float(v)
    if kw.get("material", "LNGT72") not in MATERIALS:
        raise ValueError(f"unknown material {kw.get('material')}")
    if kw.get("circuit", "potcore") not in ("none", "potcore"):
        raise ValueError(f"unknown circuit {kw.get('circuit')}")
    n = kw.get("n_gon", 8)
    if n < 8 or (n - 8) % 4:
        raise ValueError("n_gon must be 8 + 4k (8, 12, 16, 20)")
    if kw.get("pulse_mode", "single") not in ("single", "train"):
        raise ValueError("pulse_mode must be 'single' or 'train'")
    for k in ("d_mag", "l_mag", "r_face", "wire_d"):
        if k in kw and kw[k] <= 0:
            raise ValueError(f"{k} must be positive")
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


def _new_job(extra=None):
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = dict(state="running", progress=0, stage="queued",
                             **(extra or {}))
        if len(_jobs) > 64:
            for k in [k for k, v in list(_jobs.items())
                      if v.get("state") != "running"][:32]:
                _jobs.pop(k, None)
    return job_id


def _finish(job_id, result):
    with _jobs_lock:
        _jobs[job_id].update(state="done", progress=100, stage="complete",
                             result=clean(result))


def _fail(job_id, exc):
    with _jobs_lock:
        _jobs[job_id].update(state="error", error=str(exc),
                             trace=traceback.format_exc())


def _driver_json(drv):
    if drv is None or not getattr(drv, "feasible", False):
        return dict(feasible=False, notes=getattr(drv, "notes", "none"))
    return dict(feasible=True, topology=drv.topology, cap_name=drv.cap_name,
                n_caps=drv.n_caps, mosfet_name=drv.mosfet_name,
                n_fets=drv.n_fets, charger_name=drv.charger_name,
                v_bank=drv.v_bank, c_bank=drv.c_bank, e_bank=drv.e_bank,
                i_peak=drv.i_peak, mass=drv.mass, volume=drv.volume,
                price=drv.price, recharge_s=drv.recharge_s,
                cap_mass=drv.cap_mass, cap_volume=drv.cap_volume,
                batt_mass=drv.batt_mass, batt_volume=drv.batt_volume,
                batt_wh=drv.batt_wh, notes=drv.notes, bom=drv.bom())


# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/meta")
def meta():
    mats = []
    for name, m in MATERIALS.items():
        mats.append(dict(name=name, Br=m["Br"], Hcj=m["Hcj"], Hcb=m["Hcb"],
                         BHmax=m["BHmax"], mu_rec=m["mu_rec"], rho=m["rho"],
                         source=m["src"], family=m["family"],
                         note=m.get("note", ""),
                         switching=switching_class(name)))
    mats.sort(key=lambda m: (m["family"], m["Hcj"]))
    return jsonify(clean(dict(
        materials=mats, bounds=BOUNDS, limits=LIMITS, stages=STAGES,
        design_keys=list(DESIGN_KEYS), param_info=PARAM_INFO,
        cat_keys=list(CAT_KEYS), int_keys=list(INT_KEYS),
        n_gon_options=[dict(n=n, faces=face_count(n), latch=6,
                            pivot_deg=float(np.degrees(pivot_angle(n))))
                       for n in (8, 12, 16, 20)],
        notes=dict(
            screen="Screening mesh h = min(D,L)/6. Measured against full "
                   "fidelity over 24 designs: median 4.2 % force error, "
                   "worst 21 %, Spearman rank correlation 0.992. Good for "
                   "ranking, not for quoting.",
            normal="Full fidelity, h = min(D,L)/16. Slower by roughly 10x. "
                   "This is what any believed number must come from.",
            fem3d="Verified against the axisymmetric FEM for magnets with no "
                  "return path: 2 % on the operating point, 2-5 % on force, "
                  "stable across discretisations at every angle. With a "
                  "steel pot core the two solvers disagree - 20 % on "
                  "attraction, 3x on repulsion - and neither is validated "
                  "against measurement for that case, so all magnitudes in "
                  "this tool come from the axisymmetric solver."))))


@app.route("/api/matrix/files")
def matrix_files():
    return jsonify(sorted(p.name for p in ANALYSIS.glob("*.csv")))


@app.route("/api/matrix")
def matrix():
    which = request.args.get("file", "ga_front.csv")
    if "/" in which or "\\" in which or not which.endswith(".csv"):
        return jsonify(error="bad file name"), 400
    path = ANALYSIS / which
    if not path.exists():
        return jsonify(error=f"no such file: {which}"), 404

    text_cols = {"material", "circuit", "fidelity", "drv_cap", "drv_mosfet",
                 "drv_topology", "violations", "pulse_mode", "pulse_program"}
    bool_cols = {"feasible", "switched", "saturated"}
    rows, num = [], set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                if v is None or v == "":
                    out[k] = None
                elif k in text_cols:
                    out[k] = v
                elif k in bool_cols:
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


# ---- Stage 0: the module, as a drawable part list -------------------------
@app.route("/api/module")
def module_view():
    try:
        dsg = parse_design(request.args)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    sw = stage2_switching(dsg, search_pulse=False)
    drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                        sw["n_turns"], sw["mmf_need"], n_faces=dsg.n_faces)
    mod = build_module(dsg, drv if drv.feasible else None)
    N = ring_normals(dsg.n_gon)
    w = dsg.winding
    return jsonify(clean(dict(
        hull=hull_vertices(dsg.n_gon, dsg.r_face).tolist(),
        normals=N.tolist(), latch_faces=latch_faces(N),
        parts=mod.parts_json(), mass=mod.mass,
        mass_by_kind=mod.mass_by_kind,
        inertia=np.diag(mod.inertia).tolist(),
        fits=bool(mod.fits), free_volume=mod.free_volume,
        used_volume=mod.used_volume, n_faces=mod.n_faces, n_gon=dsg.n_gon,
        r_face=dsg.r_face, a_face=dsg.a_face, r_vertex=dsg.r_vertex,
        bounding_cube=dsg.bounding_cube,
        pivot_deg=float(np.degrees(pivot_angle(dsg.n_gon))),
        epm_outer_radius=epm_outer_radius(dsg),
        winding=dict(layers=w.n_layers, turns_per_layer=w.turns_per_layer,
                     turns=w.n_turns, build=w.build, resistance=w.resistance,
                     mass=w.mass, wire_length=w.wire_length,
                     fill=w.fill_factor, wire_d=w.wire_d),
        driver=_driver_json(drv))))


# ---- Stage 1: the 3-D field ----------------------------------------------
@app.route("/api/field3d")
def field3d():
    """Volumetric B of two interacting EPMs at an arbitrary relative angle.

    This is the solver's own field sampled on a grid, not a redrawing of the
    axisymmetric result, and it is the only thing in the pipeline that can
    show what happens when two modules meet at an angle.
    """
    import fem3d
    from materials import material

    try:
        dsg = parse_design(request.args)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    state = request.args.get("state", "attract")
    angle = float(request.args.get("angle_deg", 0.0))
    res = int(np.clip(float(request.args.get("res", 20)), 8, 34))
    steel = request.args.get("steel", "1") not in ("0", "false")

    key = (dsg.material, dsg.d_mag, dsg.l_mag, dsg.gap, dsg.t_steel,
           dsg.r_clear, dsg.wire_d, dsg.n_layers, state, angle, res, steel)
    with _field_lock:
        if key in _field_cache:
            return jsonify(_field_cache[key])

    mat = material(dsg.material)
    states = fem3d.REPEL if state == "repel" else fem3d.ATTRACT
    with_steel = bool(steel and dsg.circuit == "potcore")
    cells = fem3d.epm_pair_cells(
        dsg.d_mag, dsg.l_mag, mat, dsg.gap,
        t_steel=dsg.t_steel if with_steel else 0.0,
        r_clear=(dsg.winding.build + dsg.r_clear) if with_steel else 0.0,
        states=states, angle=np.radians(angle), n_across=2, n_axial=4,
        n_sect=8, with_steel=with_steel)
    overlaps = len(fem3d.check_overlap(cells))
    sol = fem3d.solve3d(cells, tol=1e-6, max_iter=30)
    F, T = fem3d.force_on_body(sol, 1, 0, max_sub=10)

    pts = np.array([c.centre for c in cells])
    pad = 2.2 * dsg.d_mag
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    gx = [np.linspace(lo[i], hi[i], res) for i in range(3)]
    X, Y, Z = np.meshgrid(*gx, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    B = fem3d.sample_field(sol, P)
    Bmag = np.linalg.norm(B, axis=1)
    step = max(1, len(P) // 1600)

    Ja, Ha = sol.magnet_state(body=0)
    out = clean(dict(
        grid=dict(lo=lo.tolist(), hi=hi.tolist(), res=res),
        b=Bmag.tolist(),
        bvec=B[::step].tolist(), bvec_pos=P[::step].tolist(),
        cells=[dict(c=c.centre.tolist(), h=c.half.tolist(),
                    axes=c.axes.tolist(), kind=c.kind, body=int(c.body))
               for c in cells],
        force=F.tolist(), torque=T.tolist(),
        f_mag=float(np.linalg.norm(F)), t_mag=float(np.linalg.norm(T)),
        state=state, angle_deg=angle, overlaps=overlaps,
        converged=bool(sol.converged), residual=float(sol.residual),
        iters=int(sol.iters), n_cells=len(cells), steel=with_steel,
        j_a=Ja, h_a=Ha))
    with _field_lock:
        if len(_field_cache) > 16:
            _field_cache.clear()
        _field_cache[key] = out
    return jsonify(out)


def _angle_job(job_id, dsg, n):
    import fem3d
    from materials import material
    try:
        mat = material(dsg.material)
        tgt = float(np.degrees(pivot_angle(dsg.n_gon)))
        degs = np.linspace(0.0, max(tgt, 45.0), n)
        out = {}
        total = 2 * len(degs)
        done = 0
        for tag, states in (("attract", fem3d.ATTRACT),
                            ("repel", fem3d.REPEL)):
            rows = []
            for d in degs:
                with _jobs_lock:
                    _jobs[job_id].update(
                        stage=f"{tag} at {d:.0f} deg",
                        progress=5 + 90 * done / total)
                cells = fem3d.epm_pair_cells(dsg.d_mag, dsg.l_mag, mat,
                                             dsg.gap, states=states,
                                             angle=np.radians(d),
                                             n_across=2, n_axial=4,
                                             with_steel=False)
                sol = fem3d.solve3d(cells, tol=1e-6, max_iter=30)
                F, T = fem3d.force_on_body(sol, 1, 0, max_sub=10)
                rows.append(dict(deg=float(d), f=float(np.linalg.norm(F)),
                                 fx=float(F[0]), fz=float(F[2]),
                                 ty=float(T[1])))
                done += 1
            out[tag] = rows
        _finish(job_id, dict(sweep=out, pivot_deg=tgt))
    except Exception as exc:                                # noqa: BLE001
        _fail(job_id, exc)


@app.route("/api/angle_sweep", methods=["POST"])
def start_angle_sweep():
    payload = request.get_json(force=True)
    try:
        dsg = parse_design(payload)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    n = int(np.clip(float(payload.get("n", 7)), 3, 13))
    job_id = _new_job(dict(kind="angle_sweep"))
    threading.Thread(target=_angle_job, args=(job_id, dsg, n),
                     daemon=True).start()
    return jsonify(job=job_id)


# ---- Stage 2: the circuit -------------------------------------------------
def _circuit_job(job_id, dsg, use_fem, search):
    from circuit_sim import best_program
    from coil import circuit as mag_circuit
    try:
        n_eff, mag = None, None
        if use_fem:
            with _jobs_lock:
                _jobs[job_id].update(stage="stage 1: measuring the circuit",
                                     progress=15)
            mag = stage1_magnetics(dsg, fidelity="screen")
            n_eff = mag.get("n_eff")

        with _jobs_lock:
            _jobs[job_id].update(stage="stage 2: transient solve",
                                 progress=55)
        sw = stage2_switching(dsg, n_eff=n_eff, search_pulse=False)
        tr = sw["transient"]
        w = dsg.winding
        circ = mag_circuit(dsg.d_mag, dsg.l_mag,
                           MATERIALS[dsg.material]["mu_rec"],
                           t_steel=dsg.t_steel,
                           r_clear=w.build + dsg.r_clear, gap=dsg.gap,
                           has_steel=dsg.circuit == "potcore",
                           has_neighbour=True, n_eff=n_eff,
                           source="fem" if n_eff else "estimate")

        step = max(1, len(tr.t) // 1400)
        out = dict(
            t_us=(tr.t[::step] * 1e6).tolist(),
            i=tr.i[::step].tolist(), v_c=tr.v_c[::step].tolist(),
            h_ka=(tr.h_mag[::step] / 1e3).tolist(),
            gate=tr.gate[::step].tolist(),
            h_need_ka=sw["h_need"] / 1e3,
            summary={k: sw[k] for k in
                     ("n_turns", "n_layers", "turns_per_layer",
                      "winding_build", "R_coil", "L_coil", "n_eff",
                      "n_eff_source", "i_peak", "h_peak", "h_need",
                      "switch_margin", "switched", "saturated",
                      "b_steel_peak", "v_need", "e_bank", "e_drawn",
                      "e_resistive", "e_required", "e_total_module",
                      "t_peak", "mmf", "mmf_need", "coil_mass",
                      "fill_factor", "wire_length")},
            circuit=dict(n_eff=circ.n_eff, r_magnet=circ.r_magnet,
                         r_ext=circ.r_ext, source=circ.source,
                         reach=1.0 - circ.n_eff,
                         inductance=circ.inductance(w.n_turns)),
            magnetics=(dict(mag) if mag else None))

        if search:
            with _jobs_lock:
                _jobs[job_id].update(stage="searching pulse programmes",
                                     progress=75)
            base, best = best_program(circ, w.n_turns, w.resistance,
                                      dsg.c_cap, dsg.v_cap, sw["h_need"])
            out["search"] = dict(
                base=dict(program=base.meta["program"], e=base.e_drawn,
                          h=base.h_peak, i=base.i_peak,
                          switched=bool(base.switched)),
                best=dict(program=best.meta["program"], e=best.e_drawn,
                          h=best.h_peak, i=best.i_peak,
                          switched=bool(best.switched)),
                saving=(1.0 - best.e_drawn / max(base.e_drawn, 1e-12)))
        _finish(job_id, out)
    except Exception as exc:                                # noqa: BLE001
        _fail(job_id, exc)


@app.route("/api/circuit", methods=["POST"])
def start_circuit():
    payload = request.get_json(force=True)
    try:
        dsg = parse_design(payload)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    job_id = _new_job(dict(kind="circuit"))
    threading.Thread(
        target=_circuit_job,
        args=(job_id, dsg, bool(payload.get("fem", True)),
              bool(payload.get("search", False))), daemon=True).start()
    return jsonify(job=job_id)


# ---- Stage 3: rolling -----------------------------------------------------
def _rolling_job(job_id, dsg, drives, seconds):
    try:
        from dynamics import make_spec
        from rolling import CONFIGURATIONS, run_configuration

        with _jobs_lock:
            _jobs[job_id].update(stage="stage 1: magnetics", progress=5)
        mag = stage1_magnetics(dsg, fidelity="screen")
        with _jobs_lock:
            _jobs[job_id].update(stage="stage 2: switching", progress=20)
        sw = stage2_switching(dsg, n_eff=mag.get("n_eff"), search_pulse=False)

        if not sw["switched"]:
            _finish(job_id, dict(
                skipped=True,
                reason=(f"Switching failed: the coil reaches "
                        f"{sw['h_peak']/1e3:.0f} kA/m of the "
                        f"{sw['h_need']/1e3:.0f} kA/m needed to reverse the "
                        f"magnet. Mechanics is not run for a design that "
                        f"cannot switch - that is the gate that makes the "
                        f"search affordable."),
                switching={k: sw[k] for k in ("h_peak", "h_need", "i_peak",
                                              "switched", "v_need")}))
            return

        with _jobs_lock:
            _jobs[job_id].update(stage="stage 0: module", progress=30)
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"],
                            n_faces=dsg.n_faces)
        mod = build_module(dsg, drv if drv.feasible else None)
        spec = make_spec(dsg, mag, fidelity="screen")

        runs = {}
        total = max(len(drives) * len(CONFIGURATIONS), 1)
        done = 0
        for drive in drives:
            for name in CONFIGURATIONS:
                with _jobs_lock:
                    _jobs[job_id].update(
                        stage=f"stage 3: {name} / {drive}",
                        progress=35 + 60 * done / total)
                r = run_configuration(mod, spec, name, seconds=seconds,
                                      drive=drive)
                runs[f"{drive}|{name}"] = dict(
                    name=r.name, drive=r.drive, note=r.note,
                    verdict=r.verdict(), target_deg=r.target_deg,
                    peak_deg=r.peak_deg, settled_deg=r.settled_deg,
                    steps=r.steps, completed=r.completed,
                    overshot=r.overshot, detached=r.detached,
                    max_sep=r.max_sep, final_sep=r.final_sep,
                    t_complete=r.t_complete, frames=r.trace)
                done += 1

        mech = stage3_mechanics(dsg, mag, mod=mod, fidelity="screen")
        _finish(job_id, dict(
            skipped=False, runs=runs,
            hull=hull_vertices(dsg.n_gon, dsg.r_face).tolist(),
            r_face=dsg.r_face, n_gon=dsg.n_gon,
            mechanics={k: v for k, v in mech.items() if k != "module"},
            limits=LIMITS))
    except Exception as exc:                                # noqa: BLE001
        _fail(job_id, exc)


@app.route("/api/rolling", methods=["POST"])
def start_rolling():
    payload = request.get_json(force=True)
    try:
        dsg = parse_design(payload)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    drives = payload.get("drives") or ["push_off"]
    seconds = float(np.clip(float(payload.get("seconds", 0.8)), 0.1, 3.0))
    job_id = _new_job(dict(kind="rolling"))
    threading.Thread(target=_rolling_job,
                     args=(job_id, dsg, drives, seconds),
                     daemon=True).start()
    return jsonify(job=job_id)


# ---- the local optimiser, on demand ---------------------------------------
def _refine_job(job_id, genome, budget, fidelity):
    try:
        import optimise as O
        from framework import evaluate

        seen, n = {}, [0]

        def ev_many(gs):
            out = []
            for g in gs:
                k = O.genome_key(g)
                if k not in seen:
                    try:
                        seen[k] = evaluate(O.to_design(g), fidelity=fidelity)
                    except Exception as exc:               # noqa: BLE001
                        row = O.to_design(g).as_row()
                        row.update(feasible=False, scalar=0.0,
                                   violations=f"eval failed: {exc}")
                        seen[k] = row
                    n[0] += 1
                    with _jobs_lock:
                        if job_id in _jobs:
                            _jobs[job_id].update(
                                progress=min(95, 100 * n[0] / max(budget, 1)),
                                stage=f"{n[0]} of {budget} evaluations")
                out.append(seen[k])
            return out

        g, m, used, hist = O.local_refine(genome, ev_many, budget=budget)
        row = ev_many([g])[0]
        _finish(job_id, dict(genome=g, merit=m, evals=used, history=hist,
                             row=row,
                             design={k: getattr(O.to_design(g), k)
                                     for k in DESIGN_KEYS}))
    except Exception as exc:                                # noqa: BLE001
        _fail(job_id, exc)


@app.route("/api/refine", methods=["POST"])
def start_refine():
    payload = request.get_json(force=True)
    if "genome" not in payload:
        return jsonify(error="need a genome to refine"), 400
    budget = int(np.clip(float(payload.get("budget", 40)), 4, 400))
    fidelity = payload.get("fidelity", "screen")
    job_id = _new_job(dict(kind="refine"))
    threading.Thread(target=_refine_job,
                     args=(job_id, payload["genome"], budget, fidelity),
                     daemon=True).start()
    return jsonify(job=job_id)


@app.route("/api/genome", methods=["POST"])
def design_to_genome():
    """Best-effort inverse of the genome decoding, so a design loaded from the
    matrix can be handed to the local optimiser.

    The round trip is returned alongside so the UI can show how faithful it
    is: the encoding is not injective - several genomes decode to the same
    design - so this recovers a genome that decodes to something very close,
    not necessarily the one the GA held.
    """
    import optimise as O
    payload = request.get_json(force=True)
    try:
        d = parse_design(payload)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400

    build = d.winding.build
    r_out = d.d_mag / 2 + build + (d.r_clear + d.t_steel
                                  if d.circuit == "potcore" else 0.0)
    avail = max(r_out - build, 1e-4)
    g = dict(material=d.material, circuit=d.circuit, n_gon=int(d.n_gon),
             pulse_mode=d.pulse_mode, r_face=float(d.r_face),
             d_frac=float(r_out / (0.5 * 0.92 * d.a_face)),
             l_frac=float((d.l_mag + (d.t_steel if d.circuit == "potcore"
                                      else 0.0)) / (0.85 * d.r_face)),
             f_clear=float(d.r_clear / avail),
             f_steel=float(d.t_steel / avail),
             gap=float(d.gap), wire_d=float(d.wire_d),
             n_layers=int(d.n_layers), v_cap=float(d.v_cap),
             c_cap=float(d.c_cap), f_pulse=float(d.f_pulse),
             duty=float(d.duty), n_pulses=int(d.n_pulses))
    for spec in O.GENOME:
        if spec[1] == "cat":
            continue
        name, kind, lo, hi = spec
        g[name] = O._clip(name, kind, g[name], lo, hi)
    return jsonify(clean(dict(genome=g,
                              roundtrip={k: getattr(O.to_design(g), k)
                                         for k in DESIGN_KEYS})))


# ---- full evaluation ------------------------------------------------------
def _evaluate_job(job_id, dsg, fidelity):
    def note(msg, pct):
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(stage=msg, progress=pct)

    try:
        note("stage 0: module geometry", 5)
        build_module(dsg)

        note("stage 2 (estimated): switching and driver", 12)
        sw = stage2_switching(dsg, search_pulse=False)
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"],
                            n_faces=dsg.n_faces)
        ok, why = prescreen(dsg, sw, drv)

        note("stage 1: magnetics FEM, both states", 20)
        mag = stage1_magnetics(dsg, fidelity=fidelity)

        note("stage 2: switching on the measured circuit", 55)
        sw = stage2_switching(dsg, n_eff=mag.get("n_eff"), search_pulse=False)
        drv = select_driver(sw["v_need"], sw["L_coil"], sw["R_coil"],
                            sw["n_turns"], sw["mmf_need"],
                            n_faces=dsg.n_faces)
        mod = build_module(dsg, drv if drv.feasible else None)

        gated = not sw["switched"]
        if gated:
            mech = dict(m_module=mod.mass, hold_ratio=0.0, pivot_ratio=0.0,
                        E_barrier=float("nan"), W_drive=0.0, fits=mod.fits)
            sc = dict(feasible=False, scalar=0.0, violations=[
                f"coil reaches only {sw['h_peak']/1e3:.0f} kA/m of the "
                f"{sw['h_need']/1e3:.0f} kA/m needed to switch"])
        else:
            note("stage 3: mechanics and pivot work", 68)
            mech = stage3_mechanics(dsg, mag, mod=mod, fidelity=fidelity)
            note("scoring", 88)
            sc = score(dsg, mag, mech, sw, drv)

        _finish(job_id, dict(
            design={k: getattr(dsg, k) for k in DESIGN_KEYS},
            fidelity=fidelity, gated=gated,
            derived=dict(n_faces=dsg.n_faces, a_face=dsg.a_face,
                         bounding_cube=dsg.bounding_cube,
                         r_vertex=dsg.r_vertex, n_latch=6,
                         pivot_deg=float(np.degrees(pivot_angle(dsg.n_gon))),
                         epm_r_out=epm_outer_radius(dsg),
                         ld_ratio=dsg.l_mag / dsg.d_mag),
            prescreen=dict(passed=bool(ok), reasons=list(why)),
            magnetics=dict(mag),
            switching={k: v for k, v in sw.items() if k != "transient"},
            mechanics={k: v for k, v in mech.items() if k != "module"},
            driver=_driver_json(drv),
            module=dict(mass=mod.mass, parts=mod.parts_json(),
                        mass_by_kind=mod.mass_by_kind, fits=bool(mod.fits),
                        free_volume=mod.free_volume,
                        used_volume=mod.used_volume,
                        inertia=np.diag(mod.inertia).tolist(),
                        n_faces=mod.n_faces, latch_faces=mod.latch,
                        hull=hull_vertices(dsg.n_gon, dsg.r_face).tolist()),
            verdict=dict(feasible=bool(sc["feasible"]), scalar=sc["scalar"],
                         violations=list(sc["violations"])),
            limits=LIMITS))
    except Exception as exc:                                # noqa: BLE001
        _fail(job_id, exc)


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
    job_id = _new_job(dict(kind="evaluate", fidelity=fidelity))
    threading.Thread(target=_evaluate_job, args=(job_id, dsg, fidelity),
                     daemon=True).start()
    return jsonify(job=job_id)


@app.route("/api/job/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if j is None:
        return jsonify(error="no such job"), 404
    return jsonify(clean(j))


# ---- matplotlib figures ---------------------------------------------------
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
                names or ["LNG37", "LNGT44", "FerriteY30"],
                marks=json.loads(args.get("marks", "[]"))))

        dsg = parse_design(args)
        fid = args.get("fidelity", "screen")
        if kind == "field":
            return _png(plots.field_map(dsg, state=args.get("state",
                                                            "attract"),
                                        fidelity=fid))
        if kind == "force_gap":
            return _png(plots.force_vs_gap(dsg, fidelity=fid))
        if kind == "pivot":
            return _png(plots.pivot_work_plot(dsg, fidelity=fid))
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
    plots.warm_experiment_fit()
    print(f"design explorer:  http://{a.host}:{a.port}", flush=True)
    app.run(host=a.host, port=a.port, threaded=True, debug=False)
