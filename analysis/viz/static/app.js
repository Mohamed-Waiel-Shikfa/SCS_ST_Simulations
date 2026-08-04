/* Design explorer.
 *
 * One rule shapes this file: every number shown comes from an endpoint that
 * called the real pipeline.  Nothing is recomputed here with a simplified
 * formula, because a viewer that quietly disagrees with the thing it views is
 * worse than no viewer.
 */
import { Scene3D, ramp, quatRotate, add, mul, frameFrom } from './scene3d.js';

const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));

const S = {
  meta: null, matrix: null, design: null, result: null, genome: null,
  sortKey: 'scalar', sortDir: -1, selected: null,
};

/* ==================================================================== util */
const fmt = (v, unit) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '&mdash;';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v !== 'number') return String(v);
  const scales = {
    mm: [1e3, 'mm', 2], mm3: [1e3, 'mm', 3], g: [1e3, 'g', 1],
    cc: [1e6, 'cc', 2], N: [1, 'N', 3], mJ: [1e3, 'mJ', 1],
    uH: [1e6, '&micro;H', 1], kA: [1e-3, 'kA/m', 0], A: [1, 'A', 1],
    V: [1, 'V', 0], T: [1, 'T', 3], us: [1e6, '&micro;s', 0],
    ohm: [1, '&Omega;', 3], uF: [1e6, '&micro;F', 1], mWb: [1e6, '', 2],
    pct: [100, '%', 1], deg: [1, '&deg;', 1], m: [1, 'm', 2],
  };
  if (unit && scales[unit]) {
    const [k, u, d] = scales[unit];
    return `${(v * k).toFixed(d)} ${u}`;
  }
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : a < 100 ? 3 : 1);
};

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function rows(pairs) {
  return `<dl class="kv">${pairs.map(([k, v, info]) =>
    `<dt${info ? ` data-tip="${esc(info)}"` : ''}>${k}</dt><dd>${v}</dd>`)
    .join('')}</dl>`;
}

async function getJSON(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

/* Poll a background job to completion. */
async function runJob(url, body, onStage) {
  const { job } = await postJSON(url, body);
  for (;;) {
    await new Promise(r => setTimeout(r, 400));
    const s = await getJSON(`/api/job/${job}`);
    if (onStage) onStage(s.stage || '', s.progress || 0);
    if (s.state === 'done') return s.result;
    if (s.state === 'error') throw new Error(s.error);
  }
}

/* ================================================================ tooltips */
const tip = $('#tip');
function bindTips(root = document) {
  root.querySelectorAll('[data-tip]').forEach(el => {
    if (el._tipBound) return;
    el._tipBound = true;
    el.addEventListener('mouseenter', () => {
      tip.innerHTML = el.dataset.tip;
      tip.classList.remove('hidden');
    });
    el.addEventListener('mousemove', e => {
      const w = tip.offsetWidth, h = tip.offsetHeight;
      let x = e.clientX + 16, y = e.clientY + 16;
      if (x + w > innerWidth - 8) x = e.clientX - w - 16;
      if (y + h > innerHeight - 8) y = e.clientY - h - 16;
      tip.style.left = `${x}px`;
      tip.style.top = `${y}px`;
    });
    el.addEventListener('mouseleave', () => tip.classList.add('hidden'));
  });
}

function tipFor(key) {
  const p = S.meta?.param_info?.[key];
  if (!p) return '';
  return `<b>${key}</b><br><i>what</i> ${esc(p.what)}<br>` +
    `<i>effect</i> ${esc(p.effect)}<br><i>cost</i> ${esc(p.cost)}`;
}

/* ==================================================================== tabs */
$$('#tabs button').forEach(b => b.addEventListener('click', () => {
  $$('#tabs button').forEach(x => x.classList.remove('active'));
  $$('.tab').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  $(`#tab-${b.dataset.tab}`).classList.add('active');
  onTabShown(b.dataset.tab);
}));

function showTab(name) {
  const b = $(`#tabs button[data-tab="${name}"]`);
  if (b) b.click();
}

function onTabShown(name) {
  if (name === 'module') drawModule();
  if (name === 'magnetics') refreshMagnetics();
  if (name === 'mechanics') refreshPivotPlot();
  if (name === 'validation') {
    $('#val-img').src = '/api/plot/experiment';
  }
  if (viewers[name]) viewers[name].forEach(v => v.draw());
}
const viewers = {};

/* =================================================================== boot */
(async function boot() {
  S.meta = await getJSON('/api/meta');
  $('#subtitle').textContent =
    `${S.meta.materials.length} magnet grades \u00b7 pipeline: module \u2192 ` +
    `magnetics \u2192 switching \u2192 mechanics \u00b7 mechanics is skipped ` +
    `when switching fails`;
  buildForm();
  buildMaterialTable();
  await loadMatrixFiles();
  bindTips();
})().catch(e => {
  $('#subtitle').textContent = `failed to load: ${e.message}`;
});

/* ============================================================ config form */
const FIELD_GROUPS = [
  ['Geometry', ['n_gon', 'r_face', 'd_mag', 'l_mag', 'gap']],
  ['Magnetic circuit', ['material', 'circuit', 't_steel', 'r_clear']],
  ['Winding', ['wire_d', 'n_layers']],
  ['Drive', ['v_cap', 'c_cap', 'pulse_mode', 'f_pulse', 'duty', 'n_pulses']],
];

const UNITS = {
  r_face: 'mm', d_mag: 'mm', l_mag: 'mm', gap: 'mm', t_steel: 'mm',
  r_clear: 'mm', wire_d: 'mm', c_cap: 'uF', v_cap: 'V', f_pulse: 'kHz',
};
const UNIT_SCALE = { mm: 1e3, uF: 1e6, kHz: 1e-3, V: 1 };

function buildForm() {
  const f = $('#cfg-form');
  f.innerHTML = FIELD_GROUPS.map(([title, keys]) => `
    <fieldset><legend>${title}</legend>${keys.map(k => field(k)).join('')}
    </fieldset>`).join('');
  bindTips(f);
  f.addEventListener('input', () => { S.dirty = true; });
  $('#cfg-reset').addEventListener('click', () => { buildForm(); });
  $('#cfg-run').addEventListener('click', runEvaluate);
  $('#cfg-refine').addEventListener('click', runRefine);
}

function field(k) {
  const info = tipFor(k);
  const lab = `<label data-tip="${esc(info)}">${k}${
    UNITS[k] ? ` <span class="unit">${UNITS[k]}</span>` : ''}</label>`;
  if (k === 'material') {
    const byFam = {};
    S.meta.materials.forEach(m => {
      (byFam[m.family] = byFam[m.family] || []).push(m);
    });
    const opts = Object.entries(byFam).map(([fam, ms]) =>
      `<optgroup label="${fam}">${ms.map(m =>
        `<option value="${m.name}"${m.name === 'LNGT44' ? ' selected' : ''}>` +
        `${m.name} \u00b7 Br ${m.Br.toFixed(2)} T \u00b7 Hcj ` +
        `${(m.Hcj / 1e3).toFixed(0)} kA/m \u00b7 ${m.switching}` +
        `${m.source === 'lit' ? ' (lit)' : ''}</option>`).join('')}</optgroup>`)
      .join('');
    return `<div class="f">${lab}<select name="material">${opts}</select></div>`;
  }
  if (k === 'circuit') {
    return `<div class="f">${lab}<select name="circuit">
      <option value="potcore" selected>pot core (steel return path)</option>
      <option value="none">bare rod</option></select></div>`;
  }
  if (k === 'pulse_mode') {
    return `<div class="f">${lab}<select name="pulse_mode">
      <option value="single" selected>single shot</option>
      <option value="train">pulse train</option></select></div>`;
  }
  if (k === 'n_gon') {
    return `<div class="f">${lab}<select name="n_gon">${
      S.meta.n_gon_options.map(o =>
        `<option value="${o.n}"${o.n === 8 ? ' selected' : ''}>n=${o.n} \u00b7 ` +
        `${o.faces} faces \u00b7 6 latching \u00b7 pivot ` +
        `${o.pivot_deg.toFixed(1)}\u00b0</option>`).join('')}</select></div>`;
  }
  const defaults = {
    r_face: 19.4e-3, d_mag: 4.2e-3, l_mag: 8.4e-3, gap: 0.1e-3,
    t_steel: 1.0e-3, r_clear: 0.4e-3, wire_d: 0.25e-3, n_layers: 6,
    v_cap: 120, c_cap: 47e-6, f_pulse: 20e3, duty: 0.5, n_pulses: 4,
  };
  const u = UNITS[k], s = u ? UNIT_SCALE[u] : 1;
  const b = S.meta.bounds[k];
  const step = k === 'n_layers' || k === 'n_pulses' ? 1
    : (u === 'mm' ? 0.01 : u === 'uF' ? 0.1 : u === 'kHz' ? 0.5 : 0.01);
  return `<div class="f">${lab}<input type="number" name="${k}"
    value="${(defaults[k] * s).toFixed(step >= 1 ? 0 : 3)}" step="${step}"
    ${b ? `data-lo="${b[0] * s}" data-hi="${b[1] * s}"` : ''}></div>`;
}

function readForm() {
  const d = {};
  new FormData($('#cfg-form')).forEach((v, k) => {
    const u = UNITS[k];
    d[k] = (S.meta.cat_keys.includes(k)) ? v
      : (u ? parseFloat(v) / UNIT_SCALE[u] : parseFloat(v));
  });
  return d;
}

function writeForm(d) {
  Object.entries(d).forEach(([k, v]) => {
    const el = $(`#cfg-form [name="${k}"]`);
    if (!el || v === null || v === undefined) return;
    const u = UNITS[k];
    el.value = (typeof v === 'number' && u) ? (v * UNIT_SCALE[u]).toFixed(4)
      : v;
  });
}

function designQuery(d) {
  return Object.entries(d || readForm())
    .map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
}

/* ============================================================== evaluate */
function progress(on, stage, pct) {
  $('#cfg-progress').classList.toggle('hidden', !on);
  $('#cfg-stage').textContent = stage || '';
  $('#cfg-bar').style.width = `${pct || 0}%`;
}

async function runEvaluate() {
  const d = readForm();
  S.design = d;
  progress(true, 'starting', 2);
  $('#cfg-badge').textContent = 'running';
  $('#cfg-badge').className = 'pill';
  try {
    const res = await runJob('/api/evaluate',
      { ...d, fidelity: $('#cfg-fidelity').value },
      (s, p) => progress(true, s, p));
    S.result = res;
    renderResult(res);
    drawModule();
    renderDriverBom(res);
    progress(false);
  } catch (e) {
    progress(false);
    $('#cfg-badge').textContent = 'error';
    $('#cfg-badge').className = 'pill bad';
    $('#cfg-result').innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
}

function renderResult(r) {
  const v = r.verdict;
  $('#cfg-badge').textContent = v.feasible ? 'feasible'
    : (r.gated ? 'gated: switching failed' : 'infeasible');
  $('#cfg-badge').className = `pill ${v.feasible ? 'good' : 'bad'}`;

  const g = (k, val, unit) => [k, fmt(val, unit), tipFor(k)];
  const m = r.magnetics, sw = r.switching, me = r.mechanics, mo = r.module;

  let html = `<div class="fid">fidelity: <b>${r.fidelity}</b> &mdash; ${
    esc(S.meta.notes[r.fidelity] || '')}</div>`;

  if (!v.feasible) {
    html += `<div class="viol"><b>${v.violations.length} violation${
      v.violations.length === 1 ? '' : 's'}</b><ul>${
      v.violations.map(x => `<li>${esc(x)}</li>`).join('')}</ul></div>`;
  }
  if (r.gated) {
    html += `<p class="note">Mechanics was not run. A module whose coil
      cannot reverse its magnet is not a robot however well it holds, so the
      pipeline stops here &mdash; that gate is where most of the search time
      is saved.</p>`;
  }

  html += `<h3>Stage 0 &mdash; module</h3>` + rows([
    ['mass', fmt(mo.mass, 'g'), tipFor('m_module')],
    ['faces', `${mo.n_faces} (6 latching)`, tipFor('n_gon')],
    ['bounding cube', fmt(r.derived.bounding_cube, 'mm')],
    ['free volume', fmt(mo.free_volume, 'cc'), tipFor('free_volume')],
    ['electronics need', fmt(mo.used_volume, 'cc')],
    ['fits', mo.fits ? 'yes' : '<b class="bad">no</b>'],
  ]);

  html += `<h3>Stage 1 &mdash; magnetics</h3>` + rows([
    g('F_attract', m.F_attract, 'N'), g('F_repel', m.F_repel, 'N'),
    g('asymmetry', m.asymmetry), g('J_attract', m.J_attract, 'T'),
    g('J_repel', m.J_repel, 'T'), g('margin', m.margin),
    g('n_eff', m.n_eff),
  ]);

  html += `<h3>Stage 2 &mdash; switching</h3>` + rows([
    ['winding', `${sw.n_layers} layers \u00d7 ${sw.turns_per_layer} = ` +
      `${sw.n_turns} turns`, tipFor('n_layers')],
    ['coil resistance', fmt(sw.R_coil, 'ohm')],
    ['inductance', fmt(sw.L_coil, 'uH')],
    ['peak current', fmt(sw.i_peak, 'A')],
    g('h_peak', sw.h_peak, 'kA'),
    ['threshold', fmt(sw.h_need, 'kA')],
    ['margin', `${(sw.switch_margin || 0).toFixed(2)}\u00d7`],
    ['switched', sw.switched ? '<b class="good">yes</b>'
      : '<b class="bad">no</b>'],
    g('b_steel_peak', sw.b_steel_peak, 'T'),
    ['energy drawn', fmt(sw.e_drawn, 'mJ')],
    g('e_switch', sw.e_total_module, 'mJ'),
  ]);

  if (!r.gated) {
    html += `<h3>Stage 3 &mdash; mechanics</h3>` + rows([
      g('hold_ratio', me.hold_ratio), g('pivot_ratio', me.pivot_ratio),
      ['barrier', fmt(me.E_barrier, 'mJ')],
      ['drive work', fmt(me.W_drive, 'mJ')],
    ]);
  }
  html += `<h3>Score</h3>` + rows([g('scalar', v.scalar)]);
  $('#cfg-result').innerHTML = html;
  $('#cfg-result').classList.remove('empty');
  bindTips($('#cfg-result'));
}

/* ======================================================== local optimiser */
async function runRefine() {
  const d = readForm();
  const out = $('#cfg-refine-out');
  out.classList.remove('hidden');
  out.innerHTML = '<p>encoding design&hellip;</p>';
  try {
    const enc = await postJSON('/api/genome', d);
    out.innerHTML = '<p>running compass search&hellip;</p>';
    const res = await runJob('/api/refine', {
      genome: enc.genome,
      budget: parseInt($('#cfg-budget').value, 10) || 40,
      fidelity: $('#cfg-fidelity').value,
    }, (s, p) => progress(true, `local optimiser: ${s}`, p));
    progress(false);
    S.genome = res.genome;
    const before = enc.roundtrip, after = res.design;
    const changed = Object.keys(after).filter(k =>
      typeof after[k] === 'number'
        ? Math.abs(after[k] - before[k]) > 1e-9 * Math.max(1, Math.abs(before[k]))
        : after[k] !== before[k]);
    out.innerHTML = `
      <h3>Local optimiser</h3>
      <p>${res.evals} evaluations, merit <b>${res.merit.toFixed(4)}</b>.
      ${changed.length} parameter${changed.length === 1 ? '' : 's'} moved.</p>
      ${rows(changed.map(k => [k,
        `${fmt(before[k], UNITS[k])} &rarr; <b>${fmt(after[k], UNITS[k])}</b>`,
        tipFor(k)]))}
      <button id="rf-apply" class="primary" type="button">Apply and evaluate</button>`;
    bindTips(out);
    $('#rf-apply').addEventListener('click', () => {
      writeForm(after);
      runEvaluate();
    });
  } catch (e) {
    progress(false);
    out.innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
}

/* ============================================================ stage 0: 3D */
const PART_KINDS = ['magnet', 'coil', 'steel', 'cap', 'battery', 'pcb',
  'shell', 'electronics'];
const KIND_ON = Object.fromEntries(PART_KINDS.map(k =>
  [k, k !== 'electronics']));
const CUTAWAY = { on: true };

let modScene = null;

function drawModule() {
  const canvas = $('#mod-canvas');
  if (!canvas) return;
  if (!S.result?.module?.parts) {
    fetch(`/api/module?${designQuery()}`).then(r => r.json())
      .then(j => { if (!j.error) { renderModule(j, j.parts); } })
      .catch(() => {});
    return;
  }
  renderModule(S.result.module, S.result.module.parts);
}

function renderModule(mod, parts) {
  const canvas = $('#mod-canvas');
  if (!modScene) {
    modScene = new Scene3D(canvas);
    viewers.module = [modScene];
  }
  modScene.clear();
  const hull = mod.hull || S.result?.module?.hull;
  const cut = CUTAWAY.on;
  // A cutaway keeps only the parts on one side of the y = 0 plane, which is
  // the only way to actually SEE the magnets, coils and capacitor: in the
  // assembled module every one of them is buried behind a steel pole cup.
  const keep = p => !cut || p.centre[1] <= 1e-9;

  if (hull && KIND_ON.shell) {
    modScene.addHull(hull, '#39435a',
      { alpha: cut ? 0.10 : 0.26, edge: 'rgba(160,180,210,.30)' });
  } else if (hull) {
    modScene.addHull(hull, '#39435a',
      { alpha: 0.05, edge: 'rgba(150,170,200,.22)' });
  }
  for (const p of parts) {
    if (!KIND_ON[p.kind] || !keep(p)) continue;
    const c = p.centre, ax = p.axis, d = p.dims;
    const alpha = p.kind === 'steel' ? 0.9 : 1;
    if (p.shape === 'cylinder') {
      modScene.addCylinder(c, ax, d.r, d.h, p.colour, { seg: 22, alpha });
    } else if (p.shape === 'tube') {
      modScene.addTube(c, ax, d.r_in, d.r_out, d.h, p.colour,
        { seg: 22, alpha });
    } else if (p.shape === 'box') {
      modScene.addBox(c, frameFrom(ax), [d.a / 2, d.b / 2, d.c / 2],
        p.colour, { alpha });
    }
  }

  // mark the six latching faces with a thin ring, not a disc: they are
  // annotation, and a filled disc drowned the components it was annotating
  const R = mod.r_face || 0.02;
  const rEpm = mod.epm_outer_radius || 0.004;
  (mod.latch_faces || []).forEach(k => {
    const n = (mod.normals || [])[k];
    if (!n) return;
    modScene.addTube(mul(n, R * 1.005), n, rEpm * 1.12, rEpm * 1.3,
      R * 0.010, '#6cd39a', { seg: 24, alpha: 0.95 });
  });

  modScene.fit(hull || [[0, 0, 0], [R, R, R]]);
  modScene.draw();
  renderModuleToggles();
  renderModuleInfo(mod);
}

function renderModuleToggles() {
  const el = $('#mod-toggles');
  if (el.dataset.built) return;
  el.dataset.built = '1';
  el.innerHTML =
    `<label class="chip"><input type="checkbox" data-cut="1"${
      CUTAWAY.on ? ' checked' : ''}> cutaway</label>` +
    PART_KINDS.map(k =>
      `<label class="chip"><input type="checkbox" data-kind="${k}"${
        KIND_ON[k] ? ' checked' : ''}> ${k}</label>`).join('');
  el.addEventListener('change', e => {
    if (e.target.dataset.cut) CUTAWAY.on = e.target.checked;
    else KIND_ON[e.target.dataset.kind] = e.target.checked;
    drawModule();
  });
}

function renderModuleInfo(mod) {
  const w = mod.winding || {};
  const byKind = mod.mass_by_kind || {};
  const total = mod.mass || 1;
  const bars = Object.entries(byKind).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<div class="massrow"><span>${k}</span>
      <div class="massbar"><i style="width:${(v / total * 100).toFixed(1)}%"></i></div>
      <b>${fmt(v, 'g')}</b><span class="pct">${(v / total * 100).toFixed(1)}%</span></div>`)
    .join('');
  $('#mod-info').classList.remove('empty');
  $('#mod-info').innerHTML = `
    ${rows([
      ['total mass', fmt(mod.mass, 'g'), tipFor('m_module')],
      ['faces', `${mod.n_faces} total, 6 latching`, tipFor('n_gon')],
      ['pivot angle', fmt(mod.pivot_deg, 'deg'), tipFor('n_gon')],
      ['bounding cube', fmt(mod.bounding_cube, 'mm')],
      ['EPM outer radius', fmt(mod.epm_outer_radius, 'mm')],
      ['free volume', fmt(mod.free_volume, 'cc'), tipFor('free_volume')],
      ['electronics need', fmt(mod.used_volume, 'cc')],
      ['fits', mod.fits ? '<b class="good">yes</b>' : '<b class="bad">no</b>'],
    ])}
    <h3>Winding</h3>
    ${rows([
      ['layers', `${w.layers} \u00d7 ${w.turns_per_layer} turns = ${w.turns}`,
        tipFor('n_layers')],
      ['wire', fmt(w.wire_d, 'mm'), tipFor('wire_d')],
      ['build', fmt(w.build, 'mm')],
      ['wire length', fmt(w.wire_length, 'm')],
      ['resistance', fmt(w.resistance, 'ohm')],
      ['copper mass', fmt(w.mass, 'g')],
      ['fill factor', (w.fill || 0).toFixed(2)],
    ])}
    <h3>Mass budget</h3><div class="masschart">${bars}</div>`;
  bindTips($('#mod-info'));
}

/* ========================================================= stage 1: field */
let f3Scene = null, f3Data = null;

function refreshMagnetics() {
  const q = designQuery();
  $('#mag-field').src = `/api/plot/field?${q}&state=${$('#mag-state').value}`;
  $('#mag-forcegap').src = `/api/plot/force_gap?${q}`;
}
$('#mag-state').addEventListener('change', refreshMagnetics);
$('#f3-angle').addEventListener('input', e => {
  $('#f3-angle-v').textContent = `${e.target.value}\u00b0`;
});
$('#f3-mode').addEventListener('change', () => drawField());
$('#f3-run').addEventListener('click', async () => {
  const q = designQuery() +
    `&state=${$('#f3-state').value}&angle_deg=${$('#f3-angle').value}` +
    `&steel=${$('#f3-steel').checked ? 1 : 0}`;
  $('#f3-stage').textContent = 'solving\u2026';
  try {
    f3Data = await getJSON(`/api/field3d?${q}`);
    $('#f3-stage').textContent = '';
    drawField();
  } catch (e) {
    $('#f3-stage').textContent = '';
    $('#f3-info').innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
});

function drawField() {
  if (!f3Data) return;
  const canvas = $('#f3-canvas');
  if (!f3Scene) {
    f3Scene = new Scene3D(canvas);
    viewers.magnetics = [f3Scene];
  }
  const d = f3Data, mode = $('#f3-mode').value;
  f3Scene.clear();

  for (const c of d.cells) {
    const col = c.kind === 'magnet' ? (c.body ? '#d2683f' : '#c8503c')
      : '#59636f';
    f3Scene.addBox(c.c, c.axes, c.h, col, { alpha: 0.95 });
  }

  const { lo, hi, res } = d.grid;
  const bmax = Math.max(...d.b);
  const idx = (i, j, k) => (i * res + j) * res + k;
  const at = (i, j, k) => [
    lo[0] + (hi[0] - lo[0]) * i / (res - 1),
    lo[1] + (hi[1] - lo[1]) * j / (res - 1),
    lo[2] + (hi[2] - lo[2]) * k / (res - 1)];

  if (mode === 'cloud') {
    const pts = [], cols = [];
    for (let i = 0; i < res; i++) for (let j = 0; j < res; j++)
      for (let k = 0; k < res; k++) {
        const t = d.b[idx(i, j, k)] / bmax;
        if (t < 0.055) continue;
        pts.push(at(i, j, k));
        cols.push(ramp(Math.pow(t, 0.42)));
      }
    f3Scene.addPoints(pts, cols, { size: 3, alpha: 0.5 });
  } else if (mode === 'slice') {
    const jm = Math.floor(res / 2);
    const pts = [], cols = [];
    for (let i = 0; i < res; i++) for (let k = 0; k < res; k++) {
      pts.push(at(i, jm, k));
      cols.push(ramp(Math.pow(d.b[idx(i, jm, k)] / bmax, 0.42)));
    }
    const km = Math.floor(res / 2);
    for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
      pts.push(at(i, j, km));
      cols.push(ramp(Math.pow(d.b[idx(i, j, km)] / bmax, 0.42)));
    }
    f3Scene.addPoints(pts, cols, { size: 5, alpha: 0.85 });
  } else {
    const segs = [];
    const span = Math.max(hi[0] - lo[0], hi[2] - lo[2]) / res * 1.6;
    for (let n = 0; n < d.bvec.length; n++) {
      const p = d.bvec_pos[n], v = d.bvec[n];
      const m = Math.hypot(v[0], v[1], v[2]);
      if (m / bmax < 0.05) continue;
      const s = span / m;
      segs.push([p, add(p, mul(v, s)), ramp(Math.pow(m / bmax, 0.42))]);
    }
    f3Scene.addLines(segs, '#8ab', { width: 1.2, alpha: 0.75 });
  }

  f3Scene.fit(d.cells.map(c => c.c), 1.9);
  f3Scene.draw();

  const conv = d.converged ? '<b class="good">converged</b>'
    : '<b class="bad">did not converge</b>';
  $('#f3-info').classList.remove('empty');
  $('#f3-info').innerHTML = rows([
    ['state', `${d.state} at ${d.angle_deg}\u00b0`],
    ['|F| on module B', fmt(d.f_mag, 'N')],
    ['F components', `${fmt(d.force[0], 'N')}, ${fmt(d.force[1], 'N')}, ` +
      `${fmt(d.force[2], 'N')}`],
    ['torque about y', `${(d.torque[1] * 1e3).toFixed(3)} mN&middot;m`],
    ['J in magnet A', fmt(d.j_a, 'T'), tipFor('J_attract')],
    ['H in magnet A', fmt(d.h_a, 'kA')],
    ['cells', `${d.n_cells}${d.steel ? ' (with steel)' : ' (magnets only)'}`],
    ['overlaps', d.overlaps === 0 ? '<b class="good">none</b>'
      : `<b class="bad">${d.overlaps}</b>`],
    ['solve', `${conv}, ${d.iters} passes, residual ` +
      d.residual.toExponential(1)],
  ]) + (d.steel ? `<p class="note">${esc(S.meta.notes.fem3d)}</p>` : '');
  bindTips($('#f3-info'));
}

/* angle sweep */
$('#ang-run').addEventListener('click', async () => {
  $('#ang-stage').textContent = 'running\u2026';
  try {
    const r = await runJob('/api/angle_sweep', { ...readForm(), n: 7 },
      s => { $('#ang-stage').textContent = s; });
    $('#ang-stage').textContent = '';
    plotAngle(r);
  } catch (e) {
    $('#ang-stage').textContent = e.message;
  }
});

function plotAngle(r) {
  const c = $('#ang-plot');
  const series = [
    { pts: r.sweep.attract.map(p => [p.deg, p.f]), col: '#5aa9e0',
      label: '|F| attract (N)' },
    { pts: r.sweep.repel.map(p => [p.deg, p.f]), col: '#e0865a',
      label: '|F| repel (N)' },
    { pts: r.sweep.repel.map(p => [p.deg, Math.abs(p.ty) * 1e3]),
      col: '#8ae05a', label: '|torque| repel (mN\u00b7m)', dash: true },
  ];
  linePlot(c, series, 'pivot angle (deg)', '',
    [{ x: r.pivot_deg, label: `one step = ${r.pivot_deg.toFixed(1)}\u00b0` }]);
}

/* ======================================================== stage 2: driver */
$('#dv-run').addEventListener('click', async () => {
  $('#dv-stage').textContent = 'running\u2026';
  try {
    const r = await runJob('/api/circuit', {
      ...readForm(), fem: $('#dv-fem').checked,
      search: $('#dv-search').checked,
    }, s => { $('#dv-stage').textContent = s; });
    $('#dv-stage').textContent = '';
    plotCircuit(r);
    renderCircuitInfo(r);
  } catch (e) {
    $('#dv-stage').textContent = '';
    $('#dv-info').innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
});

function plotCircuit(r) {
  const imax = Math.max(...r.i.map(Math.abs)) || 1;
  const hmax = Math.max(...r.h_ka.map(Math.abs), r.h_need_ka) || 1;
  const vmax = Math.max(...r.v_c.map(Math.abs)) || 1;
  linePlot($('#dv-plot'), [
    { pts: r.t_us.map((t, k) => [t, r.i[k] / imax]), col: '#e0865a',
      label: `coil current, peak ${imax.toFixed(1)} A` },
    { pts: r.t_us.map((t, k) => [t, r.h_ka[k] / hmax]), col: '#5aa9e0',
      label: `field in magnet, peak ${hmax.toFixed(0)} kA/m` },
    { pts: r.t_us.map((t, k) => [t, r.v_c[k] / vmax]), col: '#8ae05a',
      label: `bank voltage, ${vmax.toFixed(0)} V`, dash: true },
    { pts: r.t_us.map((t, k) => [t, r.gate[k] * 0.06 - 1.02]),
      col: '#c9a227', label: 'gate' },
  ], 'time (\u00b5s)', 'normalised', [],
  [{ y: r.h_need_ka / hmax, label: 'switching threshold', col: '#5aa9e0' }]);
}

function renderCircuitInfo(r) {
  const s = r.summary, c = r.circuit;
  let html = rows([
    ['n_eff', `${c.n_eff.toFixed(3)} (${c.source})`, tipFor('n_eff')],
    ['coil mmf reaching the magnet', `${(c.reach * 100).toFixed(1)} %`,
      tipFor('n_eff')],
    ['magnet reluctance', `${(c.r_magnet / 1e6).toFixed(2)} MA/Wb`],
    ['external reluctance', `${(c.r_ext / 1e6).toFixed(2)} MA/Wb`],
    ['inductance', fmt(c.inductance, 'uH')],
    ['turns', `${s.n_layers} \u00d7 ${s.turns_per_layer} = ${s.n_turns}`,
      tipFor('n_layers')],
    ['resistance', fmt(s.R_coil, 'ohm')],
    ['peak current', fmt(s.i_peak, 'A')],
    ['peak field', fmt(s.h_peak, 'kA'), tipFor('h_peak')],
    ['threshold', fmt(s.h_need, 'kA')],
    ['switched', s.switched ? '<b class="good">yes</b>'
      : '<b class="bad">no</b>'],
    ['iron flux', fmt(s.b_steel_peak, 'T'), tipFor('b_steel_peak')],
    ['saturated', s.saturated ? '<b class="warn">yes</b>' : 'no'],
    ['energy drawn', fmt(s.e_drawn, 'mJ')],
    ['lost in resistance',
      `${(100 * s.e_resistive / Math.max(s.e_drawn, 1e-12)).toFixed(0)} %`],
    ['whole module', fmt(s.e_total_module, 'mJ'), tipFor('e_switch')],
  ]);
  if (r.search) {
    const sv = r.search.saving;
    html += `<h3>Pulse programme search</h3>` + rows([
      ['single shot', `${fmt(r.search.base.e, 'mJ')}, ` +
        `${(r.search.base.h / 1e3).toFixed(0)} kA/m`],
      ['best train', esc(r.search.best.program)],
      ['', `${fmt(r.search.best.e, 'mJ')}, ` +
        `${(r.search.best.h / 1e3).toFixed(0)} kA/m`],
      ['saving', sv > 0 ? `<b class="good">${(sv * 100).toFixed(0)} %</b>`
        : 'none'],
    ]);
  }
  $('#dv-info').classList.remove('empty');
  $('#dv-info').innerHTML = html;
  bindTips($('#dv-info'));
}

function renderDriverBom(r) {
  const d = r.driver;
  if (!d || !d.feasible) {
    $('#dv-bom').innerHTML =
      `<p class="err">No driver can be built for this design: ${
        esc(d?.notes || 'unknown')}</p>`;
    return;
  }
  $('#dv-bom').classList.remove('empty');
  $('#dv-bom').innerHTML = rows([
    ['topology', esc(d.topology)],
    ['capacitor', `${d.n_caps} \u00d7 ${esc(d.cap_name)}`],
    ['MOSFET', `${d.n_fets} \u00d7 ${esc(d.mosfet_name)}`],
    ['charger', esc(d.charger_name)],
    ['bank', `${fmt(d.c_bank, 'uF')} at ${fmt(d.v_bank, 'V')} = ` +
      `${fmt(d.e_bank, 'mJ')}`],
    ['recharge', `${(d.recharge_s * 1e3).toFixed(0)} ms`],
    ['battery', `${d.batt_wh.toFixed(2)} Wh, ${fmt(d.batt_mass, 'g')}`],
    ['driver mass', fmt(d.mass, 'g')],
    ['driver volume', fmt(d.volume, 'cc')],
    ['price', `$${d.price.toFixed(2)}`],
    ['damping', esc(d.notes)],
  ]);
}

/* ===================================================== stage 3: mechanics */
function refreshPivotPlot() {
  $('#rl-pivot').src = `/api/plot/pivot?${designQuery()}`;
}

$('#rl-run').addEventListener('click', async () => {
  const sel = $('#rl-drive').value;
  const drives = sel === 'all' ? ['push_off', 'trailing_only', 'reach']
    : [sel];
  $('#rl-stage').textContent = 'running\u2026';
  $('#rl-grid').innerHTML = '';
  try {
    const r = await runJob('/api/rolling', {
      ...readForm(), drives, seconds: parseFloat($('#rl-seconds').value),
    }, s => { $('#rl-stage').textContent = s; });
    $('#rl-stage').textContent = '';
    renderRolling(r);
  } catch (e) {
    $('#rl-stage').textContent = '';
    $('#rl-note').innerHTML = `<p class="err">${esc(e.message)}</p>`;
  }
});

function renderRolling(r) {
  const note = $('#rl-note');
  note.classList.remove('empty');
  if (r.skipped) {
    note.innerHTML = `<p class="warn-box">${esc(r.reason)}</p>`;
    return;
  }
  const me = r.mechanics;
  note.innerHTML = rows([
    ['hold ratio', `${fmt(me.hold_ratio)} (needs ${r.limits.hold_ratio})`,
      tipFor('hold_ratio')],
    ['pivot ratio', `${fmt(me.pivot_ratio)} (needs ${r.limits.pivot_ratio})`,
      tipFor('pivot_ratio')],
    ['barrier', fmt(me.E_barrier, 'mJ')],
    ['available work', fmt(me.W_drive, 'mJ')],
  ]);
  bindTips(note);

  viewers.mechanics = [];
  const grid = $('#rl-grid');
  grid.innerHTML = '';
  Object.entries(r.runs).forEach(([key, run]) => {
    const el = document.createElement('div');
    el.className = 'viewcard';
    el.innerHTML = `
      <div class="vc-head">
        <b>${run.name.replace(/_/g, ' ')}</b>
        <span class="tag ${run.verdict}">${run.verdict}</span>
        <span class="hint">${run.drive.replace(/_/g, ' ')}</span>
      </div>
      <canvas class="viewport small"></canvas>
      <div class="vc-controls">
        <button class="play" type="button">&#9654;</button>
        <button class="step-b" type="button">&#9198;</button>
        <button class="step-f" type="button">&#9197;</button>
        <input class="scrub" type="range" min="0" max="${run.frames.length - 1}" value="0">
        <span class="mono tstamp">0 ms</span>
        <label class="chip"><input type="checkbox" class="loop" checked> loop</label>
      </div>
      <div class="vc-foot">
        reached ${run.settled_deg.toFixed(1)}\u00b0 of
        ${run.target_deg.toFixed(0)}\u00b0${
        run.t_complete ? ` in ${(run.t_complete * 1e3).toFixed(0)} ms` : ''},
        final gap ${(run.final_sep * 1e3).toFixed(2)} mm
        <br><span class="hint">${esc(run.note)}</span>
      </div>`;
    grid.appendChild(el);
    setupPlayback(el, run, r.hull, r.r_face);
  });
}

function setupPlayback(card, run, hull, rFace) {
  const canvas = card.querySelector('canvas');
  const scene = new Scene3D(canvas, { showAxes: false });
  viewers.mechanics.push(scene);
  const scrub = card.querySelector('.scrub');
  const stamp = card.querySelector('.tstamp');
  const playBtn = card.querySelector('.play');
  let frame = 0, playing = false, timer = null;

  function render() {
    const f = run.frames[frame];
    scene.clear();
    // module A: fixed, at the origin, raised so its centre is at r_face
    scene.addHull(hull.map(v => add(v, [0, 0, rFace])), '#4a5a86',
      { alpha: 0.95, edge: 'rgba(190,205,230,.25)' });
    // module B: posed from the trajectory
    scene.addHull(hull.map(v => add(quatRotate(f.quat, v), f.pos)),
      '#c8763c', { alpha: 0.95, edge: 'rgba(240,200,160,.3)' });
    scene.fit([...hull.map(v => add(v, [0, 0, rFace])),
      ...hull.map(v => add(quatRotate(f.quat, v), f.pos))], 1.5);
    scene.draw();
    stamp.textContent = `${(f.t * 1e3).toFixed(0)} ms \u00b7 ` +
      `${f.ang.toFixed(1)}\u00b0 \u00b7 ${f.phase}`;
    scrub.value = frame;
  }

  function step(d) {
    frame += d;
    if (frame >= run.frames.length) {
      if (card.querySelector('.loop').checked) frame = 0;
      else { frame = run.frames.length - 1; stop(); }
    }
    if (frame < 0) frame = 0;
    render();
  }
  function stop() {
    playing = false;
    playBtn.innerHTML = '&#9654;';
    if (timer) clearInterval(timer);
    timer = null;
  }
  playBtn.addEventListener('click', () => {
    if (playing) { stop(); return; }
    playing = true;
    playBtn.innerHTML = '&#10074;&#10074;';
    timer = setInterval(() => step(1), 40);
  });
  card.querySelector('.step-f').addEventListener('click', () => {
    stop(); step(1);
  });
  card.querySelector('.step-b').addEventListener('click', () => {
    stop(); step(-1);
  });
  scrub.addEventListener('input', () => {
    stop();
    frame = parseInt(scrub.value, 10);
    render();
  });
  render();
}

/* ================================================================= matrix */
async function loadMatrixFiles() {
  const files = await getJSON('/api/matrix/files');
  const sel = $('#mx-file');
  sel.innerHTML = files.map(f =>
    `<option${f === 'ga_front.csv' ? ' selected' : ''}>${f}</option>`).join('');
  sel.addEventListener('change', loadMatrix);
  $('#mx-reload').addEventListener('click', loadMatrix);
  $('#mx-filter').addEventListener('change', renderMatrix);
  $('#mx-search').addEventListener('input', renderMatrix);
  $('#mx-export').addEventListener('click', exportCSV);
  $('#mx-load').addEventListener('click', () => {
    if (!S.selected) return;
    writeForm(S.selected);
    showTab('config');
  });
  await loadMatrix();
}

const SHOW_COLS = ['material', 'circuit', 'n_gon', 'r_face', 'd_mag', 'l_mag',
  'n_layers', 'F_attract', 'F_repel', 'asymmetry', 'margin', 'n_eff',
  'h_peak', 'switched', 'pivot_ratio', 'hold_ratio', 'm_module', 'e_switch',
  'scalar', 'feasible'];

async function loadMatrix() {
  try {
    S.matrix = await getJSON(`/api/matrix?file=${$('#mx-file').value}`);
  } catch (e) {
    $('#mx-count').textContent = e.message;
    return;
  }
  const numeric = S.matrix.numeric;
  const pick = (id, def) => {
    const el = $(id);
    el.innerHTML = numeric.map(c =>
      `<option${c === def ? ' selected' : ''}>${c}</option>`).join('');
    el.addEventListener('change', renderMatrix);
  };
  if (!$('#sc-x').dataset.built) {
    pick('#sc-x', 'm_module'); pick('#sc-y', 'F_attract');
    pick('#sc-c', 'e_switch');
    $('#sc-x').dataset.built = '1';
  }
  renderMatrix();
}

function viewRows() {
  if (!S.matrix) return [];
  const f = $('#mx-filter').value;
  const q = $('#mx-search').value.trim().toLowerCase();
  let rs = S.matrix.rows;
  if (f === 'feasible') rs = rs.filter(r => r.feasible);
  else if (f === 'infeasible') rs = rs.filter(r => !r.feasible);
  else if (f === 'switched') rs = rs.filter(r => r.switched);
  if (q) {
    rs = rs.filter(r => `${r.material} ${r.circuit} ${r.violations || ''} ` +
      `${r.pulse_program || ''}`.toLowerCase().includes(q));
  }
  const k = S.sortKey;
  return rs.slice().sort((a, b) => {
    const x = a[k], y = b[k];
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    return (x > y ? 1 : x < y ? -1 : 0) * S.sortDir;
  });
}

function renderMatrix() {
  const rs = viewRows();
  $('#mx-count').innerHTML = `${rs.length} of ${S.matrix.rows.length} designs
    \u00b7 ${S.matrix.rows.filter(r => r.feasible).length} feasible`;
  const cols = SHOW_COLS.filter(c => c in (S.matrix.rows[0] || {}));
  const th = $('#mx-table thead');
  th.innerHTML = `<tr>${cols.map(c =>
    `<th data-k="${c}" data-tip="${esc(tipFor(c))}">${c}${
      S.sortKey === c ? (S.sortDir > 0 ? ' \u25b2' : ' \u25bc') : ''}</th>`)
    .join('')}</tr>`;
  th.querySelectorAll('th').forEach(el => el.addEventListener('click', () => {
    if (S.sortKey === el.dataset.k) S.sortDir *= -1;
    else { S.sortKey = el.dataset.k; S.sortDir = -1; }
    renderMatrix();
  }));
  bindTips(th);

  const tb = $('#mx-table tbody');
  tb.innerHTML = rs.slice(0, 600).map((r, i) =>
    `<tr data-i="${i}" class="${r.feasible ? '' : 'dim'}">${cols.map(c =>
      `<td>${typeof r[c] === 'number' ? fmt(r[c], colUnit(c))
        : (r[c] === true ? 'yes' : r[c] === false ? 'no'
          : esc(r[c] ?? ''))}</td>`).join('')}</tr>`).join('');
  tb.querySelectorAll('tr').forEach(tr => tr.addEventListener('click', () => {
    selectRow(rs[parseInt(tr.dataset.i, 10)]);
  }));
  drawScatter(rs);
}

const colUnit = c => ({
  r_face: 'mm', d_mag: 'mm', l_mag: 'mm', t_steel: 'mm', gap: 'mm',
  wire_d: 'mm', m_module: 'g', e_switch: 'mJ', F_attract: 'N', F_repel: 'N',
  h_peak: 'kA', drv_mass: 'g',
}[c]);

function selectRow(r) {
  if (!r) return;
  S.selected = Object.fromEntries(S.meta.design_keys
    .filter(k => r[k] !== null && r[k] !== undefined).map(k => [k, r[k]]));
  $('#mx-load').disabled = false;
  const d = $('#mx-detail');
  d.classList.remove('empty');
  d.innerHTML = `
    <h3>${esc(r.material)} \u00b7 n=${r.n_gon} \u00b7 ${esc(r.circuit)}</h3>
    ${r.feasible ? '<p class="good">feasible</p>'
      : `<p class="bad">${esc(r.violations || 'infeasible')}</p>`}
    ${rows(SHOW_COLS.filter(c => c in r).map(c =>
      [c, typeof r[c] === 'number' ? fmt(r[c], colUnit(c))
        : (r[c] === true ? 'yes' : r[c] === false ? 'no' : esc(r[c] ?? '')),
      tipFor(c)]))}
    ${r.pulse_program ? `<p class="note">pulse: ${esc(r.pulse_program)}</p>`
      : ''}`;
  bindTips(d);
}

function exportCSV() {
  const rs = viewRows();
  if (!rs.length) return;
  const cols = Object.keys(rs[0]);
  const csv = [cols.join(',')].concat(rs.map(r =>
    cols.map(c => {
      const v = r[c];
      const s = v === null || v === undefined ? '' : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    }).join(','))).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = 'design_view.csv';
  a.click();
}

/* ================================================================ plotting */
function plotBox(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight || 320;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0e1218';
  ctx.fillRect(0, 0, w, h);
  return { ctx, w, h, pad: { l: 58, r: 18, t: 14, b: 44 } };
}

function axes(ctx, w, h, pad, xr, yr, xlab, ylab) {
  ctx.strokeStyle = '#2a3340';
  ctx.fillStyle = '#7f8b9c';
  ctx.font = '11px ui-monospace, monospace';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const x = pad.l + (w - pad.l - pad.r) * i / 5;
    const y = h - pad.b - (h - pad.t - pad.b) * i / 5;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    const xv = xr[0] + (xr[1] - xr[0]) * i / 5;
    const yv = yr[0] + (yr[1] - yr[0]) * i / 5;
    ctx.fillText(sig(xv), x - 14, h - pad.b + 15);
    ctx.fillText(sig(yv), 6, y + 4);
  }
  ctx.fillStyle = '#9fb0c4';
  ctx.fillText(xlab, (w + pad.l) / 2 - 40, h - 8);
  if (ylab) {
    ctx.save(); ctx.translate(12, (h + pad.t) / 2 + 30); ctx.rotate(-Math.PI / 2);
    ctx.fillText(ylab, 0, 0); ctx.restore();
  }
}

const sig = v => Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0)
  ? v.toExponential(1) : (Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(1));

function linePlot(canvas, series, xlab, ylab, vlines = [], hlines = []) {
  const { ctx, w, h, pad } = plotBox(canvas);
  const all = series.flatMap(s => s.pts);
  if (!all.length) return;
  const xr = [Math.min(...all.map(p => p[0])), Math.max(...all.map(p => p[0]))];
  const yr = [Math.min(...all.map(p => p[1])), Math.max(...all.map(p => p[1]))];
  if (yr[0] === yr[1]) yr[1] += 1;
  const px = v => pad.l + (w - pad.l - pad.r) * (v - xr[0]) / (xr[1] - xr[0] || 1);
  const py = v => h - pad.b - (h - pad.t - pad.b) * (v - yr[0]) / (yr[1] - yr[0]);
  axes(ctx, w, h, pad, xr, yr, xlab, ylab);

  for (const L of hlines) {
    ctx.strokeStyle = L.col || '#4b5563';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad.l, py(L.y)); ctx.lineTo(w - pad.r, py(L.y));
    ctx.stroke(); ctx.setLineDash([]);
  }
  for (const L of vlines) {
    ctx.strokeStyle = '#4b5563';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(px(L.x), pad.t); ctx.lineTo(px(L.x), h - pad.b);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#7f8b9c';
    ctx.fillText(L.label, px(L.x) + 4, pad.t + 12);
  }
  series.forEach((s, i) => {
    ctx.strokeStyle = s.col;
    ctx.lineWidth = 1.8;
    if (s.dash) ctx.setLineDash([5, 3]);
    ctx.beginPath();
    s.pts.forEach((p, k) => k ? ctx.lineTo(px(p[0]), py(p[1]))
      : ctx.moveTo(px(p[0]), py(p[1])));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = s.col;
    ctx.fillRect(pad.l + 8, pad.t + 6 + i * 15, 9, 3);
    ctx.fillStyle = '#c3cede';
    ctx.fillText(s.label, pad.l + 22, pad.t + 11 + i * 15);
  });
}

function drawScatter(rs) {
  const canvas = $('#scatter');
  const { ctx, w, h, pad } = plotBox(canvas);
  const kx = $('#sc-x').value, ky = $('#sc-y').value, kc = $('#sc-c').value;
  const pts = rs.filter(r => typeof r[kx] === 'number' &&
    typeof r[ky] === 'number');
  if (!pts.length) return;
  const xs = pts.map(r => r[kx]), ys = pts.map(r => r[ky]);
  const cs = pts.map(r => typeof r[kc] === 'number' ? r[kc] : 0);
  const xr = [Math.min(...xs), Math.max(...xs)];
  const yr = [Math.min(...ys), Math.max(...ys)];
  const cr = [Math.min(...cs), Math.max(...cs)];
  if (xr[0] === xr[1]) xr[1] += 1;
  if (yr[0] === yr[1]) yr[1] += 1;
  axes(ctx, w, h, pad, xr, yr, kx, ky);
  const px = v => pad.l + (w - pad.l - pad.r) * (v - xr[0]) / (xr[1] - xr[0]);
  const py = v => h - pad.b - (h - pad.t - pad.b) * (v - yr[0]) / (yr[1] - yr[0]);
  canvas._pts = [];
  pts.forEach((r, i) => {
    const x = px(xs[i]), y = py(ys[i]);
    ctx.globalAlpha = r.feasible ? 0.92 : 0.25;
    ctx.fillStyle = ramp((cs[i] - cr[0]) / (cr[1] - cr[0] || 1));
    ctx.beginPath();
    ctx.arc(x, y, r.feasible ? 4 : 2.6, 0, 6.283);
    ctx.fill();
    canvas._pts.push({ x, y, r });
  });
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#7f8b9c';
  ctx.fillText(`colour: ${kc}  ${sig(cr[0])} \u2192 ${sig(cr[1])}`,
    pad.l + 8, pad.t + 12);
  if (!canvas._bound) {
    canvas._bound = true;
    canvas.addEventListener('click', e => {
      const b = canvas.getBoundingClientRect();
      const mx = e.clientX - b.left, my = e.clientY - b.top;
      let best = null, bd = 1e9;
      for (const p of canvas._pts || []) {
        const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
        if (d < bd) { bd = d; best = p; }
      }
      if (best && bd < 400) selectRow(best.r);
    });
  }
}

/* ============================================================== materials */
function buildMaterialTable() {
  const cols = ['name', 'family', 'Br', 'Hcb', 'Hcj', 'BHmax', 'mu_rec',
    'rho', 'switching', 'source'];
  $('#mat-table thead').innerHTML =
    `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
  $('#mat-table tbody').innerHTML = S.meta.materials.map(m =>
    `<tr class="${m.source === 'lit' ? 'dim' : ''}">${cols.map(c => {
      let v = m[c];
      if (c === 'Hcb' || c === 'Hcj') v = `${(v / 1e3).toFixed(0)} kA/m`;
      else if (c === 'BHmax') v = `${(v / 1e3).toFixed(0)} kJ/m\u00b3`;
      else if (c === 'Br') v = `${v.toFixed(2)} T`;
      else if (c === 'rho') v = `${v} kg/m\u00b3`;
      return `<td>${esc(v)}</td>`;
    }).join('')}</tr>`).join('');
}

window.addEventListener('resize', () => {
  renderMatrix();
  Object.values(viewers).flat().forEach(v => v.draw());
});
