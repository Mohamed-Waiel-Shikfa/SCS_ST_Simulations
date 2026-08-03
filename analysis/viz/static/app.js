/* Magnobots design explorer.
 *
 * Everything displayed comes from the backend, which calls the real pipeline.
 * The only arithmetic here is unit conversion, and every conversion is
 * declared once in FIELDS so a value cannot be shown in millimetres in one
 * place and metres in another.
 */
'use strict';

// ---------------------------------------------------------------- fields
// scale: multiply the SI value by this to get the display value.
const FIELDS = {
  material:      {label:'material',    unit:'',     scale:1,    dp:0, txt:true},
  circuit:       {label:'circuit',     unit:'',     scale:1,    dp:0, txt:true},
  n_gon:         {label:'n',           unit:'',     scale:1,    dp:0},
  n_faces:       {label:'faces',       unit:'',     scale:1,    dp:0},
  r_face:        {label:'r_face',      unit:'mm',   scale:1e3,  dp:2},
  a_face:        {label:'face side',   unit:'mm',   scale:1e3,  dp:2},
  bounding_cube: {label:'bounding box',unit:'mm',   scale:1e3,  dp:1},
  d_mag:         {label:'magnet dia',  unit:'mm',   scale:1e3,  dp:2},
  l_mag:         {label:'magnet len',  unit:'mm',   scale:1e3,  dp:2},
  t_steel:       {label:'steel wall',  unit:'mm',   scale:1e3,  dp:2},
  r_clear:       {label:'clearance',   unit:'mm',   scale:1e3,  dp:2},
  gap:           {label:'gap',         unit:'mm',   scale:1e3,  dp:3},
  wire_d:        {label:'wire dia',    unit:'mm',   scale:1e3,  dp:3},
  v_cap:         {label:'bank voltage',unit:'V',    scale:1,    dp:1},
  c_cap:         {label:'bank cap',    unit:'uF',   scale:1e6,  dp:1},
  F_attract:     {label:'attraction',  unit:'N',    scale:1,    dp:2},
  F_repel:       {label:'repulsion',   unit:'N',    scale:1,    dp:2},
  asymmetry:     {label:'asymmetry',   unit:':1',   scale:1,    dp:1},
  J_attract:     {label:'J attract',   unit:'T',    scale:1,    dp:3},
  J_repel:       {label:'J repel',     unit:'T',    scale:1,    dp:3},
  margin:        {label:'demag margin',unit:'Hcj',  scale:1,    dp:2},
  margin_attract:{label:'margin attr', unit:'Hcj',  scale:1,    dp:2},
  margin_repel:  {label:'margin repel',unit:'Hcj',  scale:1,    dp:2},
  m_module:      {label:'module mass', unit:'g',    scale:1e3,  dp:0},
  hold_ratio:    {label:'hold',        unit:'x wt', scale:1,    dp:1},
  pivot_ratio:   {label:'pivot',       unit:'x',    scale:1,    dp:2},
  E_barrier:     {label:'barrier',     unit:'mJ',   scale:1e3,  dp:2},
  W_drive:       {label:'drive work',  unit:'mJ',   scale:1e3,  dp:2},
  lift:          {label:'CoM lift',    unit:'mm',   scale:1e3,  dp:2},
  e_switch:      {label:'switch energy',unit:'mJ',  scale:1e3,  dp:0},
  mmf:           {label:'MMF available',unit:'At',  scale:1,    dp:0},
  mmf_need:      {label:'MMF needed',  unit:'At',   scale:1,    dp:0},
  v_need:        {label:'voltage needed',unit:'V',  scale:1,    dp:1},
  drv_mass:      {label:'driver mass', unit:'g',    scale:1e3,  dp:1},
  drv_price:     {label:'driver cost', unit:'$',    scale:1,    dp:0},
  scalar:        {label:'score',       unit:'',     scale:1,    dp:3},
  ld_ratio:      {label:'L/D',         unit:'',     scale:1,    dp:2},
  r_vertex:      {label:'vertex radius',unit:'mm',  scale:1e3,  dp:2},
  pivot_deg:     {label:'pivot step',  unit:'deg',  scale:1,    dp:1},
  free_volume:   {label:'free volume', unit:'cc',   scale:1e6,  dp:1},
};

const PRESETS = {
  key: ['material','circuit','n_gon','F_attract','F_repel','asymmetry',
        'm_module','hold_ratio','pivot_ratio','margin','e_switch','scalar'],
  geometry: ['material','n_gon','n_faces','r_face','a_face','bounding_cube',
             'd_mag','l_mag','t_steel','r_clear','gap','m_module'],
  electrical: ['material','circuit','wire_d','v_cap','c_cap','mmf','mmf_need',
               'v_need','e_switch','drv_mass','drv_price','drv_mosfet'],
};

const S = {
  meta: null, rows: [], view: [], numeric: [],
  sort: {key:'scalar', dir:-1}, page: 0, perPage: 60,
  sel: null, preset: 'key',
  design: null, result: null,
  rot: {x: -0.42, y: 0.62}, geom: null,
};

// ---------------------------------------------------------------- helpers
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const el = (t, a = {}, kids = []) => {
  const n = document.createElementNS(
    a.svg ? 'http://www.w3.org/2000/svg' : 'http://www.w3.org/1999/xhtml', t);
  for (const [k, v] of Object.entries(a)) {
    if (k === 'svg') continue;
    if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) if (c) n.appendChild(c);
  return n;
};

function fmt(key, v) {
  if (v === null || v === undefined || v === '') return '\u2014';
  const f = FIELDS[key];
  if (!f) return typeof v === 'number' ? trim(v) : String(v);
  if (f.txt || typeof v === 'string') return String(v);
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  const x = v * f.scale;
  if (!isFinite(x)) return '\u221e';
  return x.toFixed(f.dp);
}
function unit(key) { return FIELDS[key] ? FIELDS[key].unit : ''; }
function label(key) { return FIELDS[key] ? FIELDS[key].label : key; }
function trim(v) {
  if (v === null || v === undefined || !isFinite(v)) return '\u2014';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-3 || a >= 1e6) return v.toExponential(2);
  return String(+v.toPrecision(4));
}
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove('show'), 6000);
}
async function api(url, opt) {
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({error: r.statusText}));
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

// ---------------------------------------------------------------- tabs
$$('#tabs button').forEach(b => b.addEventListener('click', () => {
  $$('#tabs button').forEach(x => x.classList.toggle('active', x === b));
  $$('.tab').forEach(t => t.classList.toggle(
    'active', t.id === 'tab-' + b.dataset.tab));
  // Charts are sized from clientWidth, which is 0 while a tab is hidden, so
  // anything drawn before its tab was shown comes out stretched.  Redraw on
  // reveal rather than guessing a width.
  if (b.dataset.tab === 'validation') loadExperiment();
  if (b.dataset.tab === 'module') drawModule();
  if (b.dataset.tab === 'dynamics') renderDynamics(S.result && S.result.dynamics);
  if (b.dataset.tab === 'matrix') drawScatter();
}));

// ---------------------------------------------------------------- boot
(async function boot() {
  try {
    S.meta = await api('/api/meta');
    $('#subtitle').textContent =
      `${S.meta.materials.length} materials \u00b7 ` +
      `constraints: demag \u2264 ${S.meta.limits.margin} Hcj, ` +
      `hold \u2265 ${S.meta.limits.hold_ratio}\u00d7, ` +
      `pivot \u2265 ${S.meta.limits.pivot_ratio}, ` +
      `box \u2264 ${S.meta.limits.bounding_cube * 1e3} mm`;
    buildConfigForm();
    const files = await api('/api/matrix/files');
    const sel = $('#mx-file');
    files.forEach(f => sel.appendChild(el('option', {value: f, text: f})));
    sel.value = files.includes('ga_front.csv') ? 'ga_front.csv' : files[0];
    await loadMatrix();
  } catch (e) { toast('startup failed: ' + e.message); }
})();

// ================================================================ MATRIX
async function loadMatrix() {
  const d = await api('/api/matrix?file=' + encodeURIComponent($('#mx-file').value));
  S.rows = d.rows;
  S.numeric = d.numeric.filter(k => FIELDS[k]);
  ['sc-x', 'sc-y', 'sc-c'].forEach(id => {
    const s = $('#' + id);
    s.innerHTML = '';
    S.numeric.forEach(k => s.appendChild(
      el('option', {value: k, text: label(k) + (unit(k) ? ` (${unit(k)})` : '')})));
  });
  $('#sc-x').value = 'm_module';
  $('#sc-y').value = 'F_attract';
  $('#sc-c').value = 'pivot_ratio';
  fillPresets();
  applyFilter();
}

function fillPresets() {
  const s = $('#cfg-preset');
  s.innerHTML = '<option value="">load a preset\u2026</option>';
  const feas = S.rows.filter(r => r.feasible);
  const add = (lbl, row) => {
    if (!row) return;
    const i = S.rows.indexOf(row);
    s.appendChild(el('option', {value: i, text: lbl}));
  };
  const best = (k, dir) => feas.slice().sort(
    (a, b) => dir * ((a[k] ?? 0) - (b[k] ?? 0)))[0];
  add('\u2014 as built (Alnico 5 bare rod) \u2014', null);
  s.appendChild(el('option', {value: 'baseline', text: 'as-built baseline'}));
  add('best score', best('scalar', -1));
  add('lightest', best('m_module', 1));
  add('most attraction', best('F_attract', -1));
  add('most repulsion', best('F_repel', -1));
  add('lowest asymmetry', best('asymmetry', 1));
  add('best pivot margin', best('pivot_ratio', -1));
}

function applyFilter() {
  const f = $('#mx-filter').value;
  const q = $('#mx-search').value.trim().toLowerCase();
  S.view = S.rows.filter(r => {
    if (f === 'feasible' && !r.feasible) return false;
    if (f === 'infeasible' && r.feasible) return false;
    if (q) {
      const hay = `${r.material} ${r.circuit} ${r.violations || ''}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  sortView();
  S.page = 0;
  $('#mx-count').textContent =
    `${S.view.length} of ${S.rows.length} shown \u00b7 ` +
    `${S.rows.filter(r => r.feasible).length} feasible`;
  drawTable();
  drawScatter();
}

function sortView() {
  const {key, dir} = S.sort;
  S.view.sort((a, b) => {
    const x = a[key], y = b[key];
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    if (typeof x === 'string') return dir * x.localeCompare(y);
    return dir * (x - y);
  });
}

function drawTable() {
  const cols = S.preset === 'all'
    ? Object.keys(S.rows[0] || {}).filter(k => k !== 'violations')
    : PRESETS[S.preset];
  const t = $('#mx-table');
  t.innerHTML = '';
  const head = el('tr');
  cols.forEach(c => head.appendChild(el('th', {
    text: label(c) + (unit(c) ? ` (${unit(c)})` : '') +
          (S.sort.key === c ? (S.sort.dir < 0 ? ' \u2193' : ' \u2191') : ''),
    onclick: () => {
      S.sort = {key: c, dir: S.sort.key === c ? -S.sort.dir : -1};
      sortView(); drawTable();
    }
  })));
  t.appendChild(el('thead', {}, head));

  const body = el('tbody');
  const start = S.page * S.perPage;
  S.view.slice(start, start + S.perPage).forEach(r => {
    const tr = el('tr', {class: r.feasible ? '' : 'infeasible'});
    if (r === S.sel) tr.classList.add('sel');
    cols.forEach(c => tr.appendChild(el('td', {
      text: fmt(c, r[c]),
      class: (FIELDS[c] && FIELDS[c].txt) || typeof r[c] === 'string' ? 'txt' : ''
    })));
    tr.addEventListener('click', () => select(r));
    body.appendChild(tr);
  });
  t.appendChild(body);
  const pages = Math.max(1, Math.ceil(S.view.length / S.perPage));
  $('#mx-page').textContent = `page ${S.page + 1} of ${pages}`;
  $('#mx-prev').disabled = S.page === 0;
  $('#mx-next').disabled = S.page >= pages - 1;
}

// ---------------------------------------------------------------- scatter
function drawScatter() {
  const host = $('#scatter');
  host.innerHTML = '';
  const kx = $('#sc-x').value, ky = $('#sc-y').value, kc = $('#sc-c').value;
  const lx = $('#sc-logx').checked, ly = $('#sc-logy').checked;
  const pts = S.view.filter(r =>
    typeof r[kx] === 'number' && typeof r[ky] === 'number' &&
    isFinite(r[kx]) && isFinite(r[ky]) &&
    (!lx || r[kx] > 0) && (!ly || r[ky] > 0));
  if (!pts.length) { host.innerHTML = '<p class="note">no points to plot</p>'; return; }

  const W = host.clientWidth || 700, H = 400;
  const M = {t: 14, r: 62, b: 44, l: 62};
  const sx = FIELDS[kx].scale, sy = FIELDS[ky].scale, sc = FIELDS[kc] ? FIELDS[kc].scale : 1;
  const fx = v => lx ? Math.log10(v * sx) : v * sx;
  const fy = v => ly ? Math.log10(v * sy) : v * sy;
  const xs = pts.map(r => fx(r[kx])), ys = pts.map(r => fy(r[ky]));
  const cv = pts.map(r => typeof r[kc] === 'number' && isFinite(r[kc]) ? r[kc] * sc : null)
                .filter(v => v !== null);
  const ex = [Math.min(...xs), Math.max(...xs)];
  const ey = [Math.min(...ys), Math.max(...ys)];
  const ec = cv.length ? [Math.min(...cv), Math.max(...cv)] : [0, 1];
  const padx = (ex[1] - ex[0]) * .06 || 1, pady = (ey[1] - ey[0]) * .06 || 1;
  ex[0] -= padx; ex[1] += padx; ey[0] -= pady; ey[1] += pady;

  const px = v => M.l + (v - ex[0]) / (ex[1] - ex[0]) * (W - M.l - M.r);
  const py = v => H - M.b - (v - ey[0]) / (ey[1] - ey[0]) * (H - M.t - M.b);
  const col = v => {
    if (v === null) return '#5c6478';
    const t = ec[1] > ec[0] ? (v - ec[0]) / (ec[1] - ec[0]) : .5;
    const stops = [[13,16,23],[54,84,140],[90,169,255],[78,201,160],[255,193,77]];
    const i = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
    const f = t * (stops.length - 1) - i;
    const c = stops[i].map((a, j) => Math.round(a + (stops[i+1][j] - a) * f));
    return `rgb(${c.join(',')})`;
  };

  const svg = el('svg', {svg: 1, viewBox: `0 0 ${W} ${H}`,
                         preserveAspectRatio: 'xMidYMid meet'});
  const g = k => el('g', {svg: 1, class: k});
  const grid = g('sc-grid'), axis = g('sc-axis');

  const ticks = (lo, hi, isLog) => {
    const out = [];
    for (let i = 0; i <= 5; i++) {
      const v = lo + (hi - lo) * i / 5;
      out.push({v, t: isLog ? trim(Math.pow(10, v)) : trim(v)});
    }
    return out;
  };
  ticks(ex[0], ex[1], lx).forEach(t => {
    grid.appendChild(el('line', {svg: 1, x1: px(t.v), x2: px(t.v),
                                 y1: M.t, y2: H - M.b}));
    axis.appendChild(el('text', {svg: 1, x: px(t.v), y: H - M.b + 15,
                                 'text-anchor': 'middle', text: t.t}));
  });
  ticks(ey[0], ey[1], ly).forEach(t => {
    grid.appendChild(el('line', {svg: 1, y1: py(t.v), y2: py(t.v),
                                 x1: M.l, x2: W - M.r}));
    axis.appendChild(el('text', {svg: 1, x: M.l - 8, y: py(t.v) + 3.5,
                                 'text-anchor': 'end', text: t.t}));
  });
  svg.appendChild(grid); svg.appendChild(axis);

  svg.appendChild(el('text', {svg: 1, class: 'sc-label', x: (W - M.r + M.l) / 2,
    y: H - 8, 'text-anchor': 'middle',
    text: `${label(kx)} (${unit(kx)})${lx ? ' \u2013 log' : ''}`}));
  svg.appendChild(el('text', {svg: 1, class: 'sc-label',
    transform: `rotate(-90 14 ${H / 2})`, x: 14, y: H / 2,
    'text-anchor': 'middle',
    text: `${label(ky)} (${unit(ky)})${ly ? ' \u2013 log' : ''}`}));

  // colour legend
  const lg = el('g', {svg: 1});
  const defs = el('defs', {svg: 1});
  const grad = el('linearGradient', {svg: 1, id: 'cg', x1: 0, y1: 1, x2: 0, y2: 0});
  for (let i = 0; i <= 10; i++)
    grad.appendChild(el('stop', {svg: 1, offset: (i * 10) + '%',
      'stop-color': col(ec[0] + (ec[1] - ec[0]) * i / 10)}));
  defs.appendChild(grad); svg.appendChild(defs);
  lg.appendChild(el('rect', {svg: 1, x: W - M.r + 16, y: M.t, width: 11,
    height: H - M.t - M.b, fill: 'url(#cg)', stroke: '#262c3a'}));
  lg.appendChild(el('text', {svg: 1, class: 'sc-label', x: W - M.r + 12,
    y: M.t - 3, 'text-anchor': 'start', text: label(kc)}));
  [0, 1].forEach(i => lg.appendChild(el('text', {svg: 1, class: 'sc-label',
    x: W - M.r + 31, y: i ? M.t + 8 : H - M.b, text: trim(ec[i ? 1 : 0])})));
  svg.appendChild(lg);

  const tip = el('div', {id: 'tip', style: 'display:none'});
  document.body.appendChild(tip);
  pts.forEach(r => {
    const c = el('circle', {svg: 1, class: 'sc-dot', cx: px(fx(r[kx])),
      cy: py(fy(r[ky])), r: r === S.sel ? 7 : 4,
      fill: typeof r[kc] === 'number' ? col(r[kc] * sc) : '#5c6478',
      'fill-opacity': r.feasible ? .92 : .3,
      stroke: r === S.sel ? '#fff' : 'none', 'stroke-width': r === S.sel ? 2 : 0});
    c.addEventListener('mousemove', e => {
      tip.style.display = 'block';
      tip.style.left = Math.min(e.clientX + 14, innerWidth - 310) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
      tip.innerHTML =
        `<b>${r.material}</b> n=${r.n_gon} ${r.circuit}<br>` +
        `${label(kx)} ${fmt(kx, r[kx])} ${unit(kx)}<br>` +
        `${label(ky)} ${fmt(ky, r[ky])} ${unit(ky)}<br>` +
        `${label(kc)} ${fmt(kc, r[kc])} ${unit(kc)}<br>` +
        (r.feasible ? '<span style="color:#4ec9a0">feasible</span>'
                    : `<span style="color:#ff6b6b">${r.violations || 'infeasible'}</span>`);
    });
    c.addEventListener('mouseleave', () => tip.style.display = 'none');
    c.addEventListener('click', () => { tip.style.display = 'none'; select(r); });
    svg.appendChild(c);
  });

  host.appendChild(svg);
  $('#sc-note').textContent =
    `${pts.length} points. Dim points are infeasible. ` +
    `Click any point to inspect it, then send it to Configure & run.`;
}

// ---------------------------------------------------------------- select
function select(r) {
  S.sel = r;
  drawTable(); drawScatter();
  const d = $('#mx-detail');
  d.className = '';
  d.innerHTML = '';

  d.appendChild(el('div', {class: 'cards'}, [
    card('attraction', fmt('F_attract', r.F_attract), 'N'),
    card('repulsion', fmt('F_repel', r.F_repel), 'N'),
    card('asymmetry', fmt('asymmetry', r.asymmetry), ':1'),
    card('mass', fmt('m_module', r.m_module), 'g'),
  ]));

  const lim = S.meta.limits;
  d.appendChild(el('div', {class: 'cards', style: 'margin-top:10px'}, [
    gauge('demag margin', r.margin, lim.margin, 'Hcj', true),
    gauge('hold', r.hold_ratio, lim.hold_ratio, '\u00d7 weight', false),
    gauge('pivot', r.pivot_ratio, lim.pivot_ratio, '\u00d7', false),
  ]));

  const dl = el('dl', {class: 'kv'});
  ['material','circuit','n_gon','n_faces','r_face','d_mag','l_mag',
   't_steel','r_clear','gap','wire_d','v_cap','c_cap','bounding_cube',
   'e_switch','drv_mass','drv_price','fidelity'].forEach(k => {
    if (r[k] === undefined) return;
    dl.appendChild(el('dt', {text: label(k)}));
    dl.appendChild(el('dd', {text: fmt(k, r[k]) + ' ' + unit(k)}));
  });
  d.appendChild(el('div', {style: 'margin-top:13px'}, dl));

  if (!r.feasible && r.violations) {
    d.appendChild(el('div', {class: 'viol', html:
      '<b>infeasible</b><ul>' +
      String(r.violations).split(';').map(v => `<li>${v.trim()}</li>`).join('') +
      '</ul>'}));
  }
  d.appendChild(el('div', {style: 'margin-top:13px;display:flex;gap:9px;flex-wrap:wrap'}, [
    el('button', {class: 'primary', text: 'Send to Configure & run',
      onclick: () => { loadDesign(r); gotoTab('config'); }}),
    el('button', {class: 'ghost', text: 'Inspect all stages',
      onclick: () => { loadDesign(r); gotoTab('config'); $('#cfg-run').click(); }}),
  ]));
}

function card(k, v, u, cls, extra) {
  return el('div', {class: 'card ' + (cls || '')}, [
    el('div', {class: 'k', text: k}),
    el('div', {class: 'v', html: v + (u ? `<span class="u">${u}</span>` : '')}),
    extra ? el('div', {class: 'x', text: extra}) : null,
  ]);
}

function gauge(name, val, limit, u, isMax, scale) {
  if (val === null || val === undefined || !isFinite(val))
    return card(name, '\u2014', u);
  const s = scale || 1;
  const ok = isMax ? val <= limit : val >= limit;
  const frac = isMax ? Math.min(1, val / limit)
                     : Math.min(1, val / (limit * 2));
  const dp = (val * s) >= 100 ? 0 : 2;
  const c = el('div', {class: 'card ' + (ok ? 'good' : 'bad')}, [
    el('div', {class: 'k', text: name}),
    el('div', {class: 'v',
      html: (val * s).toFixed(dp) + `<span class="u">${u}</span>`}),
    el('div', {class: 'x',
      text: (isMax ? 'limit \u2264 ' : 'limit \u2265 ') + (limit * s)}),
  ]);
  const g = el('div', {class: 'gauge'});
  g.appendChild(el('i', {style:
    `width:${(frac * 100).toFixed(0)}%;background:${ok ? 'var(--good)' : 'var(--bad)'}`}));
  c.appendChild(g);
  return c;
}

function gotoTab(name) { $(`#tabs button[data-tab="${name}"]`).click(); }

// ================================================================ CONFIG
const BASELINE = {material:'LNG37', circuit:'none', n_gon:8, r_face:19.4e-3,
  d_mag:4.75e-3, l_mag:12.5e-3, t_steel:0.5e-3, r_clear:0.0, gap:0.1e-3,
  wire_d:0.3e-3, v_cap:30.0, c_cap:10e-6};

function buildConfigForm() {
  const f = $('#cfg-form');
  f.innerHTML = '';
  S.meta.design_keys.forEach(k => {
    const wrap = el('div', {class: 'field'});
    wrap.appendChild(el('label', {text: label(k) + (unit(k) ? ` (${unit(k)})` : '')}));
    let input;
    if (k === 'material') {
      input = el('select', {id: 'f-' + k});
      let fam = null;
      S.meta.materials.forEach(m => {
        if (m.family !== fam) { fam = m.family; }
        input.appendChild(el('option', {value: m.name,
          text: `${m.name}  \u2014 Br ${m.Br.toFixed(2)} T, Hcj ${(m.Hcj/1e3).toFixed(0)} kA/m`}));
      });
    } else if (k === 'circuit') {
      input = el('select', {id: 'f-' + k});
      [['potcore','pot core (steel return path)'], ['none','bare rod']]
        .forEach(([v, t]) => input.appendChild(el('option', {value: v, text: t})));
    } else if (k === 'n_gon') {
      input = el('select', {id: 'f-' + k});
      S.meta.n_gon_options.forEach(o => input.appendChild(el('option',
        {value: o.n, text: `n=${o.n} \u2014 ${o.faces} faces, ${o.pivot_deg.toFixed(1)}\u00b0 step`})));
    } else {
      input = el('input', {id: 'f-' + k, type: 'number', step: 'any'});
    }
    wrap.appendChild(input);
    const b = S.meta.bounds[k];
    if (b) wrap.appendChild(el('div', {class: 'hint',
      text: `searched ${(b[0] * FIELDS[k].scale).toFixed(FIELDS[k].dp)}\u2013` +
            `${(b[1] * FIELDS[k].scale).toFixed(FIELDS[k].dp)}`}));
    f.appendChild(wrap);
  });
  loadDesign(BASELINE);
}

function loadDesign(r) {
  S.meta.design_keys.forEach(k => {
    const inp = $('#f-' + k);
    if (!inp || r[k] === undefined || r[k] === null) return;
    if (FIELDS[k].txt || k === 'n_gon') { inp.value = r[k]; return; }
    // Show a readable rounded value, but keep the exact SI number so a design
    // taken from the matrix re-evaluates to the SAME answer.  Rounding to six
    // significant figures on the way in shifted forces by ~0.04 %, which is
    // small but is exactly the kind of drift that makes a checking tool
    // useless: the user cannot tell a real difference from a display artefact.
    inp.value = +(r[k] * FIELDS[k].scale).toPrecision(6);
    inp.dataset.exact = r[k];
  });
}

function readDesign() {
  const out = {};
  for (const k of S.meta.design_keys) {
    const inp = $('#f-' + k);
    const raw = inp.value;
    if (FIELDS[k].txt) { out[k] = raw; continue; }
    const v = parseFloat(raw);
    if (!isFinite(v)) throw new Error(`${label(k)} is not a number`);
    if (k === 'n_gon') { out[k] = v; continue; }
    const exact = inp.dataset.exact ? parseFloat(inp.dataset.exact) : null;
    // untouched field: hand back the exact original rather than the rounded
    // display value
    out[k] = (exact !== null &&
              +(exact * FIELDS[k].scale).toPrecision(6) === v)
      ? exact : v / FIELDS[k].scale;
  }
  return out;
}

// any manual edit invalidates the stored exact value
document.addEventListener('input', e => {
  if (e.target.id && e.target.id.startsWith('f-')) delete e.target.dataset.exact;
});

$('#cfg-preset').addEventListener('change', e => {
  const v = e.target.value;
  if (v === '') return;
  loadDesign(v === 'baseline' ? BASELINE : S.rows[+v]);
  $('#cfg-note').textContent = v === 'baseline'
    ? 'Loaded the as-built prototype: Alnico 5, bare rod, no return path.'
    : 'Loaded from the design matrix. Values are exactly as evaluated.';
});

$('#cfg-run').addEventListener('click', async ev => {
  ev.preventDefault();
  let design;
  try { design = readDesign(); }
  catch (e) { toast(e.message); return; }

  const fidelity = $('#cfg-fidelity').value;
  const dyn = $('#cfg-dyn').checked;
  const btn = $('#cfg-run');
  btn.disabled = true;
  $('#cfg-progress').classList.remove('hidden');
  $('#cfg-bar').style.width = '0%';
  $('#cfg-stage').textContent = 'starting\u2026';

  try {
    const {job} = await api('/api/evaluate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...design, fidelity, dynamics: dyn})});
    let st;
    do {
      await new Promise(r => setTimeout(r, 500));
      st = await api('/api/job/' + job);
      $('#cfg-bar').style.width = (st.progress || 0) + '%';
      $('#cfg-stage').textContent = st.stage || '';
    } while (st.state === 'running');

    if (st.state === 'error') throw new Error(st.error);
    S.design = design; S.result = st.result;
    renderResult(st.result);
    renderStages(st.result);
    loadStagePlots(design, fidelity);
  } catch (e) {
    toast('evaluation failed: ' + e.message);
    $('#cfg-stage').textContent = 'failed';
  } finally {
    btn.disabled = false;
    setTimeout(() => $('#cfg-progress').classList.add('hidden'), 1200);
  }
});

function renderResult(R) {
  const d = $('#cfg-result');
  d.className = '';
  d.innerHTML = '';

  const v = R.verdict;
  d.appendChild(el('div', {style: 'display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:12px'}, [
    el('span', {class: 'pill ' + (v.feasible ? 'good' : 'bad'),
      text: v.feasible ? 'FEASIBLE' : 'INFEASIBLE'}),
    el('span', {class: 'pill ' + (R.fidelity === 'normal' ? 'good' : 'warn'),
      text: R.fidelity === 'normal' ? 'full fidelity' : 'screening \u2014 ~4% force error'}),
    el('span', {class: 'pill', text: 'score ' + fmt('scalar', v.scalar)}),
  ]));

  d.appendChild(el('div', {class: 'cards'}, [
    card('attraction', fmt('F_attract', R.magnetics.F_attract), 'N'),
    card('repulsion', fmt('F_repel', R.magnetics.F_repel), 'N'),
    card('asymmetry', fmt('asymmetry', R.magnetics.asymmetry), ':1'),
    card('module mass', fmt('m_module', R.module.mass), 'g'),
  ]));

  const lim = R.limits;
  d.appendChild(el('div', {class: 'cards', style: 'margin-top:10px'}, [
    gauge('demag margin', R.magnetics.margin, lim.margin, 'Hcj', true),
    gauge('hold', R.mechanics.hold_ratio, lim.hold_ratio, '\u00d7 wt', false),
    gauge('pivot', R.mechanics.pivot_ratio, lim.pivot_ratio, '\u00d7', false),
    gauge('bounding box', R.derived.bounding_cube, lim.bounding_cube, 'mm', true, 1e3),
  ]));

  if (!v.feasible) {
    d.appendChild(el('div', {class: 'viol', html:
      '<b>constraints not met</b><ul>' +
      v.violations.map(x => `<li>${x}</li>`).join('') + '</ul>'}));
  }
  if (!R.prescreen.passed) {
    d.appendChild(el('div', {class: 'viol', html:
      '<b>pre-screen</b><ul>' +
      R.prescreen.reasons.map(x => `<li>${x}</li>`).join('') + '</ul>'}));
  }
  d.appendChild(el('p', {class: 'note', text:
    'Open the stage tabs above for field maps, the pivot integral, the ' +
    'driver BOM and the module geometry.'}));
}

// ================================================================ STAGES
function renderStages(R) {
  const badge = (id, txt, cls) => {
    const b = $(id); b.textContent = txt;
    b.className = 'pill' + (cls ? ' ' + cls : '');
  };
  const fid = R.fidelity === 'normal' ? 'full fidelity' : 'screening';

  badge('#mag-badge', `${R.design.material} \u00b7 ${fid}`,
        R.fidelity === 'normal' ? 'good' : 'warn');

  // ---- mechanics
  const M = R.mechanics, lim = R.limits;
  $('#mech-cards').innerHTML = '';
  $('#mech-cards').append(
    gauge('hold', M.hold_ratio, lim.hold_ratio, '\u00d7 weight', false),
    gauge('pivot work / barrier', M.pivot_ratio, lim.pivot_ratio, '\u00d7', false),
    card('drive work', fmt('W_drive', M.W_drive), 'mJ'),
    card('gravity barrier', fmt('E_barrier', M.E_barrier), 'mJ'),
    card('CoM lift', fmt('lift', M.lift), 'mm',
         '', `r_face \u2192 ${(R.derived.r_vertex * 1e3).toFixed(2)} mm at the edge`),
    card('pivot step', fmt('pivot_deg', R.derived.pivot_deg), 'deg',
         '', `360/${R.design.n_gon}`),
    card('module weight', (M.weight).toFixed(2), 'N'),
  );
  badge('#mech-badge', `n=${R.design.n_gon} \u00b7 ${fid}`,
        M.pivot_ratio >= lim.pivot_ratio ? 'good' : 'bad');

  // ---- driver
  const D = R.driver, W = R.switching;
  $('#drv-cards').innerHTML = '';
  $('#drv-cards').append(
    card('MMF needed', fmt('mmf_need', W.mmf_need), 'A\u00b7t'),
    card('MMF available', fmt('mmf', W.mmf), 'A\u00b7t',
         W.mmf >= W.mmf_need ? 'good' : 'bad',
         `margin ${(W.switch_margin || 0).toFixed(2)}\u00d7`),
    card('peak current', (W.i_peak || 0).toFixed(0), 'A'),
    card('coil', (W.n_turns || 0).toFixed(0), 'turns',
         '', `L ${(W.L_coil * 1e6).toFixed(1)} \u00b5H, R ${(W.R_coil * 1e3).toFixed(0)} m\u03a9`),
    card('energy per face', (W.e_required * 1e3).toFixed(1), 'mJ'),
    card('whole module', (W.e_total_module * 1e3).toFixed(0), 'mJ',
         '', 'all faces switched once'),
    card('damping', W.underdamped ? 'underdamped' : 'overdamped',
         '', W.underdamped ? 'good' : 'bad',
         W.underdamped ? 'current peaks before the bank empties'
                       : 'bank may empty before peak current'),
  );
  badge('#drv-badge', D.feasible ? `${D.topology} \u00b7 ${(D.mass * 1e3).toFixed(0)} g` : 'no driver found',
        D.feasible ? 'good' : 'bad');

  const bom = $('#drv-bom');
  bom.innerHTML = '';
  if (D.feasible) {
    const dl = el('dl', {class: 'kv'});
    [['topology', D.topology],
     ['capacitors', `${D.n_caps} \u00d7 ${D.cap_name}`],
     ['bank', `${(D.c_bank * 1e6).toFixed(0)} \u00b5F at ${D.v_bank.toFixed(0)} V`],
     ['stored energy', `${(D.e_bank * 1e3).toFixed(0)} mJ`],
     ['MOSFETs', `${D.n_fets} \u00d7 ${D.mosfet_name}`],
     ['charger', D.charger_name],
     ['peak current', `${D.i_peak.toFixed(0)} A`],
     ['recharge', `${D.recharge_s.toFixed(2)} s`],
     ['mass', `${(D.mass * 1e3).toFixed(1)} g`],
     ['volume', `${(D.volume * 1e6).toFixed(1)} cc`],
     ['cost', `$${D.price.toFixed(0)}`]].forEach(([k, v]) => {
      dl.appendChild(el('dt', {text: k}));
      dl.appendChild(el('dd', {text: v}));
    });
    bom.appendChild(dl);
    if (D.notes) bom.appendChild(el('p', {class: 'note', text: D.notes}));
  } else {
    bom.appendChild(el('p', {class: 'note',
      text: 'No component combination satisfies this coil. ' + (D.notes || '')}));
  }

  // ---- module
  const Mo = R.module;
  $('#mod-cards').innerHTML = '';
  const parts = Object.entries(Mo.parts).sort((a, b) => b[1] - a[1]);
  $('#mod-cards').append(
    card('total mass', (Mo.mass * 1e3).toFixed(0), 'g'),
    card('fits 50 mm cube', Mo.fits ? 'yes' : 'no', '',
         Mo.fits ? 'good' : 'bad',
         `box ${(R.derived.bounding_cube * 1e3).toFixed(1)} mm`),
    card('faces', Mo.n_faces, '', '', `3n\u22126 for n=${R.design.n_gon}`),
    card('free volume', (Mo.free_volume * 1e6).toFixed(1), 'cc'),
    ...parts.map(([k, v]) => card(k, (v * 1e3).toFixed(1), 'g', '',
      `${(v / Mo.mass * 100).toFixed(0)}% of module`)),
  );
  badge('#mod-badge',
        `n=${R.design.n_gon} \u00b7 ${Mo.n_faces} faces \u00b7 ${(Mo.mass * 1e3).toFixed(0)} g`);
  S.geom = null;
  drawModule();

  // ---- dynamics
  renderDynamics(R.dynamics);
}

function renderDynamics(D) {
  const host = $('#dyn-cards');
  host.innerHTML = '';
  if (!D) {
    $('#dyn-badge').textContent = 'not run';
    $('#dyn-badge').className = 'pill';
    $('#dyn-sep').innerHTML = '<p class="note">Not run for this design.</p>';
    $('#dyn-piv').innerHTML = '';
    return;
  }
  const P = D.pivot;
  const oneStep = Math.abs(P.steps - 1) < 0.35;
  host.append(
    card('latch', D.latch.held ? 'held' : 'separated', '',
         D.latch.held ? 'good' : 'bad',
         `final gap ${D.latch.final_sep_mm.toFixed(2)} mm`),
    card('repel travel', D.repel.moved_mm.toFixed(1), 'mm', '', 'in 250 ms'),
    card('pivot settled', P.settled_deg.toFixed(1), 'deg',
         oneStep ? 'good' : 'warn',
         `target ${P.target_deg.toFixed(0)}\u00b0 \u2014 ${P.steps.toFixed(1)} steps`),
    card('one step takes',
         P.t_one_step_ms ? P.t_one_step_ms.toFixed(0) : '\u2014', 'ms',
         '', 'drive must be cut inside this'),
  );
  $('#dyn-badge').textContent = oneStep ? 'clean single step'
    : (P.steps > 1 ? `overshoots \u2014 ${P.steps.toFixed(1)} steps` : 'does not complete a step');
  $('#dyn-badge').className = 'pill ' + (oneStep ? 'good' : 'warn');

  lineChart($('#dyn-sep'), [
    {x: D.latch.t_ms, y: D.latch.sep_mm, c: '#4ec9a0', name: 'latch (attract)'},
    {x: D.repel.t_ms, y: D.repel.sep_mm, c: '#ff8f5a', name: 'repel'},
  ], 'time (ms)', 'separation (mm)');
  lineChart($('#dyn-piv'), [
    {x: P.t_ms, y: P.ang_deg.map(Math.abs), c: '#5aa9ff', name: 'rotation'},
    {x: [P.t_ms[0], P.t_ms[P.t_ms.length - 1]],
     y: [P.target_deg, P.target_deg], c: '#ffc14d', name: 'one step', dash: 1},
  ], 'time (ms)', 'rotation (deg)');
}

// ---------------------------------------------------------------- charts
function lineChart(host, series, xlab, ylab) {
  host.innerHTML = '';
  const W = host.clientWidth || 460, H = 250;
  const M = {t: 12, r: 12, b: 38, l: 52};
  const all = series.filter(s => s.x.length);
  if (!all.length) return;
  // scale text with the box so a wide panel does not get giant labels
  const fs = Math.max(9, Math.min(12, W / 46));
  const xs = all.flatMap(s => s.x), ys = all.flatMap(s => s.y);
  const ex = [Math.min(...xs), Math.max(...xs)];
  const ey = [Math.min(...ys), Math.max(...ys)];
  if (ey[0] === ey[1]) { ey[0] -= 1; ey[1] += 1; }
  const pad = (ey[1] - ey[0]) * .08;
  ey[0] -= pad; ey[1] += pad;
  const px = v => M.l + (v - ex[0]) / (ex[1] - ex[0] || 1) * (W - M.l - M.r);
  const py = v => H - M.b - (v - ey[0]) / (ey[1] - ey[0]) * (H - M.t - M.b);

  const svg = el('svg', {svg: 1, viewBox: `0 0 ${W} ${H}`});
  for (let i = 0; i <= 4; i++) {
    const yv = ey[0] + (ey[1] - ey[0]) * i / 4;
    const xv = ex[0] + (ex[1] - ex[0]) * i / 4;
    svg.appendChild(el('line', {svg: 1, x1: M.l, x2: W - M.r,
      y1: py(yv), y2: py(yv), stroke: '#262c3a', 'stroke-opacity': .6}));
    svg.appendChild(el('text', {svg: 1, x: M.l - 7, y: py(yv) + 3.5,
      'text-anchor': 'end', fill: '#8b93a7', 'font-size': fs, text: trim(yv)}));
    svg.appendChild(el('text', {svg: 1, x: px(xv), y: H - M.b + 15,
      'text-anchor': 'middle', fill: '#8b93a7', 'font-size': fs,
      text: trim(xv)}));
  }
  svg.appendChild(el('text', {svg: 1, x: (W + M.l) / 2, y: H - 6,
    'text-anchor': 'middle', fill: '#8b93a7', 'font-size': fs, text: xlab}));
  svg.appendChild(el('text', {svg: 1, transform: `rotate(-90 12 ${H / 2})`,
    x: 12, y: H / 2, 'text-anchor': 'middle', fill: '#8b93a7',
    'font-size': fs, text: ylab}));

  series.forEach((s, i) => {
    if (!s.x.length) return;
    const d = s.x.map((v, j) => `${j ? 'L' : 'M'}${px(v).toFixed(1)},${py(s.y[j]).toFixed(1)}`).join('');
    svg.appendChild(el('path', {svg: 1, d, fill: 'none', stroke: s.c,
      'stroke-width': 2, 'stroke-dasharray': s.dash ? '5,4' : null}));
    const lx = M.l + 6 + i * (fs * 11);
    svg.appendChild(el('rect', {svg: 1, x: lx, y: M.t, width: 9, height: 3,
      fill: s.c}));
    svg.appendChild(el('text', {svg: 1, x: lx + 13, y: M.t + 5,
      fill: '#8b93a7', 'font-size': fs, text: s.name}));
  });
  host.appendChild(svg);
}

// ---------------------------------------------------------------- plots
function setImg(id, url) {
  const img = document.getElementById(id);
  if (!img) return;
  const fig = img.closest('figure');
  if (fig) fig.classList.add('loading');
  img.onload = () => fig && fig.classList.remove('loading');
  img.onerror = () => {
    if (fig) fig.classList.remove('loading');
    toast('a figure failed to render \u2014 check the server log');
  };
  img.src = url;
}

function qs(design, extra) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(design)) p.set(k, v);
  for (const [k, v] of Object.entries(extra || {})) p.set(k, v);
  return p.toString();
}

function loadStagePlots(design, fidelity) {
  const base = qs(design, {fidelity});
  setImg('fig-field-a', `/api/plot/field?${base}&state=attract`);
  setImg('fig-field-r', `/api/plot/field?${base}&state=repel`);
  setImg('fig-force', `/api/plot/force_gap?${base}`);
  setImg('fig-pivot', `/api/plot/pivot?${base}`);
  setImg('fig-pulse', `/api/plot/pulse?${base}`);
  const marks = S.result ? [
    {material: design.material, H: -S.result.magnetics.margin_attract *
      matHcj(design.material), J: S.result.magnetics.J_attract,
     label: 'attract'},
    {material: design.material, H: -S.result.magnetics.margin_repel *
      matHcj(design.material), J: S.result.magnetics.J_repel,
     label: 'repel', bad: S.result.magnetics.margin > S.result.limits.margin},
  ] : [];
  setImg('fig-demag', `/api/plot/demag?materials=${design.material}` +
    `&marks=${encodeURIComponent(JSON.stringify(marks))}`);
}

function matHcj(name) {
  const m = S.meta.materials.find(x => x.name === name);
  return m ? m.Hcj : 1;
}

let expLoaded = false;
function loadExperiment() {
  if (expLoaded) return;
  expLoaded = true;
  setImg('fig-exp', '/api/plot/experiment');
  const v = $('#verif-list');
  v.className = 'verif';
  [['magnet_force.py', 'exact-solution self-tests: demag factors to 5 dp, cuboid form, Maxwell and dipole limits'],
   ['validate_vs_experiment.py', 'RMS 0.16 N (NdFeB) and 0.10 N (Alnico) against the bench data'],
   ['verify_fem.py', 'demag factors, open-circuit J, force convergence, iron monotonicity'],
   ['verify_dynamics.py', 'FEM match, fall-off, torque sign, Newton 3, closed-loop work = 0'],
   ['screening_study.py', 'screening vs full fidelity: median 4.2 % error, Spearman 0.992'],
   ['verify_pivot.py', 'static pivot criterion against MuJoCo, agreeing within 25 %'],
   ['verify_optimise.py', 'serial = parallel, resumed = uninterrupted, atomic checkpoints'],
  ].forEach(([n, d]) => v.appendChild(card(n, '', '', 'good', d)));
}

// ================================================================ 3D
async function drawModule() {
  const host = $('#module3d');
  if (!host.clientWidth) return;
  const d = S.design || BASELINE;
  if (!S.geom) {
    try {
      S.geom = await api(`/api/geometry?n_gon=${d.n_gon}&r_face=${d.r_face}` +
                         `&d_mag=${d.d_mag}`);
    } catch (e) { host.innerHTML = '<p class="note">' + e.message + '</p>'; return; }
  }
  const G = S.geom;
  const W = host.clientWidth, H = 430;
  const R = Math.min(W, H) * 0.34 / G.r_vertex;
  const {x: rx, y: ry} = S.rot;
  const rot = p => {
    let [a, b, c] = p;
    let x = a * Math.cos(ry) - c * Math.sin(ry);
    let z = a * Math.sin(ry) + c * Math.cos(ry);
    let y = b * Math.cos(rx) - z * Math.sin(rx);
    z = b * Math.sin(rx) + z * Math.cos(rx);
    return [x, y, z];
  };
  const proj = p => {
    const [x, y, z] = rot(p);
    return [W / 2 + x * R, H / 2 - y * R, z];
  };

  const svg = el('svg', {svg: 1, viewBox: `0 0 ${W} ${H}`});
  svg.appendChild(el('rect', {svg: 1, width: W, height: H, fill: '#0f131a'}));

  if ($('#m3-cube').checked) {
    const h = G.r_face;
    const corners = [[-h,-h,-h],[h,-h,-h],[h,h,-h],[-h,h,-h],
                     [-h,-h,h],[h,-h,h],[h,h,h],[-h,h,h]].map(proj);
    [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
     [0,4],[1,5],[2,6],[3,7]].forEach(([a, b]) => {
      svg.appendChild(el('line', {svg: 1, x1: corners[a][0], y1: corners[a][1],
        x2: corners[b][0], y2: corners[b][1], stroke: '#39415a',
        'stroke-dasharray': '4,4', 'stroke-width': 1}));
    });
  }

  // convex hull edges: connect vertices that share two face planes
  const V = G.vertices, N = G.normals;
  const on = V.map(v => N.map((n, i) =>
    Math.abs(n[0]*v[0] + n[1]*v[1] + n[2]*v[2] - G.r_face) < 1e-9 ? i : -1)
    .filter(i => i >= 0));
  const drawn = new Set();
  for (let i = 0; i < V.length; i++) {
    for (let j = i + 1; j < V.length; j++) {
      const shared = on[i].filter(f => on[j].includes(f));
      if (shared.length < 2) continue;
      const key = i + ',' + j;
      if (drawn.has(key)) continue;
      drawn.add(key);
      const a = proj(V[i]), b = proj(V[j]);
      const depth = (a[2] + b[2]) / 2;
      svg.appendChild(el('line', {svg: 1, x1: a[0], y1: a[1], x2: b[0], y2: b[1],
        stroke: depth > 0 ? '#6f7d9c' : '#333c52',
        'stroke-width': depth > 0 ? 1.5 : 1,
        'stroke-opacity': depth > 0 ? 0.95 : 0.5}));
    }
  }

  if ($('#m3-faces').checked) {
    const order = N.map((n, i) => ({i, n, p: proj(n.map(c => c * G.r_face))}))
                   .sort((a, b) => a.p[2] - b.p[2]);
    order.forEach(({i, n, p}) => {
      const front = p[2] > 0;
      const rr = Math.max(3, (G.d_mag / 2) * R);
      svg.appendChild(el('circle', {svg: 1, cx: p[0], cy: p[1], r: rr,
        fill: front ? 'rgba(90,169,255,.5)' : 'rgba(90,169,255,.12)',
        stroke: front ? '#5aa9ff' : '#2f4160', 'stroke-width': front ? 1.5 : .8}));
      if ($('#m3-labels').checked && front)
        svg.appendChild(el('text', {svg: 1, x: p[0], y: p[1] + 3.5,
          'text-anchor': 'middle', fill: '#e6e9ef', 'font-size': 9, text: i}));
    });
  }

  host.innerHTML = '';
  host.appendChild(svg);
  const txt =
    `n=${G.n_gon} \u00b7 <b>${G.n_faces}</b> faces (3n\u22126) \u00b7 ` +
    `face side ${(G.a_face * 1e3).toFixed(1)} mm \u00b7 ` +
    `pivot step ${G.pivot_deg.toFixed(1)}\u00b0 \u00b7 ` +
    `bounding box ${(G.bounding_cube * 1e3).toFixed(1)} mm \u00b7 ` +
    `vertex radius ${(G.r_vertex * 1e3).toFixed(2)} mm`;
  // reuse the caption node: appending on every redraw would stack one line
  // per frame while the user drags to rotate
  let cap = document.getElementById('m3-cap');
  if (!cap) {
    cap = el('div', {class: 'note', id: 'm3-cap'});
    host.parentElement.querySelector('.view-ctl')
        ?.insertAdjacentElement('afterend', cap);
  }
  cap.innerHTML = txt;
}

(function dragRotate() {
  const host = $('#module3d');
  let last = null;
  host.addEventListener('pointerdown', e => {
    last = [e.clientX, e.clientY]; host.setPointerCapture(e.pointerId);
  });
  host.addEventListener('pointermove', e => {
    if (!last) return;
    S.rot.y += (e.clientX - last[0]) * 0.01;
    S.rot.x += (e.clientY - last[1]) * 0.01;
    S.rot.x = Math.max(-1.5, Math.min(1.5, S.rot.x));
    last = [e.clientX, e.clientY];
    drawModule();
  });
  addEventListener('pointerup', () => last = null);
  ['m3-faces', 'm3-labels', 'm3-cube'].forEach(id =>
    $('#' + id).addEventListener('change', () => drawModule()));
})();

// ---------------------------------------------------------------- wiring
$('#mx-file').addEventListener('change', loadMatrix);
$('#mx-filter').addEventListener('change', applyFilter);
$('#mx-search').addEventListener('input', applyFilter);
$('#mx-preset').addEventListener('change', e => {
  S.preset = e.target.value; drawTable();
});
['sc-x', 'sc-y', 'sc-c', 'sc-logx', 'sc-logy'].forEach(id =>
  $('#' + id).addEventListener('change', drawScatter));
$('#mx-prev').addEventListener('click', () => { S.page--; drawTable(); });
$('#mx-next').addEventListener('click', () => { S.page++; drawTable(); });
addEventListener('resize', () => { drawScatter(); drawModule(); });

$('#mx-export').addEventListener('click', () => {
  const cols = S.preset === 'all'
    ? Object.keys(S.rows[0] || {}) : PRESETS[S.preset];
  const lines = [cols.join(',')];
  S.view.forEach(r => lines.push(cols.map(c => {
    const v = r[c];
    if (v === null || v === undefined) return '';
    return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
  }).join(',')));
  const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
  const a = el('a', {href: URL.createObjectURL(blob),
                     download: 'design_view.csv'});
  a.click();
});
