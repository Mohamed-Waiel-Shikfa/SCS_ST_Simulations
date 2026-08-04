/* A small 3-D renderer on a 2-D canvas.
 *
 * Why not a library.  Everything drawn here is made of cylinders, tubes,
 * boxes, a convex hull and point clouds, all opaque or simply blended, at a
 * few thousand triangles.  A painter's-algorithm renderer covers that in a
 * few hundred lines, has no network dependency, and - the reason that
 * actually decided it - lets the module viewer, the field viewer and the
 * MuJoCo playback share one camera, one lighting model and one set of orbit
 * controls, so the three views are visually consistent and behave the same
 * way under the mouse.
 *
 * Faces are depth-sorted by centroid.  That is exact for the convex parts and
 * wrong only where two long thin surfaces interpenetrate, which does not
 * happen in these scenes.
 */

export class Scene3D {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.items = [];
    this.yaw = opts.yaw ?? -0.62;
    this.pitch = opts.pitch ?? 0.42;
    // zoom is a MULTIPLIER on the scale that fit() computes, so it must
    // start at 1 - otherwise fit() frames the object correctly and the zoom
    // then immediately pushes it off the edges
    this.zoom = opts.zoom ?? 1;
    this.target = opts.target ?? [0, 0, 0];
    this.scale = opts.scale ?? 1;
    this.bg = opts.bg ?? '#0e1218';
    this.showAxes = opts.showAxes ?? true;
    this.light = norm([0.45, 0.7, 0.55]);
    this._bindControls();
  }

  clear() { this.items = []; }

  /* ---- primitives ---------------------------------------------------- */
  addTriangles(tris, colour, opts = {}) {
    this.items.push({ kind: 'tri', tris, colour, ...opts });
  }

  addCylinder(centre, axis, r, h, colour, opts = {}) {
    this.addTriangles(cylinderTris(centre, axis, r, r, h, opts.seg ?? 20),
      colour, opts);
  }

  addTube(centre, axis, rIn, rOut, h, colour, opts = {}) {
    this.addTriangles(tubeTris(centre, axis, rIn, rOut, h, opts.seg ?? 20),
      colour, opts);
  }

  addBox(centre, axes, half, colour, opts = {}) {
    this.addTriangles(boxTris(centre, axes, half), colour, opts);
  }

  addHull(vertices, colour, opts = {}) {
    this.addTriangles(hullTris(vertices), colour, opts);
  }

  addPoints(pts, colours, opts = {}) {
    this.items.push({ kind: 'pts', pts, colours, size: opts.size ?? 2,
      alpha: opts.alpha ?? 1, ...opts });
  }

  addLines(segs, colour, opts = {}) {
    this.items.push({ kind: 'lines', segs, colour, width: opts.width ?? 1,
      alpha: opts.alpha ?? 1, ...opts });
  }

  /* ---- camera -------------------------------------------------------- */
  fit(pts, pad = 1.35) {
    if (!pts.length) return;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) for (let i = 0; i < 3; i++) {
      if (p[i] < lo[i]) lo[i] = p[i];
      if (p[i] > hi[i]) hi[i] = p[i];
    }
    this.target = [0, 1, 2].map(i => (lo[i] + hi[i]) / 2);
    const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9);
    this.scale = 1 / (span * pad);
  }

  _basis() {
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    // right, up, forward (camera looks down -forward)
    return {
      r: [cy, 0, -sy],
      u: [sy * sp, cp, cy * sp],
      f: [sy * cp, -sp, cy * cp],
    };
  }

  project(p) {
    const b = this._basis();
    const d = [p[0] - this.target[0], p[1] - this.target[1],
      p[2] - this.target[2]];
    const x = dot(d, b.r), y = dot(d, b.u), z = dot(d, b.f);
    const W = this.canvas.width, H = this.canvas.height;
    const s = Math.min(W, H) * this.scale * this.zoom;
    return [W / 2 + x * s, H / 2 - y * s, z];
  }

  _bindControls() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('pointerdown', e => {
      drag = { x: e.clientX, y: e.clientY };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', e => {
      if (!drag) return;
      this.yaw += (e.clientX - drag.x) * 0.008;
      this.pitch = Math.max(-1.5, Math.min(1.5,
        this.pitch + (e.clientY - drag.y) * 0.008));
      drag = { x: e.clientX, y: e.clientY };
      this.draw();
    });
    const stop = e => { drag = null; try { c.releasePointerCapture(e.pointerId); } catch (_) {} };
    c.addEventListener('pointerup', stop);
    c.addEventListener('pointercancel', stop);
    c.addEventListener('wheel', e => {
      e.preventDefault();
      this.zoom = Math.max(0.25, Math.min(9,
        this.zoom * (e.deltaY > 0 ? 0.9 : 1.111)));
      this.draw();
    }, { passive: false });
  }

  /* ---- drawing ------------------------------------------------------- */
  draw() {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) {
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = this.bg;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    const prims = [];
    for (const it of this.items) {
      if (it.hidden) continue;
      if (it.kind === 'tri') {
        for (const t of it.tris) {
          const a = this.project(t[0]), b = this.project(t[1]),
            c = this.project(t[2]);
          const n = norm(cross(sub(t[1], t[0]), sub(t[2], t[0])));
          const lam = 0.34 + 0.66 * Math.abs(dot(n, this.light));
          prims.push({ z: (a[2] + b[2] + c[2]) / 3, kind: 'tri',
            pts: [a, b, c], colour: shade(it.colour, lam),
            alpha: it.alpha ?? 1, edge: it.edge });
        }
      } else if (it.kind === 'pts') {
        for (let i = 0; i < it.pts.length; i++) {
          const p = this.project(it.pts[i]);
          prims.push({ z: p[2], kind: 'pt', pts: [p], size: it.size,
            colour: it.colours ? it.colours[i] : it.colour,
            alpha: it.alpha });
        }
      } else if (it.kind === 'lines') {
        for (const s of it.segs) {
          const a = this.project(s[0]), b = this.project(s[1]);
          prims.push({ z: (a[2] + b[2]) / 2, kind: 'line', pts: [a, b],
            colour: s[2] || it.colour, alpha: it.alpha, width: it.width });
        }
      }
    }
    prims.sort((p, q) => q.z - p.z);

    for (const p of prims) {
      ctx.globalAlpha = p.alpha ?? 1;
      if (p.kind === 'tri') {
        ctx.beginPath();
        ctx.moveTo(p.pts[0][0], p.pts[0][1]);
        ctx.lineTo(p.pts[1][0], p.pts[1][1]);
        ctx.lineTo(p.pts[2][0], p.pts[2][1]);
        ctx.closePath();
        ctx.fillStyle = p.colour;
        ctx.fill();
        if (p.edge) {
          ctx.strokeStyle = p.edge;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      } else if (p.kind === 'pt') {
        ctx.fillStyle = p.colour;
        ctx.fillRect(p.pts[0][0] - p.size / 2, p.pts[0][1] - p.size / 2,
          p.size, p.size);
      } else {
        ctx.strokeStyle = p.colour;
        ctx.lineWidth = p.width;
        ctx.beginPath();
        ctx.moveTo(p.pts[0][0], p.pts[0][1]);
        ctx.lineTo(p.pts[1][0], p.pts[1][1]);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    if (this.showAxes) this._drawAxes();
  }

  _drawAxes() {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const ox = 46 * dpr, oy = this.canvas.height - 42 * dpr, L = 26 * dpr;
    const b = this._basis();
    const axes = [[[1, 0, 0], '#e05a5a', 'x'], [[0, 1, 0], '#5ae08a', 'y'],
      [[0, 0, 1], '#5a9ae0', 'z']];
    ctx.save();
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;
    for (const [v, col, name] of axes) {
      const x = dot(v, b.r) * L, y = -dot(v, b.u) * L;
      ctx.strokeStyle = col;
      ctx.fillStyle = col;
      ctx.lineWidth = 1.6 * dpr;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + x, oy + y);
      ctx.stroke();
      ctx.fillText(name, ox + x * 1.28 - 3 * dpr, oy + y * 1.28 + 4 * dpr);
    }
    ctx.restore();
  }
}

/* ---- vector helpers -------------------------------------------------- */
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export function norm(a) {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}

export function frameFrom(axis) {
  const n = norm(axis);
  let t = Math.abs(n[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const u = norm(cross(t, n));
  const v = cross(n, u);
  return [u, v, n];
}

function shade(hex, lam) {
  const c = hex.replace('#', '');
  const r = parseInt(c.slice(0, 2), 16), g = parseInt(c.slice(2, 4), 16),
    b = parseInt(c.slice(4, 6), 16);
  const f = x => Math.max(0, Math.min(255, Math.round(x * lam)));
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}

/* ---- primitive builders ---------------------------------------------- */
export function cylinderTris(centre, axis, r0, r1, h, seg = 20) {
  const [u, v, n] = frameFrom(axis);
  const tris = [];
  const ring = (rad, off) => {
    const out = [];
    for (let i = 0; i < seg; i++) {
      const a = 2 * Math.PI * i / seg;
      out.push(add(add(centre, mul(n, off)),
        add(mul(u, rad * Math.cos(a)), mul(v, rad * Math.sin(a)))));
    }
    return out;
  };
  const A = ring(r0, -h / 2), B = ring(r1, h / 2);
  const ca = add(centre, mul(n, -h / 2)), cb = add(centre, mul(n, h / 2));
  for (let i = 0; i < seg; i++) {
    const j = (i + 1) % seg;
    tris.push([A[i], A[j], B[j]], [A[i], B[j], B[i]]);
    tris.push([ca, A[j], A[i]], [cb, B[i], B[j]]);
  }
  return tris;
}

export function tubeTris(centre, axis, rIn, rOut, h, seg = 20) {
  const [u, v, n] = frameFrom(axis);
  const tris = [];
  const pt = (rad, off, a) => add(add(centre, mul(n, off)),
    add(mul(u, rad * Math.cos(a)), mul(v, rad * Math.sin(a))));
  for (let i = 0; i < seg; i++) {
    const a0 = 2 * Math.PI * i / seg, a1 = 2 * Math.PI * (i + 1) / seg;
    const oi0 = pt(rOut, -h / 2, a0), oi1 = pt(rOut, -h / 2, a1);
    const oo0 = pt(rOut, h / 2, a0), oo1 = pt(rOut, h / 2, a1);
    const ii0 = pt(rIn, -h / 2, a0), ii1 = pt(rIn, -h / 2, a1);
    const io0 = pt(rIn, h / 2, a0), io1 = pt(rIn, h / 2, a1);
    tris.push([oi0, oi1, oo1], [oi0, oo1, oo0]);
    tris.push([ii1, ii0, io0], [ii1, io0, io1]);
    tris.push([oi1, oi0, ii0], [oi1, ii0, ii1]);
    tris.push([oo0, oo1, io1], [oo0, io1, io0]);
  }
  return tris;
}

export function boxTris(centre, axes, half) {
  const [u, v, n] = axes;
  const c = (a, b, d) => add(centre,
    add(add(mul(u, a * half[0]), mul(v, b * half[1])), mul(n, d * half[2])));
  const p = [c(-1, -1, -1), c(1, -1, -1), c(1, 1, -1), c(-1, 1, -1),
    c(-1, -1, 1), c(1, -1, 1), c(1, 1, 1), c(-1, 1, 1)];
  const f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6],
    [1, 2, 6, 5], [0, 4, 7, 3]];
  const tris = [];
  for (const q of f) {
    tris.push([p[q[0]], p[q[1]], p[q[2]]], [p[q[0]], p[q[2]], p[q[3]]]);
  }
  return tris;
}

/* Convex hull of a point set (gift wrapping over faces).  The module has at
 * most a few dozen vertices, so an O(n^4) construction is instant and needs
 * no library. */
export function hullTris(pts) {
  const n = pts.length;
  if (n < 4) return [];
  const tris = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        const nrm = cross(sub(pts[j], pts[i]), sub(pts[k], pts[i]));
        const len = Math.hypot(nrm[0], nrm[1], nrm[2]);
        if (len < 1e-12) continue;
        const u = mul(nrm, 1 / len);
        const d = dot(u, pts[i]);
        let pos = 0, neg = 0;
        for (let m = 0; m < n; m++) {
          const s = dot(u, pts[m]) - d;
          if (s > 1e-9) pos++;
          else if (s < -1e-9) neg++;
        }
        if (pos && neg) continue;
        tris.push(neg ? [pts[i], pts[j], pts[k]] : [pts[i], pts[k], pts[j]]);
      }
    }
  }
  return tris;
}

export function quatRotate(q, p) {
  const [w, x, y, z] = q;
  const R = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
  return [dot(R[0], p), dot(R[1], p), dot(R[2], p)];
}

/* Perceptually ordered colour ramp for field magnitude. */
export function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[0.05, 0.02, 0.20], [0.25, 0.10, 0.52], [0.55, 0.17, 0.55],
    [0.85, 0.30, 0.36], [0.99, 0.60, 0.20], [0.99, 0.94, 0.72]];
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const c = stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f);
  return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
}
