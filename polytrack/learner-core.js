// How the fly learns to drive. Pure functions and classes, no DOM, so they run
// in the page, in a worker and in Node tests.
//
// - fitDecoder: weighted ridge regression from neuron features to steering
//   and throttle (imitation of your laps and corrections).
// - RouteMemory: stored views from your laps with what you did there; the fly
//   does what you did where the view matches best (like ants follow routes).
// - rewardExamples / nudgeDecoder: dopamine and pain. The fly wobbles its
//   steering a little; a reward reinforces the recent wobbles (eligibility
//   trace), a punishment reverses them.

export const ELIGIBILITY_FRAMES = 75; // how far back a reward reaches (1.5 s)
const ELIGIBILITY_DECAY = 0.97; // per frame, so recent frames count most

// Mean and standard deviation of every feature (unweighted).
function moments(X, n, d, rows) {
  const mean = new Float64Array(d);
  const std = new Float64Array(d);
  for (const i of rows) for (let f = 0; f < d; f++) mean[f] += X[i * d + f];
  for (let f = 0; f < d; f++) mean[f] /= rows.length;
  for (const i of rows) for (let f = 0; f < d; f++) std[f] += (X[i * d + f] - mean[f]) ** 2;
  for (let f = 0; f < d; f++) std[f] = Math.sqrt(std[f] / rows.length) || 1;
  return { mean, std };
}

function solve(A, b) {
  const d = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < d; c++) {
    let p = c;
    for (let r = c + 1; r < d; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < d; r++) {
      if (r === c) continue;
      const m = M[r][c] / M[c][c];
      if (m) for (let j = c; j <= d; j++) M[r][j] -= m * M[c][j];
    }
  }
  return M.map((row, i) => row[d] / row[i]);
}

// data: { X: Float32Array (n*d), n, d, steer, throttle, weight } (typed arrays).
// Every 5th block of 100 frames is held out to score the result (R²).
export function fitDecoder(data, { ridgePerFrame = 0.5, stride = 2 } = {}) {
  const { X, n, d } = data;
  const all = [...Array(n).keys()];
  const heldOut = (i) => Math.floor(i / 100) % 5 === 4;
  const train = all.filter((i) => !heldOut(i) && i % stride === 0);
  const test = all.filter(heldOut);
  const { mean, std } = moments(X, n, d, train);
  const D = d + 1;
  const row = (i, out) => {
    out[0] = 1;
    for (let f = 0; f < d; f++) out[f + 1] = (X[i * d + f] - mean[f]) / std[f];
    return out;
  };
  const A = Array.from({ length: D }, () => new Float64Array(D));
  const bs = new Float64Array(D);
  const bt = new Float64Array(D);
  const r = new Float64Array(D);
  let wsum = 0;
  for (const i of train) {
    row(i, r);
    const w = data.weight[i];
    wsum += w;
    for (let a = 0; a < D; a++) {
      const wa = w * r[a];
      const Aa = A[a];
      for (let b = a; b < D; b++) Aa[b] += wa * r[b];
      bs[a] += wa * data.steer[i];
      bt[a] += wa * data.throttle[i];
    }
  }
  for (let a = 0; a < D; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
  for (let a = 1; a < D; a++) A[a][a] += ridgePerFrame * wsum;
  const decoder = { mean, std, steer: solve(A, bs), throttle: solve(A, bt), nFrames: n };

  const score = {};
  for (const target of ["steer", "throttle"]) {
    const y = data[target];
    const m = test.reduce((s, i) => s + y[i], 0) / (test.length || 1);
    let se = 0;
    let ss = 0;
    for (const i of test) {
      row(i, r);
      const p = clampTarget(target, dot(decoder[target], r));
      se += (p - y[i]) ** 2;
      ss += (y[i] - m) ** 2;
    }
    score[target] = test.length && ss > 0 ? 1 - se / ss : 0;
  }
  decoder.score = score;
  return decoder;
}

const dot = (w, x) => {
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i] * x[i];
  return s;
};
const clampTarget = (target, v) => (target === "steer" ? Math.max(-1, Math.min(1, v)) : Math.max(0, Math.min(1, v)));

// Feature vector -> [1, standardized features].
export function standardize(decoder, x, out = new Float64Array(x.length + 1)) {
  out[0] = 1;
  for (let f = 0; f < x.length; f++) out[f + 1] = (x[f] - decoder.mean[f]) / decoder.std[f];
  return out;
}

export function decode(decoder, x) {
  const z = standardize(decoder, x);
  return {
    steer: clampTarget("steer", dot(decoder.steer, z)),
    throttle: dot(decoder.throttle, z) > 0.5,
  };
}

// Dopamine-style plasticity on the decoder itself: move the steering
// weights along the eligibility trace of recent wobbles (reward > 0) or
// against it (reward < 0). `recent` is oldest-first: { x, eps }.
export function nudgeDecoder(decoder, recent, reward, rate = 0.3) {
  const D = decoder.steer.length;
  const e = new Float64Array(D);
  const z = new Float64Array(D);
  recent.forEach((s, k) => {
    const age = recent.length - 1 - k;
    if (age >= ELIGIBILITY_FRAMES) return;
    standardize(decoder, s.x, z);
    const g = ELIGIBILITY_DECAY ** age * s.eps;
    for (let j = 0; j < D; j++) e[j] += g * z[j];
  });
  const norm = Math.sqrt(dot(e, e));
  if (norm < 1e-9) return 0;
  const step = (reward * rate) / (norm * Math.sqrt(D));
  for (let j = 0; j < D; j++) decoder.steer[j] += step * e[j];
  return Math.abs(step) * norm;
}

// Turn a reward into training examples from the fly's own recent driving: a
// good wobble becomes "do that again", a bad one "do the opposite".
export function rewardExamples(recent, reward) {
  const out = [];
  recent.forEach((s, k) => {
    const age = recent.length - 1 - k;
    if (age >= ELIGIBILITY_FRAMES) return;
    const steer = Math.max(-1, Math.min(1, s.mean + Math.sign(reward) * s.eps));
    out.push({ x: s.x, steer, throttle: s.throttle ? 1 : 0, weight: Math.abs(reward) * ELIGIBILITY_DECAY ** age, source: reward > 0 ? "good" : "bad" });
  });
  return out;
}

// Stored views and what was done there, in driving order. Matching searches
// near the last match (the fly knows roughly where it is on the route) and
// re-localizes over the whole memory every so often.
export class RouteMemory {
  constructor({ maxEntries = 12000, k = 3 } = {}) {
    this.maxEntries = maxEntries;
    this.k = k;
    this.raw = [];
    this.z = [];
    this.steer = [];
    this.throttle = [];
    this.penalty = [];
    this.norm = null;
    this.last = -1;
    this.calls = 0;
  }

  get size() {
    return this.raw.length;
  }

  setNorm(norm) {
    this.norm = norm;
    this.z = this.raw.map((x) => this.#standardize(x));
  }

  #standardize(x) {
    const z = new Float32Array(x.length);
    for (let f = 0; f < x.length; f++) z[f] = (x[f] - this.norm.mean[f]) / this.norm.std[f];
    return z;
  }

  add(x, steer, throttle) {
    if (this.raw.length >= this.maxEntries) {
      // Keep the route's coverage: thin out every other entry.
      for (const key of ["raw", "z", "steer", "throttle", "penalty"]) this[key] = this[key].filter((_, i) => i % 2 === 0);
      this.last = Math.floor(this.last / 2);
    }
    this.raw.push(Float32Array.from(x));
    this.z.push(this.norm ? this.#standardize(x) : null);
    this.steer.push(steer);
    this.throttle.push(throttle);
    this.penalty.push(0);
  }

  act(x) {
    if (!this.norm || !this.raw.length) return null;
    const q = this.#standardize(x);
    const n = this.raw.length;
    const global = this.last < 0 || this.calls++ % 25 === 0;
    const lo = global ? 0 : Math.max(0, this.last - 30);
    const hi = global ? n : Math.min(n, this.last + 150);
    const best = []; // [distance, index], ascending
    for (let i = lo; i < hi; i++) {
      const z = this.z[i];
      let dist = 0;
      for (let f = 0; f < z.length; f++) dist += (z[f] - q[f]) ** 2;
      dist = dist / z.length + 0.5 * this.penalty[i]; // each pain adds distance
      if (best.length < this.k || dist < best[best.length - 1][0]) {
        best.push([dist, i]);
        best.sort((a, b) => a[0] - b[0]);
        if (best.length > this.k) best.pop();
      }
    }
    let ws = 0;
    let steer = 0;
    let throttle = 0;
    for (const [dist, i] of best) {
      const w = 1 / (dist + 1e-3);
      ws += w;
      steer += w * this.steer[i];
      throttle += w * this.throttle[i];
    }
    this.last = best[0][1];
    return { steer: steer / ws, throttle: throttle / ws > 0.5, index: best[0][1], indices: best.map((b) => b[1]), distance: best[0][0] };
  }

  // Pain: make the memories that led here less attractive.
  penalize(indices, amount = 1) {
    for (const i of indices) if (i >= 0 && i < this.penalty.length) this.penalty[i] += amount;
  }

  clear() {
    for (const key of ["raw", "z", "steer", "throttle", "penalty"]) this[key] = [];
    this.last = -1;
  }
}
