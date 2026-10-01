// Closed-loop flight: wind spins the fly, its eye sees the world rotate, its
// T4/T5 motion detectors respond, and a steering rule turns it back. Only the
// yaw rotation is simulated. No DOM access (runs in a worker and in Node).
import { HexEye, StripSampler } from "./sim.js";
import { WORLD_WIDTH, makeGusts, makeWorld, rng } from "./world.js";
import { meanDepolarization, motionBalance, motionDetectors, typeRanges } from "./motion.js";

export const SIMPLE_GAIN = 8; // turn (px/frame) per unit of left/right balance
export const TRAINED_GAIN = 1; // how much of the decoded spin the fly turns against
export const TRAIN_WORLDS = [101, 102, 103, 104];
export const HELD_OUT_WORLDS = [201, 202];
const ADAPT_FRAMES = 50; // 1 s of looking at a still world before flying

export class FlightSim {
  constructor(net, rest, tuning) {
    this.net = net;
    this.rest = rest;
    this.eye = new HexEye(net.extent);
    this.ranges = typeRanges(net.type, net.cellTypes.length);
    this.detectors = motionDetectors(net.cellTypes, tuning);
    this.depol = new Float64Array(this.detectors.length);
    this.weights = null; // trained decoder: [bias, one weight per detector]
  }

  // Sampler for a world; the strip repeats its first columns so it can wrap.
  makeView(seed) {
    const H = this.eye.size;
    const W = WORLD_WIDTH;
    const world = makeWorld(seed, H);
    const strip = new Float32Array((W + H) * H);
    for (let y = 0; y < H; y++) {
      strip.set(world.subarray(y * W, (y + 1) * W), y * (W + H));
      strip.set(world.subarray(y * W, y * W + H), y * (W + H) + W);
    }
    const sampler = new StripSampler(this.eye, strip, W + H);
    const out = new Float32Array(this.eye.n);
    return (pos) => sampler.frame(((Math.round(pos) % W) + W) % W, out);
  }

  // Begin a flight in world `seed` with gust pattern `gustSeed`.
  start(seed, gustSeed) {
    this.view = this.makeView(seed);
    this.gusts = makeGusts(gustSeed);
    this.adapt();
  }

  adapt() {
    this.net.state.set(this.rest);
    this.pos = 0;
    this.t = 0;
    this.turn = 0;
    const still = this.view(0);
    for (let i = 0; i < ADAPT_FRAMES; i++) this.net.step(still);
    this.lastView = still;
    this.readDetectors();
  }

  readDetectors() {
    this.detectors.forEach((d, i) => {
      this.depol[i] = meanDepolarization(this.net.state, this.rest, this.ranges[d.k]);
    });
  }

  // Spin (px/frame, positive = turning right) estimated by the trained decoder.
  decode() {
    let s = this.weights[0];
    for (let i = 0; i < this.depol.length; i++) s += this.weights[i + 1] * this.depol[i];
    return s;
  }

  // Advance 20 ms. Returns the fly's spin this frame in px/frame.
  step(mode, strength) {
    const spin = strength * this.gusts(this.t) + this.turn;
    this.pos += spin;
    this.t++;
    this.lastView = this.view(this.pos);
    this.net.step(this.lastView);
    this.readDetectors();
    if (mode === "simple") {
      this.turn = SIMPLE_GAIN * motionBalance(this.detectors, this.depol).balance;
    } else if (mode === "trained" && this.weights) {
      this.turn = -TRAINED_GAIN * this.decode();
    } else {
      this.turn = 0;
    }
    return spin;
  }

  // Show the fly random spins (no steering) and record what its detectors do.
  record(seeds, frames, onFrame) {
    const X = [];
    const y = [];
    seeds.forEach((seed) => {
      this.view = this.makeView(seed);
      this.adapt();
      const random = rng(seed * 7 + 1);
      let spin = 0;
      for (let t = 0; t < frames; t++) {
        if (t % 25 === 0) spin = (random() * 2 - 1) * 8; // a new spin every 0.5 s
        this.pos += spin;
        this.net.step(this.view(this.pos));
        this.readDetectors();
        if (t % 25 >= 5) {
          X.push([1, ...this.depol]);
          y.push(spin);
        }
        onFrame?.();
      }
    });
    return { X, y };
  }

  // Fit the decoder by ridge regression, then score it on unseen worlds (R²).
  train({ frames = 400, trainWorlds = TRAIN_WORLDS, heldOutWorlds = HELD_OUT_WORLDS, onProgress } = {}) {
    const total = (trainWorlds.length + heldOutWorlds.length) * frames;
    let done = 0;
    const tick = () => onProgress?.(++done / total);
    const train = this.record(trainWorlds, frames, tick);
    this.weights = ridge(train.X, train.y, 1e-4);
    const test = this.record(heldOutWorlds, frames, tick);
    let se = 0;
    let ss = 0;
    const mean = test.y.reduce((a, b) => a + b, 0) / test.y.length;
    test.X.forEach((x, n) => {
      const pred = x.reduce((s, xi, i) => s + xi * this.weights[i], 0);
      se += (pred - test.y[n]) ** 2;
      ss += (test.y[n] - mean) ** 2;
    });
    return { weights: this.weights, r2: 1 - se / ss };
  }

  // RMS spin (px/frame) over a flight with a fixed world and gust pattern.
  testFlight(mode, seed, gustSeed, strength, frames = 400) {
    this.start(seed, gustSeed);
    let sum = 0;
    for (let t = 0; t < frames; t++) sum += this.step(mode, strength) ** 2;
    return Math.sqrt(sum / frames);
  }

  snapshot() {
    return {
      state: Float64Array.from(this.net.state),
      pos: this.pos,
      t: this.t,
      turn: this.turn,
      view: this.view,
      gusts: this.gusts,
    };
  }

  restore(s) {
    this.net.state.set(s.state);
    Object.assign(this, { pos: s.pos, t: s.t, turn: s.turn, view: s.view, gusts: s.gusts });
    this.readDetectors();
  }
}

// Least squares with an L2 penalty on every weight except the bias (column 0).
export function ridge(X, y, lambda) {
  const d = X[0].length;
  const A = Array.from({ length: d }, () => new Float64Array(d + 1));
  X.forEach((x, n) => {
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) A[i][j] += x[i] * x[j];
      A[i][d] += x[i] * y[n];
    }
  });
  for (let i = 1; i < d; i++) A[i][i] += lambda * X.length;
  for (let c = 0; c < d; c++) {
    let p = c;
    for (let r = c + 1; r < d; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < d; r++) {
      if (r === c) continue;
      const m = A[r][c] / A[c][c];
      for (let j = c; j <= d; j++) A[r][j] -= m * A[c][j];
    }
  }
  return A.map((row, i) => row[d] / row[i]);
}
