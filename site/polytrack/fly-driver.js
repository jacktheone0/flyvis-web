// A fruit fly drives PolyTrack. Each rendered frame is fed into the fly's
// 721-facet eye, the flyvis network advances, and two insect reflexes read
// its T4/T5 motion detectors:
//   - centering: steer away from the side where the world streams past
//     faster (bees fly down corridors this way);
//   - speed: accelerate while the overall streaming is below a target.
// The keys are pressed with synthetic keyboard events, like a player would.
// No game internals are touched.

import { hudSpeed } from "./hud.js";

const KEYS = {
  up: { key: "ArrowUp", code: "ArrowUp", keyCode: 38 },
  reset: { key: "Enter", code: "Enter", keyCode: 13 }, // back to the last checkpoint
  left: { key: "ArrowLeft", code: "ArrowLeft", keyCode: 37 },
  right: { key: "ArrowRight", code: "ArrowRight", keyCode: 39 },
};

export const DEFAULTS = {
  steerGain: 3, // steering per unit of left/right imbalance
  flowTarget: 0.45, // overall outward motion the fly tries to keep (speed)
  stepsMax: 4, // most 20 ms brain steps run per rendered frame (real time)
  lockstep: true, // slow motion: game time advances 20 ms per brain step
  autoReset: true, // return to the last checkpoint when stuck
  gridCols: 6, // output neurons are pooled over this grid of the eye
  gridRows: 3,
  featureGroups: ["output"], // which cell types the decoder may read
};

// The mod patches the physics worker's clock to listen on this channel.
export const CLOCK_CHANNEL = "flyvis-polytrack-clock";

export async function startFlyDriver({ siteRoot, canvas, options = {}, onUpdate } = {}) {
  const opts = { ...DEFAULTS, ...options };
  const { FlyNetwork, HexEye } = await import(`${siteRoot}/js/sim.js`);
  const data = await (await fetch(`${siteRoot}/data/network.json`)).json();
  const net = new FlyNetwork(data);
  const rest = net.steadyState(0.5, 1.0);
  const eye = new HexEye();
  const size = eye.size;

  // Horizontal-motion detectors (T4/T5 subtypes tuned to leftward or
  // rightward motion), pooled over four regions of the eye: upper/lower x
  // left/right. The upper half sees sky and distant hills, which only move
  // when the car turns; the lower half sees the road and walls, which also
  // stream past when it drives forward.
  const REGIONS = ["upperLeft", "upperRight", "lowerLeft", "lowerRight"];
  const regionOf = (u, v) => {
    const row = u + v / 2; // eye rows, negative = up
    if (Math.abs(v) <= 2 || Math.abs(row) <= 1) return null; // skip the midlines
    return (row < 0 ? "upper" : "lower") + (v < 0 ? "Left" : "Right");
  };
  const pools = []; // {region, dir, norm, cells}
  for (const [name, tu] of Object.entries(data.direction_tuning)) {
    if (tu.preferred !== "left" && tu.preferred !== "right") continue;
    const k = data.cell_types.findIndex((c) => c.name === name);
    const pol = name.startsWith("T4") ? "on" : "off";
    const byRegion = Object.fromEntries(REGIONS.map((r) => [r, []]));
    for (let i = 0; i < net.n; i++) {
      if (net.type[i] !== k) continue;
      const r = regionOf(net.u[i], net.v[i]);
      if (r) byRegion[r].push(i);
    }
    for (const r of REGIONS) {
      pools.push({ region: r, dir: tu.preferred, norm: tu[`${tu.preferred}_${pol}`], cells: Int32Array.from(byRegion[r]) });
    }
  }
  // Mean normalized depolarization of leftward- and rightward-tuned cells in
  // each region, e.g. flow.lowerLeft.left.
  const readFlow = () => {
    const flow = Object.fromEntries(REGIONS.map((r) => [r, { left: 0, right: 0 }]));
    const count = Object.fromEntries(REGIONS.map((r) => [r, { left: 0, right: 0 }]));
    for (const p of pools) {
      let s = 0;
      for (const i of p.cells) {
        const d = net.state[i] - rest[i];
        if (d > 0) s += d;
      }
      flow[p.region][p.dir] += s / p.cells.length / p.norm;
      count[p.region][p.dir]++;
    }
    for (const r of REGIONS) for (const d of ["left", "right"]) flow[r][d] /= count[r][d];
    return flow;
  };
  // What the optic lobe sends on to the rest of the brain: each output cell
  // type's mean change from rest, pooled over a coarse 6 x 3 grid of the eye.
  const COLS = opts.gridCols;
  const ROWS = opts.gridRows;
  const outputTypes = data.cell_types.flatMap((c, k) =>
    opts.featureGroups.includes(c.group) ? [k] : [],
  );
  // Grid cells outside the hexagonal eye hold no neurons and are skipped.
  const featureOf = new Int32Array(net.n).fill(-1);
  const binOf = new Int32Array(net.n).fill(-1);
  const cellCount = new Float64Array(outputTypes.length * COLS * ROWS);
  outputTypes.forEach((k, j) => {
    for (let i = 0; i < net.n; i++) {
      if (net.type[i] !== k) continue;
      const col = Math.min(COLS - 1, Math.floor(((net.v[i] + 15) / 31) * COLS));
      const r = net.u[i] + net.v[i] / 2; // -15 (top) .. 15 (bottom)
      const row = Math.min(ROWS - 1, Math.floor(((r + 15) / 31) * ROWS));
      binOf[i] = (j * ROWS + row) * COLS + col;
      cellCount[binOf[i]]++;
    }
  });
  const featureIndex = new Int32Array(cellCount.length).fill(-1);
  let nFeatures = 0;
  cellCount.forEach((c, b) => c > 0 && (featureIndex[b] = nFeatures++));
  const binCount = new Float64Array(nFeatures);
  for (let i = 0; i < net.n; i++) {
    if (binOf[i] < 0) continue;
    featureOf[i] = featureIndex[binOf[i]];
    binCount[featureOf[i]]++;
  }
  const features = new Float64Array(nFeatures);
  const readFeatures = () => {
    features.fill(0);
    for (let i = 0; i < net.n; i++) {
      const f = featureOf[i];
      if (f >= 0) features[f] += net.state[i] - rest[i];
    }
    for (let f = 0; f < nFeatures; f++) features[f] /= binCount[f];
    return features;
  };

  // Grey-scale copy of each rendered frame, squeezed into the eye's field.
  // Reading back a quarter-size image is 16x cheaper; the eye averages 13 px
  // boxes anyway, so it is upsampled back to the eye's 391 px frame.
  const small = Math.ceil(size / 4);
  const grab = document.createElement("canvas");
  grab.width = grab.height = small;
  const gctx = grab.getContext("2d", { willReadFrequently: true });
  const lum = new Float32Array(size * size);
  const lumSmall = new Float32Array(small * small);

  const held = new Map(); // key name -> time of the last keydown sent
  const send = (name, type) => {
    const k = KEYS[name];
    const init = { key: k.key, code: k.code, keyCode: k.keyCode, which: k.keyCode, bubbles: true, cancelable: true };
    (document.body ?? document).dispatchEvent(new KeyboardEvent(type, init));
  };
  // Like a real keyboard, held keys auto-repeat, so a press the game missed
  // (for example while it was still loading) is not lost.
  const press = (name, down) => {
    const now = performance.now();
    if (down && (!held.has(name) || now - held.get(name) > 250)) {
      held.set(name, now);
      send(name, "keydown");
    } else if (!down && held.has(name)) {
      held.delete(name);
      send(name, "keyup");
    }
  };

  const clock = opts.lockstep ? new BroadcastChannel(CLOCK_CHANNEL) : null;
  clock?.postMessage({ type: "lock", on: true });

  let running = true;
  let last = performance.now();
  let steerPhase = 0;
  const smooth = { left: 0, right: 0 };
  const state = { frames: 0, steps: 0, left: 0, right: 0, balance: 0, steer: 0, throttle: false, resets: 0 };

  function frame(now) {
    if (!running) return;
    requestAnimationFrame(frame); // queued after the game's own callback

    const t0 = performance.now();
    gctx.drawImage(canvas, 0, 0, small, small);
    const px = gctx.getImageData(0, 0, small, small).data;
    for (let i = 0; i < lumSmall.length; i++) {
      lumSmall[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
    }
    for (let y = 0; y < size; y++) {
      const row = ((y >> 2) * small);
      for (let x = 0; x < size; x++) lum[y * size + x] = lumSmall[row + (x >> 2)];
    }
    const input = eye.sampleFrame(lum);
    const t1 = performance.now();

    // Keep the brain in step with real time: one 20 ms step per 20 ms that
    // passed, showing the latest frame (like an eye watching a screen).
    // In lockstep the game waits for the brain: one step, then 20 ms of game.
    const steps = opts.lockstep ? 1 : Math.max(1, Math.min(opts.stepsMax, Math.round((now - last) / 20)));
    last = now;
    for (let s = 0; s < steps; s++) net.step(input);
    clock?.postMessage({ type: "advance", ms: 20 * steps });
    const t2 = performance.now();
    state.msCapture = t1 - t0;
    state.msBrain = t2 - t1;

    const regions = readFlow();
    const feats = opts.decoder || opts.record ? readFeatures() : null;
    let steer;
    let throttle;
    if (opts.decoder) {
      // A decoder trained on a teacher's driving reads the output neurons.
      ({ steer, throttle } = decode(opts.decoder, feats));
    } else if (opts.human) {
      // The player drives; the fly only watches. Their keys are the labels.
      steer = (human.right ? 1 : 0) - (human.left ? 1 : 0);
      throttle = human.up;
    } else if (opts.control) {
      // Test hook: a script decides, the fly only watches.
      const out = opts.control(state, regions, { rgb: px, width: small, height: small });
      ({ steer, throttle } = out);
      if (out.reset) {
        send("reset", "keydown");
        send("reset", "keyup");
      }
    } else {
      const l = regions.lowerLeft.left;
      const r = regions.lowerRight.right;
      smooth.left += 0.3 * (l - smooth.left);
      smooth.right += 0.3 * (r - smooth.right);
      const balance = (smooth.left - smooth.right) / (smooth.left + smooth.right + 0.02);
      steer = Math.max(-1, Math.min(1, opts.steerGain * balance));
      throttle = state.frames < 30 || (smooth.left + smooth.right) / 2 < opts.flowTarget;
      Object.assign(state, { left: smooth.left, right: smooth.right, balance });
    }

    if (!opts.human) {
      // Digital keys, so pulse them in proportion to the steering (PWM).
      steerPhase = (steerPhase + Math.abs(steer)) % 1;
      const pulse = steerPhase < Math.abs(steer);
      press("right", steer > 0.05 && pulse);
      press("left", steer < -0.05 && pulse);
      press("up", throttle);

      // Stuck helper (not the fly): stopped for 1.5 s while accelerating
      // means a crash, so return to the last checkpoint.
      stuck = throttle && state.frames > 60 && hudSpeed() < 3 ? stuck + 1 : 0;
      if (opts.autoReset && stuck > 75) {
        stuck = 0;
        send("reset", "keydown");
        send("reset", "keyup");
        state.resets++;
      }
    }

    if (opts.record && state.frames >= 25) {
      recording.features.push(Float32Array.from(feats));
      recording.steer.push(steer);
      recording.throttle.push(throttle ? 1 : 0);
    }

    Object.assign(state, { steps: state.steps + steps, steer, throttle, regions });
    state.frames++;
    onUpdate?.(state, input);
  }
  const recording = { features: [], steer: [], throttle: [] };
  let stuck = 0;

  // In human mode, track the player's real (trusted) key presses.
  const human = { left: false, right: false, up: false };
  const HUMAN_KEYS = { ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right", ArrowUp: "up", KeyW: "up" };
  const onKey = (e) => {
    if (e.isTrusted && HUMAN_KEYS[e.code]) human[HUMAN_KEYS[e.code]] = e.type === "keydown";
  };
  if (opts.human) {
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("keyup", onKey, true);
  }
  requestAnimationFrame(frame);

  return {
    state,
    options: opts,
    recording,
    stop() {
      running = false;
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("keyup", onKey, true);
      for (const k of ["up", "left", "right"]) press(k, false);
      clock?.postMessage({ type: "lock", on: false });
      clock?.close();
    },
  };
}

// Apply a trained decoder to one feature vector.
export function decode(decoder, feats) {
  const out = {};
  for (const target of ["steer", "throttle"]) {
    const w = decoder[target];
    let y = w[0];
    for (let f = 0; f < feats.length; f++) y += w[f + 1] * ((feats[f] - decoder.mean[f]) / decoder.std[f]);
    out[target] = y;
  }
  return { steer: Math.max(-1, Math.min(1, out.steer)), throttle: out.throttle > 0.5 };
}

// Fit steer and throttle decoders by ridge regression on standardized
// features. The last `holdOut` fraction of frames is kept back to score them.
export async function trainDecoder(recording, { lambda = 1, holdOut = 0.2, siteRoot } = {}) {
  const { ridge } = await import(`${siteRoot}/js/flight-sim.js`);
  const n = recording.features.length;
  const d = recording.features[0].length;
  const split = Math.floor(n * (1 - holdOut));
  const mean = new Float64Array(d);
  const std = new Float64Array(d);
  for (let i = 0; i < split; i++) for (let f = 0; f < d; f++) mean[f] += recording.features[i][f] / split;
  for (let i = 0; i < split; i++) for (let f = 0; f < d; f++) std[f] += (recording.features[i][f] - mean[f]) ** 2 / split;
  for (let f = 0; f < d; f++) std[f] = Math.sqrt(std[f]) || 1;
  const row = (i) => [1, ...Array.from(recording.features[i], (x, f) => (x - mean[f]) / std[f])];
  const X = Array.from({ length: split }, (_, i) => row(i));
  const decoder = { mean, std, nFrames: n };
  const score = {};
  for (const target of ["steer", "throttle"]) {
    const y = recording[target];
    decoder[target] = ridge(X, y.slice(0, split), lambda / split);
    let se = 0;
    let ss = 0;
    const test = y.slice(split);
    const m = test.reduce((a, b) => a + b, 0) / test.length;
    test.forEach((t, j) => {
      const x = row(split + j);
      const p = x.reduce((a, xi, k) => a + xi * decoder[target][k], 0);
      se += (p - t) ** 2;
      ss += (t - m) ** 2;
    });
    score[target] = 1 - se / ss;
  }
  decoder.score = score;
  return decoder;
}
