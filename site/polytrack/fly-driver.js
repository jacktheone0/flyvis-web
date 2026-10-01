// A fruit fly drives PolyTrack. Each rendered frame is fed into the fly's
// 721-facet eye and the flyvis network advances. What its optic lobe sends on
// to the rest of the brain (the output cell types, pooled over a grid of the
// eye, plus a slowly fading trace of the last ~0.5 s: short-term memory) is
// the feature vector a learned policy turns into keys.
//
// Two modes:
//   watch - you drive, the fly watches; every frame is a teaching example.
//   fly   - the fly drives. Press the arrow keys at any time to correct it:
//           it lets go immediately and takes back over ~0.6 s after you
//           release them. Corrections are teaching examples too.
// While the fly drives it wobbles its steering a little (exploration) so
// rewards ("dopamine") and punishments ("pain") can tell it which of its
// recent choices were good.

import { hudCheckpoint, hudSpeed } from "./hud.js";

const KEYS = {
  up: { key: "ArrowUp", code: "ArrowUp", keyCode: 38 },
  left: { key: "ArrowLeft", code: "ArrowLeft", keyCode: 37 },
  right: { key: "ArrowRight", code: "ArrowRight", keyCode: 39 },
  reset: { key: "Enter", code: "Enter", keyCode: 13 }, // back to the last checkpoint
};
const HUMAN_KEYS = {
  ArrowLeft: "left", KeyA: "left",
  ArrowRight: "right", KeyD: "right",
  ArrowUp: "up", KeyW: "up",
  ArrowDown: "down", KeyS: "down",
};

export const DEFAULTS = {
  stepsMax: 4, // most 20 ms brain steps per rendered frame in real time
  lockstep: true, // slow motion: game time advances 20 ms per brain step
  autoReset: true, // return to the last checkpoint when stuck
  autoRewards: true, // checkpoint = dopamine, crash or stuck = pain
  explore: true, // wobble the steering so rewards have something to judge
  takeoverMs: 600, // you keep control this long after releasing the keys
  correctionWeight: 2, // corrections count double in training
};

// The mod patches the physics worker's clock to listen on this channel.
export const CLOCK_CHANNEL = "flyvis-polytrack-clock";

// Pool a group of cell types over a cols x rows grid of the eye. Grid cells
// outside the hexagonal eye hold no neurons and are skipped.
function makePooling(net, types, cols, rows) {
  const binOf = new Int32Array(net.n).fill(-1);
  const counts = new Map();
  types.forEach((k, j) => {
    for (let i = 0; i < net.n; i++) {
      if (net.type[i] !== k) continue;
      const col = Math.min(cols - 1, Math.floor(((net.v[i] + 15) / 31) * cols));
      const r = net.u[i] + net.v[i] / 2; // -15 (top) .. 15 (bottom)
      const row = Math.min(rows - 1, Math.floor(((r + 15) / 31) * rows));
      binOf[i] = (j * rows + row) * cols + col;
      counts.set(binOf[i], (counts.get(binOf[i]) ?? 0) + 1);
    }
  });
  const index = new Map([...counts.keys()].sort((a, b) => a - b).map((b, f) => [b, f]));
  const featureOf = Int32Array.from(binOf, (b) => (b < 0 ? -1 : index.get(b)));
  const size = new Float64Array(index.size);
  for (const [b, f] of index) size[f] = counts.get(b);
  return { n: index.size, featureOf, size };
}

export async function startFlyDriver({ siteRoot, canvas, mode = "fly", policy, options = {}, onUpdate, onSample, onReward } = {}) {
  const opts = { ...DEFAULTS, ...(mode === "watch" ? { lockstep: false } : {}), ...options };
  const { FlyNetwork, HexEye } = await import(`${siteRoot}/js/sim.js`);
  const data = await (await fetch(`${siteRoot}/data/network.json`)).json();
  const net = new FlyNetwork(data);
  const rest = net.steadyState(0.5, 1.0);
  const eye = new HexEye();
  const size = eye.size;

  // Features: output cell types on a 6 x 3 grid now, plus a fading trace of
  // the same types on a 3 x 1 grid (time constant 0.5 s).
  const outputTypes = data.cell_types.flatMap((c, k) => (c.group === "output" ? [k] : []));
  const now = makePooling(net, outputTypes, 6, 3);
  const slow = makePooling(net, outputTypes, 3, 1);
  const d = now.n + slow.n;
  const x = new Float32Array(d);
  const trace = new Float64Array(slow.n);
  const TRACE_DECAY = Math.exp(-0.02 / 0.5);
  const sums = new Float64Array(Math.max(now.n, slow.n));
  const readFeatures = () => {
    sums.fill(0, 0, now.n);
    for (let i = 0; i < net.n; i++) if (now.featureOf[i] >= 0) sums[now.featureOf[i]] += net.state[i] - rest[i];
    for (let f = 0; f < now.n; f++) x[f] = sums[f] / now.size[f];
    sums.fill(0, 0, slow.n);
    for (let i = 0; i < net.n; i++) if (slow.featureOf[i] >= 0) sums[slow.featureOf[i]] += net.state[i] - rest[i];
    for (let f = 0; f < slow.n; f++) {
      trace[f] = TRACE_DECAY * trace[f] + (1 - TRACE_DECAY) * (sums[f] / slow.size[f]);
      x[now.n + f] = trace[f];
    }
    return x;
  };

  // Grey-scale copy of each rendered frame, squeezed into the eye's field.
  // A quarter-size read-back is 16x cheaper; the eye averages 13 px boxes
  // anyway, so it is upsampled back to the eye's 391 px frame.
  const small = Math.ceil(size / 4);
  const grab = document.createElement("canvas");
  grab.width = grab.height = small;
  const gctx = grab.getContext("2d", { willReadFrequently: true });
  const lum = new Float32Array(size * size);
  const lumSmall = new Float32Array(small * small);

  // ---- keys
  const held = new Map(); // synthetic key name -> time of its last keydown
  const send = (name, type) => {
    const k = KEYS[name];
    const init = { key: k.key, code: k.code, keyCode: k.keyCode, which: k.keyCode, bubbles: true, cancelable: true };
    (document.body ?? document).dispatchEvent(new KeyboardEvent(type, init));
  };
  // Held keys auto-repeat like a real keyboard, so a missed press recovers.
  const press = (name, down) => {
    const t = performance.now();
    if (down && (!held.has(name) || t - held.get(name) > 250)) {
      held.set(name, t);
      send(name, "keydown");
    } else if (!down && held.has(name)) {
      held.delete(name);
      send(name, "keyup");
    }
  };
  const tap = (name) => {
    send(name, "keydown");
    send(name, "keyup");
  };

  // Your real (trusted) key presses.
  const human = { left: false, right: false, up: false, down: false, lastEvent: -Infinity };
  const onKey = (e) => {
    const k = HUMAN_KEYS[e.code];
    if (!e.isTrusted || !k) return;
    human[k] = e.type === "keydown";
    human.lastEvent = performance.now();
    if (mode === "fly" && e.type === "keydown" && !takeover) beginTakeover();
  };
  window.addEventListener("keydown", onKey, true);
  window.addEventListener("keyup", onKey, true);
  const humanPressing = () => human.left || human.right || human.up || human.down;

  // Taking over: stop pressing the fly's keys, but don't send a key-up for a
  // key you are holding yourself (that would cancel your press).
  let takeover = false;
  function beginTakeover() {
    takeover = true;
    for (const name of [...held.keys()]) {
      if (human[name]) held.delete(name);
      else press(name, false);
    }
    recent.length = 0; // rewards only judge the fly's own driving
  }

  const clock = opts.lockstep ? new BroadcastChannel(CLOCK_CHANNEL) : null;
  clock?.postMessage({ type: "lock", on: true });

  // ---- rewards
  const recent = []; // the fly's own recent frames: { x, mean, eps, throttle, indices }
  const state = {
    frames: 0, steer: 0, throttle: false, takeover: false, resets: 0,
    corrections: 0, good: 0, bad: 0, lastReward: null, d,
  };
  function reward(r, why) {
    if (mode !== "fly" || !recent.length) return false;
    const frames = recent.splice(0, recent.length);
    r > 0 ? state.good++ : state.bad++;
    state.lastReward = { r, why, at: state.frames };
    onReward?.(r, why, frames);
    return true;
  }
  let lastCheckpoint = null;
  const speeds = [];
  let refractory = 0;
  let stuck = 0;

  let running = true;
  let lastTime = performance.now();
  let steerPhase = 0;
  let eps = 0;

  function frame(t) {
    if (!running) return;
    requestAnimationFrame(frame); // queued after the game's own callback

    gctx.drawImage(canvas, 0, 0, small, small);
    const px = gctx.getImageData(0, 0, small, small).data;
    for (let i = 0; i < lumSmall.length; i++) {
      lumSmall[i] = (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2]) / 255;
    }
    for (let y = 0; y < size; y++) {
      const row = (y >> 2) * small;
      for (let xx = 0; xx < size; xx++) lum[y * size + xx] = lumSmall[row + (xx >> 2)];
    }
    const input = eye.sampleFrame(lum);

    // Lockstep: one 20 ms brain step, then 20 ms of game. Real time: as many
    // steps as the time that passed, showing the latest frame.
    const steps = opts.lockstep ? 1 : Math.max(1, Math.min(opts.stepsMax, Math.round((t - lastTime) / 20)));
    lastTime = t;
    for (let s = 0; s < steps; s++) net.step(input);
    clock?.postMessage({ type: "advance", ms: 20 * steps });
    const feats = readFeatures();

    let steer = 0;
    let throttle = false;
    let source = null;
    if (mode === "watch") {
      steer = (human.right ? 1 : 0) - (human.left ? 1 : 0);
      throttle = human.up;
      source = "lap";
    } else {
      if (takeover && !humanPressing() && performance.now() - human.lastEvent > opts.takeoverMs) {
        takeover = false; // hand control back to the fly
      }
      if (takeover) {
        steer = (human.right ? 1 : 0) - (human.left ? 1 : 0);
        throttle = human.up;
        source = humanPressing() ? "correction" : null;
        if (source) state.corrections++;
      } else {
        const act = policy(feats) ?? { steer: 0, throttle: true };
        // Exploration: a slowly wandering wobble (Ornstein-Uhlenbeck noise).
        const gauss = Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
        eps = opts.explore ? 0.85 * eps + 0.13 * gauss : 0;
        steer = Math.max(-1, Math.min(1, act.steer + eps));
        throttle = act.throttle;
        recent.push({ x: Float32Array.from(feats), mean: act.steer, eps: steer - act.steer, throttle, indices: act.indices });
        if (recent.length > 150) recent.shift();

        // Digital keys, so pulse them in proportion to the steering (PWM).
        steerPhase = (steerPhase + Math.abs(steer)) % 1;
        const pulse = steerPhase < Math.abs(steer);
        press("right", steer > 0.05 && pulse);
        press("left", steer < -0.05 && pulse);
        press("up", throttle);
      }

      // Automatic rewards from the game's display.
      const kmh = hudSpeed();
      const cp = hudCheckpoint();
      if (lastCheckpoint !== null && cp !== null && cp > lastCheckpoint && opts.autoRewards) reward(1, "checkpoint");
      if (cp !== null) lastCheckpoint = cp;
      speeds.push(kmh);
      if (speeds.length > 15) speeds.shift();
      if (refractory > 0) refractory--;
      if (opts.autoRewards && !takeover && refractory === 0 && Math.max(...speeds) - kmh > 35) {
        reward(-1, "crash");
        refractory = 50;
      }
      // Stuck helper (not the fly): stopped 1.5 s while accelerating.
      stuck = !takeover && throttle && state.frames > 60 && kmh < 3 ? stuck + 1 : 0;
      if (opts.autoReset && stuck > 75) {
        stuck = 0;
        if (opts.autoRewards) reward(-1, "stuck");
        tap("reset");
        state.resets++;
      }
    }

    if (source && state.frames >= 25) {
      onSample?.({
        x: Float32Array.from(feats),
        steer,
        throttle: throttle ? 1 : 0,
        weight: source === "correction" ? opts.correctionWeight : 1,
        source,
      });
    }

    Object.assign(state, { steer, throttle, takeover });
    state.frames++;
    onUpdate?.(state, input);
  }
  requestAnimationFrame(frame);

  return {
    state,
    mode,
    reward,
    setPolicy(p) {
      policy = p;
    },
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
