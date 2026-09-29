// Runs the closed-loop flight off the main thread. Messages:
//   init {data, seed, flight}  -> ready {rest, nEdges}
//   world {seed, flight}         start over in another world
//   set {mode?, strength?}       steering rule ("off" | "simple" | "trained"), wind
//   advance {flight, to}       -> progress {flight, t, state, view, headings, spins, winds, modes, traces}
//   train {}                   -> train-progress {fraction}…, trained {r2, weights, detectors, test}
import { FlyNetwork } from "./sim.js";
import { FlightSim } from "./flight-sim.js";
import { DEG_PER_PX } from "./world.js";

const MODES = ["off", "simple", "trained"];
const TEST_GUSTS = 4242; // the same gusts for every test flight
const DEG_PER_S = DEG_PER_PX * 50; // px/frame -> degrees per second

let net;
let rest;
let sim;
let flight = 0;
let seed = 1;
let mode = "off";
let strength = 5;

self.onmessage = ({ data: msg }) => {
  if (msg.type === "init") {
    net = new FlyNetwork(msg.data);
    rest = net.steadyState(0.5, 1.0);
    sim = new FlightSim(net, rest, msg.data.direction_tuning);
    newWorld(msg);
    self.postMessage({ type: "ready", rest: Float32Array.from(rest), nEdges: net.nEdges });
  } else if (msg.type === "world") {
    newWorld(msg);
  } else if (msg.type === "set") {
    if (msg.mode) mode = msg.mode;
    if (msg.strength != null) strength = msg.strength;
  } else if (msg.type === "advance" && msg.flight === flight) {
    advance(msg.to);
  } else if (msg.type === "train") {
    train();
  }
};

function newWorld(msg) {
  seed = msg.seed;
  flight = msg.flight;
  sim.start(seed, seed + 1000);
}

function advance(to) {
  const n = Math.max(0, to - sim.t);
  const nTypes = net.cellTypes.length;
  const central = net.cellTypes.map((c) => c.central);
  const headings = new Float32Array(n);
  const spins = new Float32Array(n);
  const winds = new Float32Array(n);
  const modes = new Uint8Array(n);
  const traces = new Float32Array(n * nTypes);
  for (let r = 0; r < n; r++) {
    const wind = strength * sim.gusts(sim.t);
    const spin = sim.step(mode, strength);
    headings[r] = sim.pos * DEG_PER_PX;
    spins[r] = spin * DEG_PER_S;
    winds[r] = wind * DEG_PER_S;
    modes[r] = MODES.indexOf(mode);
    for (let k = 0; k < nTypes; k++) traces[r * nTypes + k] = net.state[central[k]] - rest[central[k]];
  }
  const state = Float32Array.from(net.state);
  const view = Float32Array.from(sim.lastView);
  self.postMessage(
    { type: "progress", flight, t: sim.t, pos: sim.pos, state, view, headings, spins, winds, modes, traces },
    [state.buffer, view.buffer, headings.buffer, spins.buffer, winds.buffer, modes.buffer, traces.buffer],
  );
}

// Train the decoder, then fly the current world three ways with identical
// gusts. The live flight is paused meanwhile and resumes where it was.
function train() {
  const snapshot = sim.snapshot();
  let reported = 0;
  const progress = (fraction) => {
    if (fraction - reported >= 0.01 || fraction >= 1) {
      reported = fraction;
      self.postMessage({ type: "train-progress", fraction });
    }
  };
  // Fresh worlds each time, never the one being flown.
  const worlds = [];
  while (worlds.length < 6) {
    const s = 1 + Math.floor(Math.random() * 1e6);
    if (s !== seed && !worlds.includes(s)) worlds.push(s);
  }
  const { weights, r2 } = sim.train({
    trainWorlds: worlds.slice(0, 4),
    heldOutWorlds: worlds.slice(4),
    onProgress: (f) => progress((f * 2) / 3),
  });
  const test = {};
  MODES.forEach((m, i) => {
    test[m] = sim.testFlight(m, seed, TEST_GUSTS, strength) * DEG_PER_S;
    progress(2 / 3 + (i + 1) / 9);
  });
  sim.restore(snapshot);
  self.postMessage({
    type: "trained",
    r2,
    weights,
    detectors: sim.detectors.map((d) => ({ name: d.name, preferred: d.preferred })),
    test,
    strength,
  });
}
