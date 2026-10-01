// Runs the fly network off the main thread. Messages:
//   init {data}                 -> ready {rest, nEdges}
//   load {run, frames, nFrames} -> loaded {run}
//   advance {run, to}           -> progress {run, t, state, traces, activity[, stats]}
import { FlyNetwork } from "./sim.js";

let net;
let rest;
let scale;
let run = -1;
let frames;
let nFrames = 0;
let t = 0;
let sumPos;
let sumNorm;

self.onmessage = ({ data: msg }) => {
  if (msg.type === "init") {
    net = new FlyNetwork(msg.data);
    rest = net.steadyState(0.5, 1.0);
    scale = Float64Array.from(net.type, (k) => net.cellTypes[k].scale);
    self.postMessage({ type: "ready", rest: Float32Array.from(rest), nEdges: net.nEdges });
  } else if (msg.type === "load") {
    run = msg.run;
    frames = msg.frames;
    nFrames = msg.nFrames;
    t = 0;
    net.state.set(rest);
    sumPos = new Float64Array(net.cellTypes.length);
    sumNorm = new Float64Array(net.cellTypes.length);
    self.postMessage({ type: "loaded", run });
  } else if (msg.type === "advance" && msg.run === run) {
    const to = Math.min(msg.to, nFrames);
    const nTypes = net.cellTypes.length;
    const traces = new Float32Array((to - t) * nTypes);
    const activity = new Float32Array(to - t); // whole-brain mean |change| per frame
    const central = net.cellTypes.map((c) => c.central);
    for (let row = 0; t < to; t++, row++) {
      net.step(frames.subarray(t * net.nHexals, (t + 1) * net.nHexals));
      const s = net.state;
      let total = 0;
      for (let i = 0; i < net.n; i++) {
        const d = s[i] - rest[i];
        if (d > 0) sumPos[net.type[i]] += d;
        const norm = Math.abs(d) / scale[i];
        sumNorm[net.type[i]] += norm;
        total += norm;
      }
      activity[row] = total / net.n;
      for (let k = 0; k < nTypes; k++) traces[row * nTypes + k] = s[central[k]] - rest[central[k]];
    }
    const state = Float32Array.from(net.state);
    const reply = { type: "progress", run, t, state, traces, activity };
    if (t === nFrames) {
      // Per-cell, per-frame averages for the fly's reply.
      const perCell = (sum) =>
        Array.from(sum, (x, k) => x / (nFrames * net.cellTypes[k].n));
      reply.stats = { meanDepolarization: perCell(sumPos), meanActivity: perCell(sumNorm) };
    }
    self.postMessage(reply, [state.buffer, traces.buffer, activity.buffer]);
  }
};
