// Checks the browser port against outputs of the real flyvis network
// (tests/fixtures/reference.json, written by scripts/export_model.py).
// Run with: node --test tests/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { FlyNetwork, HexEye, StripSampler, decodeBase64 } from "../site/js/sim.js";

const load = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const data = load("../site/data/network.json");
const ref = load("./fixtures/reference.json");
const f32 = (b64) => decodeBase64(b64, Float32Array);

function maxAbsDiff(a, b) {
  assert.equal(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// Same formula as procedural_image() in scripts/export_model.py.
function proceduralImage(size) {
  const img = new Float32Array(size * size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) img[y * size + x] = ((x * 7 + y * 13) % 17) / 16;
  return img;
}

const net = new FlyNetwork(data);

test("filter table expands to the full connectome", () => {
  assert.equal(net.n, 45669);
  assert.equal(net.nEdges, data.n_edges);
});

test("HexEye matches flyvis BoxEye", () => {
  const eye = new HexEye();
  assert.equal(eye.size, 391);
  const got = eye.sampleFrame(proceduralImage(eye.size));
  assert.ok(maxAbsDiff(got, f32(ref.eye_image_expected)) < 1e-5);
});

test("StripSampler matches BoxEye inside a wider strip", () => {
  const eye = new HexEye();
  const img = proceduralImage(eye.size);
  // Embed the frame in a wider strip; sampling at its offset must not see the
  // neighbouring columns (BoxEye zero-pads at the frame border).
  const width = eye.size + 100;
  const strip = new Float32Array(eye.size * width).fill(1);
  for (let y = 0; y < eye.size; y++)
    strip.set(img.subarray(y * eye.size, (y + 1) * eye.size), y * width + 60);
  const got = new StripSampler(eye, strip, width).frame(60);
  assert.ok(maxAbsDiff(got, f32(ref.eye_image_expected)) < 1e-5);
});

test("steady state matches flyvis", () => {
  const rest = net.steadyState(0.5, 1.0);
  assert.ok(maxAbsDiff(rest, f32(ref.rest)) < 1e-4);
});

test("simulated responses match flyvis", () => {
  net.steadyState(0.5, 1.0);
  const stim = f32(ref.stimulus);
  const T = ref.n_frames;
  const central = data.cell_types.map((c) => c.central);
  const expected = f32(ref.central_traces);
  const got = new Float32Array(T * central.length);
  for (let t = 0; t < T; t++) {
    net.step(stim.subarray(t * 721, (t + 1) * 721));
    central.forEach((i, k) => (got[t * central.length + k] = net.state[i]));
  }
  const traceErr = maxAbsDiff(got, expected);
  const stateErr = maxAbsDiff(net.state, f32(ref.final_state));
  console.log(`max |error|: traces ${traceErr.toExponential(2)}, state ${stateErr.toExponential(2)}`);
  assert.ok(traceErr < 1e-3);
  assert.ok(stateErr < 1e-3);
});
