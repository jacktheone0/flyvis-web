// The PolyTrack mod's decoder training, on synthetic "neuron" data with a
// known linear relation to the teacher's steering and throttle.
import assert from "node:assert/strict";
import { test } from "node:test";
import { decode, trainDecoder } from "../site/polytrack/fly-driver.js";
import { rng } from "../site/js/world.js";

const siteRoot = new URL("../site", import.meta.url).href;

test("the decoder learns steering and throttle from neuron features", async () => {
  const random = rng(5);
  const d = 40;
  const truth = Array.from({ length: d }, () => random() * 2 - 1);
  const recording = { features: [], steer: [], throttle: [] };
  for (let i = 0; i < 1500; i++) {
    const f = Float32Array.from({ length: d }, () => random() * 2 - 1);
    const drive = f.reduce((s, x, j) => s + x * truth[j], 0) / 4;
    recording.features.push(f);
    recording.steer.push(Math.max(-1, Math.min(1, drive)));
    recording.throttle.push(f[0] > 0 ? 1 : 0);
  }
  const decoder = await trainDecoder(recording, { siteRoot, lambda: 1 });
  assert.ok(decoder.score.steer > 0.8, `steer R² ${decoder.score.steer}`);
  assert.ok(decoder.score.throttle > 0.4, `throttle R² ${decoder.score.throttle}`);
  // Held-out frames: the decoded steering has the right sign.
  let agree = 0;
  for (let i = 1200; i < 1500; i++) {
    const out = decode(decoder, recording.features[i]);
    if (Math.sign(out.steer) === Math.sign(recording.steer[i])) agree++;
  }
  assert.ok(agree / 300 > 0.9, `sign agreement ${agree / 300}`);
});
