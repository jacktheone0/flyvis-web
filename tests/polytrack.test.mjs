// The PolyTrack mod's learning rules, on synthetic "neuron" data with a known
// relation to steering: imitation (ridge decoder), route memory, and rewards.
import assert from "node:assert/strict";
import { test } from "node:test";
import { RouteMemory, decode, fitDecoder, nudgeDecoder, rewardExamples } from "../site/polytrack/learner-core.js";
import { rng } from "../site/js/world.js";

const random = rng(5);
const d = 40;
const truth = Array.from({ length: d }, () => random() * 2 - 1);
const n = 2000;
const X = new Float32Array(n * d);
const steer = new Float32Array(n);
const throttle = new Float32Array(n);
for (let i = 0; i < n; i++) {
  let s = 0;
  for (let f = 0; f < d; f++) {
    X[i * d + f] = random() * 2 - 1;
    s += X[i * d + f] * truth[f];
  }
  steer[i] = Math.max(-1, Math.min(1, s / 4));
  throttle[i] = X[i * d] > 0 ? 1 : 0;
}
const data = { X, n, d, steer, throttle, weight: new Float32Array(n).fill(1) };
const row = (i) => X.subarray(i * d, (i + 1) * d);

test("the decoder learns steering and throttle from neuron features", () => {
  const decoder = fitDecoder(data);
  assert.ok(decoder.score.steer > 0.8, `steer R² ${decoder.score.steer}`);
  assert.ok(decoder.score.throttle > 0.3, `throttle R² ${decoder.score.throttle}`);
  let agree = 0;
  for (let i = 0; i < 200; i++) if (Math.sign(decode(decoder, row(i)).steer) === Math.sign(steer[i])) agree++;
  assert.ok(agree / 200 > 0.9, `sign agreement ${agree / 200}`);
});

test("route memory recalls what was done at the best-matching view", () => {
  const decoder = fitDecoder(data);
  const memory = new RouteMemory();
  for (let i = 0; i < 500; i++) memory.add(row(i), steer[i], throttle[i] === 1);
  memory.setNorm(decoder);
  // A slightly noisy view of stored frame 123 recalls its steering.
  const q = Float32Array.from(row(123), (v) => v + 0.05 * (random() - 0.5));
  const act = memory.act(q);
  assert.equal(act.index, 123);
  assert.ok(Math.abs(act.steer - steer[123]) < 0.3);
  // Pain on that memory makes it less attractive.
  memory.penalize([123], 1000);
  assert.notEqual(memory.act(q).index, 123);
});

test("dopamine reinforces recent wobbles and pain reverses them", () => {
  const decoder = fitDecoder(data);
  const x = row(7);
  const before = decode(decoder, x).steer;
  const recent = Array.from({ length: 10 }, () => ({ x, mean: before, eps: 0.3, throttle: true }));
  nudgeDecoder(decoder, recent, 1);
  const rewarded = decode(decoder, x).steer;
  assert.ok(rewarded > before, `reward should push steering toward the wobble (${before} -> ${rewarded})`);
  nudgeDecoder(decoder, recent, -2);
  assert.ok(decode(decoder, x).steer < rewarded);
  const good = rewardExamples(recent, 1);
  const bad = rewardExamples(recent, -1);
  assert.ok(good[0].steer > before && bad[0].steer < before);
  assert.ok(good.at(-1).weight > good[0].weight, "recent moments count most");
});
