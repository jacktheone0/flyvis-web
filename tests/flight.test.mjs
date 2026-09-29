// Behavioural checks of the flight simulator: the world, the fly's motion
// sensing, and closed-loop steering with and without a trained decoder.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { FlyNetwork } from "../site/js/sim.js";
import { WORLD_WIDTH, makeWorld } from "../site/js/world.js";
import { motionBalance } from "../site/js/motion.js";
import { FlightSim } from "../site/js/flight-sim.js";

const data = JSON.parse(readFileSync(new URL("../site/data/network.json", import.meta.url)));
const net = new FlyNetwork(data);
const rest = net.steadyState();
const sim = new FlightSim(net, rest, data.direction_tuning);
const WORLD = 7;
const GUSTS = 8;

test("worlds are deterministic, in range and wrap seamlessly", () => {
  const H = 391;
  const a = makeWorld(3, H);
  assert.deepEqual(a, makeWorld(3, H));
  assert.ok(a.every((x) => x >= 0 && x <= 1));
  // The seam between the last and first column is no rougher than the rest.
  let seam = 0;
  let inner = 0;
  for (let y = 0; y < H; y++) {
    seam += Math.abs(a[y * WORLD_WIDTH] - a[y * WORLD_WIDTH + WORLD_WIDTH - 1]);
    inner += Math.abs(a[y * WORLD_WIDTH + 100] - a[y * WORLD_WIDTH + 99]);
  }
  assert.ok(seam < 3 * inner + 1, `seam ${seam} vs inner ${inner}`);
});

test("motion detectors report which way the world turns", () => {
  const meanBalance = (spin) => {
    sim.start(WORLD, GUSTS);
    let sum = 0;
    for (let t = 0; t < 100; t++) {
      sim.pos += spin;
      net.step(sim.view(sim.pos));
      sim.readDetectors();
      if (t >= 20) sum += motionBalance(sim.detectors, sim.depol).balance;
    }
    return sum / 80;
  };
  // Turning right (positive spin) slides the world left: negative balance.
  assert.ok(meanBalance(4) < -0.1);
  assert.ok(meanBalance(-4) > 0.1);
});

test("the simple reflex steers against the wind", () => {
  const off = sim.testFlight("off", WORLD, GUSTS, 5, 300);
  const simple = sim.testFlight("simple", WORLD, GUSTS, 5, 300);
  assert.ok(simple < 0.75 * off, `simple ${simple} vs off ${off}`);
});

test("a trained decoder estimates spin and steers", () => {
  const { r2, weights } = sim.train({ frames: 250, trainWorlds: [101, 102], heldOutWorlds: [201] });
  assert.equal(weights.length, sim.detectors.length + 1);
  assert.ok(r2 > 0.6, `R² ${r2}`);
  const off = sim.testFlight("off", WORLD, GUSTS, 5, 300);
  const trained = sim.testFlight("trained", WORLD, GUSTS, 5, 300);
  assert.ok(trained < 0.75 * off, `trained ${trained} vs off ${off}`);
});
