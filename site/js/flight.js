// The flight simulator page: the closed-loop flight runs in a worker; this
// draws the world, the fly's eye, its heading and its brain.
import { HexEye } from "./sim.js";
import { loadModel } from "./model.js";
import { BrainView, FPS, VIEW, drawHexEye, frameRect } from "./brain-view.js";
import { WORLD_WIDTH, makeWorld } from "./world.js";

const WINDS = { calm: 2.5, breezy: 5, stormy: 9 }; // gust strength, px/frame
const MODES = ["off", "simple", "trained"];
const MODE_NAMES = ["Brain off", "Simple reflex", "Trained decoder"];
const MODE_BANDS = ["rgba(141, 155, 171, 0.08)", "rgba(255, 180, 84, 0.16)", "rgba(76, 195, 255, 0.16)"];
const MODE_STATUS = {
  off: "Brain off: the wind is spinning the fly around. Switch its brain on with “Simple reflex”.",
  simple: "Simple reflex: the fly turns whichever way its T4/T5 motion detectors say the world is moving.",
  trained: "Trained decoder: the fly estimates its own spin from 8 learned weights and turns against it.",
};
const PLOT = 20 * FPS; // frames in the heading plot
const SPIN_WINDOW = 5 * FPS;

const $ = (id) => document.getElementById(id);
const mod = (x, m) => ((x % m) + m) % m;
const eye = new HexEye();

let model;
let brain;
let worker;
let flight = 0;
let seed = 1;
let mode = "off";
let wind = "breezy";
let ready = false;
let playing = true;
let pending = false;
let training = false;
let t = 0; // frames simulated in this flight
let clock = 0; // frames that should have been simulated by now
let latest = null; // the latest progress message
let worldCanvas;

// Rolling history for the heading plot and the spin readout.
const history = {
  heading: new Float32Array(PLOT),
  spin: new Float32Array(PLOT),
  wind: new Float32Array(PLOT),
  mode: new Uint8Array(PLOT),
  frames: 0,
};

// ---------------------------------------------------------------- boot

async function boot() {
  model = await loadModel();
  brain = new BrainView(model, {
    groups: $("groups"),
    detail: $("detail"),
    gain: $("gain"),
    emptyTrace: "Waiting for the flight to start",
  });
  buildWorld();
  drawHexEye($("eye-view"), eye, new Float32Array(eye.n).fill(0.5));

  worker = new Worker("js/flight-worker.js", { type: "module" });
  worker.onmessage = onMessage;
  worker.onerror = (e) => status(`The simulation failed to start: ${e.message}`);
  worker.postMessage({ type: "init", data: model.data, seed, flight });
  worker.postMessage({ type: "set", mode, strength: WINDS[wind] });
}

function onMessage({ data: msg }) {
  if (msg.type === "ready") {
    ready = true;
    brain.setRest(msg.rest);
    for (const id of ["world-btn", "pause-btn", "train-btn"]) $(id).disabled = false;
    status(MODE_STATUS[mode]);
  } else if (msg.type === "progress" && msg.flight === flight) {
    record(msg);
    latest = msg;
    t = msg.t;
    pending = false;
    brain.draw(msg.state);
    brain.appendTraces(msg.traces);
    render();
  } else if (msg.type === "train-progress") {
    $("train-bar").style.width = `${msg.fraction * 100}%`;
  } else if (msg.type === "trained") {
    showTraining(msg);
  }
}

function buildWorld() {
  const H = eye.size;
  const lum = makeWorld(seed, H);
  worldCanvas = document.createElement("canvas");
  worldCanvas.width = WORLD_WIDTH;
  worldCanvas.height = H;
  const ctx = worldCanvas.getContext("2d");
  const image = ctx.createImageData(WORLD_WIDTH, H);
  const px = new Uint32Array(image.data.buffer);
  for (let i = 0; i < lum.length; i++) {
    const g = Math.round(Math.min(1, Math.max(0, lum[i])) * 255);
    px[i] = (255 << 24) | (g << 16) | (g << 8) | g;
  }
  ctx.putImageData(image, 0, 0);
}

function record(msg) {
  for (let r = 0; r < msg.headings.length; r++) {
    const slot = history.frames % PLOT;
    history.heading[slot] = msg.headings[r];
    history.spin[slot] = msg.spins[r];
    history.wind[slot] = msg.winds[r];
    history.mode[slot] = msg.modes[r];
    history.frames++;
  }
}

// ---------------------------------------------------------------- controls

function setMode(next) {
  mode = next;
  document.querySelectorAll("[data-mode]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === mode));
  worker?.postMessage({ type: "set", mode });
  if (ready) status(MODE_STATUS[mode]);
}

function setWind(next) {
  wind = next;
  document.querySelectorAll("[data-wind]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.wind === wind));
  worker?.postMessage({ type: "set", strength: WINDS[wind] });
}

function newWorld() {
  flight++;
  seed = 1 + Math.floor(Math.random() * 1e6);
  buildWorld();
  history.frames = 0;
  t = 0;
  clock = 0;
  pending = false;
  latest = null;
  brain.resetTraces();
  worker.postMessage({ type: "world", seed, flight });
}

function setPlaying(on) {
  playing = on;
  $("pause-btn").textContent = on ? "❚❚" : "▶";
  $("pause-btn").setAttribute("aria-label", on ? "Pause" : "Play");
}

function train() {
  training = true;
  $("train-btn").disabled = true;
  $("train-btn").textContent = "Training…";
  $("train-bar").style.width = "0";
  status("Training: the fly is watching four other worlds spin. The flight resumes when it's done.");
  worker.postMessage({ type: "train" });
}

function showTraining(msg) {
  training = false;
  $("train-btn").disabled = false;
  $("train-btn").textContent = "Train again";
  const trainedBtn = document.querySelector('[data-mode="trained"]');
  trainedBtn.disabled = false;
  trainedBtn.removeAttribute("title");

  const pct = Math.round(msg.r2 * 100);
  $("r2-text").textContent = `On two worlds it never saw, the decoder's estimate explains ${pct}% of the fly's spin (R² = ${msg.r2.toFixed(2)}).`;

  const w = msg.weights.slice(1);
  const maxW = Math.max(...w.map(Math.abs));
  $("weights").innerHTML = msg.detectors
    .map((d, i) => {
      const width = (50 * Math.abs(w[i])) / maxW;
      const left = w[i] < 0 ? 50 - width : 50;
      return `<li><span class="name">${d.name} <span class="muted">(${d.preferred})</span></span>
        <span class="track diverging"><span class="fill ${w[i] < 0 ? "neg" : "pos"}" style="left:${left}%;width:${width}%"></span></span>
        <span class="n">${w[i] > 0 ? "+" : ""}${w[i].toFixed(1)}</span></li>`;
    })
    .join("");

  // Turning right makes the world slide left across the eye, so detectors
  // that prefer leftward motion should vote "turning right".
  const byName = Object.fromEntries(msg.detectors.map((d, i) => [d.name, w[i]]));
  const expected = byName.T4a + byName.T5a > 0 && byName.T4b + byName.T5b < 0;
  $("weights-note").hidden = !expected;

  const maxSpin = Math.max(...Object.values(msg.test));
  $("test").innerHTML = MODES.map(
    (m, i) => `<li><span class="name">${MODE_NAMES[i]}</span>
      <span class="track"><span class="fill mode-${m}" style="width:${(100 * msg.test[m]) / maxSpin}%"></span></span>
      <span class="n">${Math.round(msg.test[m])}°/s</span></li>`,
  ).join("");
  $("test-caption").textContent = `The same world and the same ${wind} gusts, 8 seconds each. Average spin, lower is better.`;
  $("train-results").hidden = false;

  setMode("trained");
  status(
    `Trained. In the test flight, spin fell from ${Math.round(msg.test.off)}°/s with the brain off to ` +
      `${Math.round(msg.test.trained)}°/s with the decoder (${Math.round(msg.test.simple)}°/s with the simple reflex).`,
  );
}

// ---------------------------------------------------------------- loop

let lastTs = 0;
function tick(ts) {
  const elapsed = Math.min((ts - lastTs) / 1000, 0.1);
  lastTs = ts;
  if (ready && playing && !pending && !training) {
    clock = Math.min(clock + elapsed * FPS, t + FPS * 0.25);
    const to = Math.floor(clock);
    if (to > t) {
      pending = true;
      worker.postMessage({ type: "advance", flight, to });
    }
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- drawing

function render() {
  if (!latest) return;
  drawWorld(latest.pos);
  drawFront(latest.pos);
  drawHexEye($("eye-view"), eye, latest.view);
  drawCompass();
  drawPlot();
  drawSpin();
}

const lastOf = (arr) => arr[(history.frames - 1) % PLOT];

// The whole world, dimmed except for the part the eye covers.
function drawWorld(pos) {
  const ctx = $("world").getContext("2d");
  const W = WORLD_WIDTH;
  const H = eye.size;
  const x0 = mod(Math.round(pos), W);
  const w1 = Math.min(H, W - x0);
  ctx.drawImage(worldCanvas, 0, 0);
  ctx.fillStyle = "rgba(10, 14, 19, 0.6)";
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(worldCanvas, x0, 0, w1, H, x0, 0, w1, H);
  if (w1 < H) ctx.drawImage(worldCanvas, 0, 0, H - w1, H, 0, 0, H - w1, H);

  ctx.strokeStyle = "#ffb454";
  ctx.lineWidth = 4;
  const edge = (xa, xb, left, right) => {
    ctx.beginPath();
    ctx.moveTo(xa, 2);
    ctx.lineTo(xb, 2);
    ctx.moveTo(xa, H - 2);
    ctx.lineTo(xb, H - 2);
    if (left) {
      ctx.moveTo(xa + 2, 0);
      ctx.lineTo(xa + 2, H);
    }
    if (right) {
      ctx.moveTo(xb - 2, 0);
      ctx.lineTo(xb - 2, H);
    }
    ctx.stroke();
  };
  edge(x0, x0 + w1, true, w1 === H);
  if (w1 < H) edge(0, H - w1, false, true);

  // Where the fly is pointing.
  const xc = mod(x0 + H / 2, W);
  ctx.setLineDash([10, 8]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(xc, 0);
  ctx.lineTo(xc, H);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawFront(pos) {
  const ctx = $("front-view").getContext("2d");
  const W = WORLD_WIDTH;
  const H = eye.size;
  const { origin, size } = frameRect(eye);
  const s = size / H;
  const x0 = mod(Math.round(pos), W);
  const w1 = Math.min(H, W - x0);
  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(0, 0, VIEW, VIEW);
  ctx.drawImage(worldCanvas, x0, 0, w1, H, origin, origin, w1 * s, size);
  if (w1 < H) ctx.drawImage(worldCanvas, 0, 0, H - w1, H, origin + w1 * s, origin, (H - w1) * s, size);
}

// Top-down view: the fly points along its heading; the blue arc is the wind.
function drawCompass() {
  const ctx = $("compass").getContext("2d");
  const c = VIEW / 2;
  const R = 150;
  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(0, 0, VIEW, VIEW);
  ctx.strokeStyle = "#243140";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(c, c, R, 0, 2 * Math.PI);
  ctx.stroke();
  for (let a = 0; a < 360; a += 30) {
    const r = (a - 90) * (Math.PI / 180);
    ctx.lineWidth = a === 0 ? 5 : 2;
    ctx.strokeStyle = a === 0 ? "#8d9bab" : "#243140";
    ctx.beginPath();
    ctx.moveTo(c + Math.cos(r) * (R - 12), c + Math.sin(r) * (R - 12));
    ctx.lineTo(c + Math.cos(r) * (R + 6), c + Math.sin(r) * (R + 6));
    ctx.stroke();
  }
  ctx.fillStyle = "#8d9bab";
  ctx.font = "20px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("start", c, c - R - 16);

  const heading = history.frames ? lastOf(history.heading) : 0;
  const theta = (heading * Math.PI) / 180;

  // Wind: an arc from the fly's heading, clockwise when it pushes right.
  const windNow = history.frames ? lastOf(history.wind) : 0;
  const sweep = Math.max(-Math.PI * 0.9, Math.min(Math.PI * 0.9, (windNow * Math.PI) / 180 / 1.5));
  if (Math.abs(sweep) > 0.02) {
    const a0 = theta - Math.PI / 2;
    const a1 = a0 + sweep;
    const rw = R - 30;
    ctx.strokeStyle = "#4cc3ff";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(c, c, rw, Math.min(a0, a1), Math.max(a0, a1));
    ctx.stroke();
    const dir = Math.sign(sweep);
    const tipX = c + Math.cos(a1) * rw;
    const tipY = c + Math.sin(a1) * rw;
    const tangent = a1 + (dir * Math.PI) / 2;
    ctx.fillStyle = "#4cc3ff";
    ctx.beginPath();
    ctx.moveTo(tipX + Math.cos(tangent) * 14, tipY + Math.sin(tangent) * 14);
    ctx.lineTo(tipX + Math.cos(a1) * 10, tipY + Math.sin(a1) * 10);
    ctx.lineTo(tipX - Math.cos(a1) * 10, tipY - Math.sin(a1) * 10);
    ctx.closePath();
    ctx.fill();
  }

  // The fly, from above, pointing up at heading 0.
  ctx.save();
  ctx.translate(c, c);
  ctx.rotate(theta);
  ctx.fillStyle = "rgba(200, 220, 235, 0.35)";
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.ellipse(side * 34, 10, 34, 15, side * 0.35, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.fillStyle = "#ffb454";
  ctx.beginPath();
  ctx.ellipse(0, 16, 15, 38, 0, 0, 2 * Math.PI);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(0, -30, 14, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#b3261e";
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.arc(side * 9, -34, 7, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();

  const wrapped = mod(heading + 180, 360) - 180;
  ctx.fillStyle = "#e8eef4";
  ctx.font = "600 26px ui-monospace, Menlo, monospace";
  ctx.fillText(`${wrapped >= 0 ? "+" : ""}${Math.round(wrapped)}°`, c, c + 100);
}

// Heading over the last 20 s, newest at the right; bands show the brain mode.
function drawPlot() {
  const canvas = $("heading-plot");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const left = 70;
  const top = 12;
  const bottom = H - 34;
  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(0, 0, W, H);
  const n = Math.min(history.frames, PLOT);
  if (n < 2) return;
  const first = history.frames - n;
  const at = (f) => history.heading[f % PLOT];
  const x = (f) => left + ((f - (history.frames - PLOT)) / PLOT) * (W - left - 10);

  let lo = Infinity;
  let hi = -Infinity;
  for (let f = first; f < history.frames; f++) {
    lo = Math.min(lo, at(f));
    hi = Math.max(hi, at(f));
  }
  const mid = (lo + hi) / 2;
  const span = Math.max(hi - lo, 90) * 1.2;
  const y = (deg) => top + (1 - (deg - (mid - span / 2)) / span) * (bottom - top);

  for (let f = first; f < history.frames; ) {
    const m = history.mode[f % PLOT];
    let g = f;
    while (g < history.frames && history.mode[g % PLOT] === m) g++;
    ctx.fillStyle = MODE_BANDS[m];
    ctx.fillRect(x(f), top, x(g) - x(f), bottom - top);
    f = g;
  }

  ctx.fillStyle = "#8d9bab";
  ctx.font = "20px ui-monospace, Menlo, monospace";
  ctx.textAlign = "right";
  const step = [15, 30, 45, 90, 180, 360, 720].find((s) => span / s <= 4) ?? 1440;
  for (let deg = Math.ceil((mid - span / 2) / step) * step; deg <= mid + span / 2; deg += step) {
    const yy = y(deg);
    ctx.fillText(`${deg}°`, left - 10, yy + 7);
    ctx.strokeStyle = "#1b2530";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, yy);
    ctx.lineTo(W - 10, yy);
    ctx.stroke();
  }
  ctx.textAlign = "left";
  ctx.fillText("−20 s", left, H - 8);
  ctx.textAlign = "right";
  ctx.fillText("now", W - 10, H - 8);

  ctx.strokeStyle = "#e8eef4";
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let f = first; f < history.frames; f++) {
    f === first ? ctx.moveTo(x(f), y(at(f))) : ctx.lineTo(x(f), y(at(f)));
  }
  ctx.stroke();
}

function drawSpin() {
  const n = Math.min(history.frames, SPIN_WINDOW);
  let sum = 0;
  for (let f = history.frames - n; f < history.frames; f++) sum += history.spin[f % PLOT] ** 2;
  $("spin").textContent = `${Math.round(Math.sqrt(sum / n))}°/s`;
}

// ---------------------------------------------------------------- wiring

function status(text) {
  $("status").textContent = text;
}

document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
document.querySelectorAll("[data-wind]").forEach((b) => b.addEventListener("click", () => setWind(b.dataset.wind)));
$("world-btn").addEventListener("click", newWorld);
$("pause-btn").addEventListener("click", () => setPlaying(!playing));
$("train-btn").addEventListener("click", train);

requestAnimationFrame(tick);
boot().catch((e) => status(`Couldn't load the fly: ${e.message}`));
