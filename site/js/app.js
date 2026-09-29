import { HexEye, StripSampler, decodeBase64 } from "./sim.js";

const FPS = 50; // simulation frames per second (dt = 20 ms)
const CELL = 4; // panel pixels per lattice step
const PANEL = 31 * CELL; // panel canvas size
const TRACE_WINDOW = 6 * FPS; // frames shown in the cell trace
const MAX_STRIP = 16000; // widest strip canvas every browser handles

const EXAMPLES = [
  {
    label: "Lunch?",
    subject: "Lunch on Friday?",
    body: "Hey! Are you free for lunch on Friday around noon? There's a new taco place by the harbor I've been meaning to try.",
  },
  {
    label: "Spam",
    subject: "URGENT: you won a free banana!!!",
    body: "Click here to claim your prize before the fruit bowl is empty. Offer valid for flies only.",
  },
  {
    label: "Regatta",
    subject: "Regatta results are posted",
    body: "Great sailing this weekend! Results from Saturday's regatta are up on the website. See you at practice on Tuesday.",
  },
];

const GROUPS = {
  retina: {
    title: "Photoreceptors",
    blurb: "Light hits these first. Every frame of the email is fed into R1–R8.",
  },
  intermediate: {
    title: "Intermediate neurons",
    blurb: "Lamina and medulla cells that filter the photoreceptor signal and pass it on.",
  },
  output: {
    title: "Output neurons",
    blurb: "Cells that project deeper into the brain. The model was trained so that visual motion can be read out from them.",
  },
};

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en-US");

// ---------------------------------------------------------------- colors

const REST_RGB = [34, 44, 56];
const STOPS_POS = [[0, REST_RGB], [0.6, [255, 157, 61]], [1, [255, 241, 194]]];
const STOPS_NEG = [[0, REST_RGB], [0.6, [47, 168, 255]], [1, [212, 240, 255]]];

function ramp(stops, t) {
  for (let s = 1; s < stops.length; s++) {
    const [t1, c1] = stops[s];
    const [t0, c0] = stops[s - 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return c0.map((c, i) => Math.round(c + (c1[i] - c) * f));
    }
  }
  return stops[stops.length - 1][1];
}

// 511 entries for values -1..1, packed for a little-endian Uint32 ImageData view.
const LUT = new Uint32Array(511);
for (let i = 0; i < 511; i++) {
  const t = (i - 255) / 255;
  const [r, g, b] = t >= 0 ? ramp(STOPS_POS, t) : ramp(STOPS_NEG, -t);
  LUT[i] = (255 << 24) | (b << 16) | (g << 8) | r;
}
const lutIndex = (x) => (x >= 1 ? 510 : x <= -1 ? 0 : Math.round((x + 1) * 255));

// ---------------------------------------------------------------- state

const eye = new HexEye();
let model; // network.json plus decoded arrays
let worker;
let rest;
let panels = [];
let nodePixel;
let selected;
let gain = 1;
let job = null; // the email currently being shown
let runCounter = 0;
let playing = true;
let rate = 1;

// ---------------------------------------------------------------- boot

async function boot() {
  const data = await (await fetch("data/network.json")).json();
  const type = decodeBase64(data.nodes.type, Uint8Array);
  model = {
    data,
    types: data.cell_types,
    typeIndex: Object.fromEntries(data.cell_types.map((c, k) => [c.name, k])),
    type,
    u: decodeBase64(data.nodes.u, Int8Array),
    v: decodeBase64(data.nodes.v, Int8Array),
    scale: Float32Array.from(type, (k) => data.cell_types[k].scale),
    typeRange: data.cell_types.map(() => [type.length, 0]),
  };
  type.forEach((k, i) => {
    const r = model.typeRange[k];
    r[0] = Math.min(r[0], i);
    r[1] = Math.max(r[1], i + 1);
  });
  $("model-name").textContent = data.meta.model;
  $("fact-neurons").textContent = fmt.format(type.length);

  buildExamples();
  buildPanels();
  selectType(model.typeIndex.T4a);
  setGain();
  drawEye(new Float32Array(eye.n).fill(0.5));
  drawScreen();

  worker = new Worker("js/worker.js", { type: "module" });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => status(`The simulation failed to start: ${e.message}`);
  worker.postMessage({ type: "init", data });
}

function onWorkerMessage({ data: msg }) {
  if (msg.type === "ready") {
    rest = msg.rest;
    $("fact-synapses").textContent = fmt.format(msg.nEdges);
    drawPanels(rest);
    $("show-btn").disabled = false;
    $("show-btn").textContent = "Show it to the fly";
    status("Ready. Pick an example or write your own email, then press the button.");
  } else if (msg.type === "progress" && job && msg.run === job.run) {
    job.traces.set(msg.traces, job.t * model.types.length);
    job.activity.set(msg.activity, job.t);
    job.t = msg.t;
    job.state = msg.state;
    job.pending = false;
    render();
    if (msg.stats && !job.done) finish(msg.stats);
  }
}

// ---------------------------------------------------------------- email → stimulus

function buildExamples() {
  const box = document.querySelector(".examples");
  EXAMPLES.forEach((ex, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = ex.label;
    b.addEventListener("click", () => {
      $("email-subject").value = ex.subject;
      $("email-body").value = ex.body;
    });
    box.append(b);
    if (i === 0) b.click();
  });
}

function renderStrip(text, fontPx) {
  const H = eye.size;
  const canvas = document.createElement("canvas");
  let ctx = canvas.getContext("2d");
  const font = `700 ${fontPx}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  ctx.font = font;
  const fits = (s) => ctx.measureText(s).width <= MAX_STRIP - 2 * H;
  let shown = text;
  if (!fits(shown)) {
    let lo = 0;
    let hi = text.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (fits(text.slice(0, mid) + "…")) lo = mid;
      else hi = mid - 1;
    }
    shown = text.slice(0, lo).trimEnd() + "…";
  }
  canvas.width = Math.ceil(ctx.measureText(shown).width) + 2 * H;
  canvas.height = H;
  ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, H);
  ctx.fillStyle = "#000";
  ctx.font = font;
  ctx.textBaseline = "middle";
  ctx.fillText(shown, H, H / 2);

  // Paper is mid-grey (0.5, what the fly adapted to) and ink is black (0).
  const px = ctx.getImageData(0, 0, canvas.width, H).data;
  const lum = new Float32Array(canvas.width * H);
  for (let i = 0; i < lum.length; i++) {
    lum[i] = (0.5 * (0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2])) / 255;
  }
  const chars = shown === text ? text.length : shown.length - 1;
  return { canvas, lum, width: canvas.width, text: shown, chars, font };
}

function startRun(event) {
  event.preventDefault();
  if (!rest) return;
  const subject = $("email-subject").value.trim();
  const body = $("email-body").value.trim();
  const from = $("email-from").value.trim();
  const text = [subject, body].filter(Boolean).join(" — ").replace(/\s+/g, " ");
  if (!text) return;

  status("Drawing the email onto the fly's visual field…");
  const strip = renderStrip(text, Number($("opt-size").value));
  const speed = Number($("opt-speed").value);
  const sampler = new StripSampler(eye, strip.lum, strip.width);
  const nFrames = Math.floor((strip.width - eye.size) / speed) + 1;
  const frames = new Float32Array(nFrames * eye.n);
  for (let f = 0; f < nFrames; f++) sampler.frame(f * speed, frames.subarray(f * eye.n, (f + 1) * eye.n));

  job = {
    run: ++runCounter,
    subject,
    from,
    strip,
    sampler,
    speed,
    nFrames,
    t: 0,
    clock: 0,
    state: rest,
    pending: false,
    traces: new Float32Array(nFrames * model.types.length),
    activity: new Float32Array(nFrames),
    done: false,
  };
  worker.postMessage({ type: "load", run: job.run, frames, nFrames }, [frames.buffer]);
  $("reply").hidden = true;
  $("play-btn").disabled = false;
  $("skip-btn").disabled = false;
  setPlaying(true);
  meter = 0;
  if (window.matchMedia("(max-width: 960px)").matches) {
    document.querySelector(".vision-card").scrollIntoView({ behavior: "smooth" });
  }
  const shortened = strip.chars < text.length ? ` It will see the first ${strip.chars} characters.` : "";
  status(`The fly is watching your email scroll by.${shortened}`);
  render();
}

// ---------------------------------------------------------------- playback

let lastTs = 0;
function tick(ts) {
  const elapsed = Math.min((ts - lastTs) / 1000, 0.1);
  lastTs = ts;
  if (job && !job.done && !job.pending && (playing || job.skip)) {
    job.clock = Math.min(job.clock + elapsed * FPS * rate, job.t + FPS * rate * 0.25, job.nFrames);
    const to = job.skip ? job.nFrames : Math.floor(job.clock);
    if (to > job.t) {
      job.pending = true;
      worker.postMessage({ type: "advance", run: job.run, to });
    }
  }
  requestAnimationFrame(tick);
}

function setPlaying(on) {
  playing = on;
  $("play-btn").textContent = on ? "❚❚" : "▶";
  $("play-btn").setAttribute("aria-label", on ? "Pause" : "Play");
}

function skipToEnd() {
  if (!job || job.done) return;
  status("Running the rest of the simulation…");
  job.skip = true;
}

// ---------------------------------------------------------------- drawing

function render() {
  if (!job) return;
  const shown = Math.max(job.t - 1, 0);
  drawScreen(job.strip.canvas, shown * job.speed);
  drawEye(job.sampler.frame(shown * job.speed));
  drawPanels(job.state);
  drawTrace();
  drawMeter();
  $("progress-bar").style.width = `${(100 * job.t) / job.nFrames}%`;
  $("clock").textContent = `${(job.t / FPS).toFixed(1)} s`;
}

// Hexal centers on the 380 px views: 12 px per lattice step. eye.y is
// truncated like flyvis, so recompute the exact row for drawing.
const VIEW = 380;
const STEP = 12;
const hexRows = (() => {
  const rows = new Float32Array(eye.n);
  let h = 0;
  for (let u = -15; u <= 15; u++)
    for (let v = Math.max(-15, -15 - u); v <= Math.min(15, 15 - u); v++) rows[h++] = u + v / 2;
  return rows;
})();

function drawEye(values) {
  const ctx = $("eye-view").getContext("2d");
  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(0, 0, VIEW, VIEW);
  const a = (2 * STEP) / 3;
  const b = STEP / 2;
  for (let h = 0; h < eye.n; h++) {
    const cx = 10 + (eye.x[h] / eye.kernel + 15) * STEP;
    const cy = 10 + (hexRows[h] + 15) * STEP;
    const g = Math.round(Math.max(0, Math.min(1, values[h])) * 255);
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    ctx.beginPath();
    ctx.moveTo(cx - a, cy);
    ctx.lineTo(cx - a / 2, cy - b);
    ctx.lineTo(cx + a / 2, cy - b);
    ctx.lineTo(cx + a, cy);
    ctx.lineTo(cx + a / 2, cy + b);
    ctx.lineTo(cx - a / 2, cy + b);
    ctx.closePath();
    ctx.fill();
  }
}

function drawScreen(strip, offset = 0) {
  const ctx = $("screen-view").getContext("2d");
  const s = STEP / eye.kernel; // same scale as the eye view
  const origin = VIEW / 2 - (eye.size / 2) * s;
  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(0, 0, VIEW, VIEW);
  ctx.fillStyle = "#000";
  ctx.fillRect(origin, origin, eye.size * s, eye.size * s);
  ctx.globalAlpha = 0.5; // what the fly gets: paper is 50% grey
  if (strip) {
    ctx.drawImage(strip, offset, 0, eye.size, eye.size, origin, origin, eye.size * s, eye.size * s);
  } else {
    ctx.fillStyle = "#fff";
    ctx.fillRect(origin, origin, eye.size * s, eye.size * s);
  }
  ctx.globalAlpha = 1;
}

function buildPanels() {
  const { types, type, u, v } = model;
  const container = $("groups");
  const byGroup = {};
  for (const key of Object.keys(GROUPS)) {
    const section = document.createElement("div");
    section.className = "group";
    section.innerHTML = `<h3>${GROUPS[key].title}</h3><p>${GROUPS[key].blurb}</p><div class="panels"></div>`;
    container.append(section);
    byGroup[key] = section.querySelector(".panels");
  }
  panels = types.map((ct, k) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "panel";
    button.setAttribute("aria-label", `${ct.name}, ${ct.n} cells`);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = PANEL;
    const label = document.createElement("span");
    label.textContent = ct.name;
    button.append(canvas, label);
    button.addEventListener("click", () => selectType(k));
    byGroup[ct.group].append(button);
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(PANEL, PANEL);
    return { button, ctx, image, pixels: new Uint32Array(image.data.buffer) };
  });
  // Top-left pixel of each neuron's square, with columns offset by half a step.
  nodePixel = new Int32Array(type.length);
  for (let i = 0; i < type.length; i++) {
    const x = (v[i] + 15) * CELL;
    const y = (u[i] + 15) * CELL + v[i] * (CELL / 2);
    nodePixel[i] = y * PANEL + x;
  }
}

function drawPanels(state) {
  const { type, scale } = model;
  for (let i = 0; i < type.length; i++) {
    const color = LUT[lutIndex(((state[i] - rest[i]) / scale[i]) * gain)];
    const px = panels[type[i]].pixels;
    const p = nodePixel[i];
    for (let dy = 0; dy < CELL - 1; dy++) {
      const row = p + dy * PANEL;
      for (let dx = 0; dx < CELL - 1; dx++) px[row + dx] = color;
    }
  }
  for (const panel of panels) panel.ctx.putImageData(panel.image, 0, 0);
}

function setGain() {
  gain = 2 ** Number($("gain").value);
  if (rest) drawPanels(job ? job.state : rest);
}

// ---------------------------------------------------------------- detail

const FAMILIES = [
  [/^R[1-6]$/, "Outer photoreceptor. R1–R6 feed the motion-vision pathways."],
  [/^R[78]$/, "Inner photoreceptor, used for color vision. R7 and R8 bypass the lamina and connect directly in the medulla."],
  [/^L[1-5]$/, "Lamina monopolar cell, the first relay after the photoreceptors. L1 feeds the ON-motion pathway and L2 the OFF pathway."],
  [/^Lawf/, "Lamina wide-field cell. There are far fewer of these than columns."],
  [/^Am$/, "Lamina amacrine cell."],
  [/^C[23]$/, "Centrifugal neuron that sends feedback from the medulla back to the lamina."],
  [/^CT1/, "CT1 is a single giant cell whose branches reach every column. The model splits it into its medulla (M10) and lobula (Lo1) compartments."],
  [/^Mi/, "Medulla intrinsic neuron, which stays within the medulla. Mi1, Mi4 and Mi9 feed the ON-motion detectors (T4)."],
  [/^T[123]a?$/, "Columnar T-shaped neuron."],
  [/^T4/, "ON-motion detector: it responds to bright edges moving in one direction."],
  [/^T5/, "OFF-motion detector: it responds to dark edges moving in one direction."],
  [/^TmY/, "Transmedullary Y neuron. It carries signals from the medulla to both the lobula and the lobula plate."],
  [/^Tm/, "Transmedullary neuron. It carries signals from the medulla to the lobula. Tm1, Tm2, Tm4 and Tm9 feed the OFF-motion detectors (T5)."],
];

function tuningNote(name) {
  const tu = model.data.direction_tuning[name];
  if (!tu) return "";
  const pol = name.startsWith("T4") ? "on" : "off";
  const resp = ["right", "left", "up", "down"].map((d) => tu[`${d}_${pol}`]).sort((a, b) => b - a);
  const weak = resp[0] < 1.2 * resp[1];
  return ` In this model it responds most to motion toward the <strong>${tu.preferred}</strong> of the screen${weak ? " (weakly tuned)" : ""}.`;
}

function connections(k, side) {
  const f = model.data.filters;
  const totals = new Map();
  for (let e = 0; e < f.weight.length; e++) {
    const [self, other] = side === "in" ? [f.target[e], f.source[e]] : [f.source[e], f.target[e]];
    if (self !== k) continue;
    const t = totals.get(other) ?? { count: 0, sign: Math.sign(f.weight[e]) };
    t.count += f.count[e];
    totals.set(other, t);
  }
  return [...totals.entries()]
    .map(([other, t]) => ({ other, ...t }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 7);
}

function barList(k, side) {
  const list = connections(k, side);
  if (!list.length) return `<p class="muted small">None in this model.</p>`;
  const max = list[0].count;
  const sign = (s) => (s > 0 ? "exc" : "inh");
  return `<ul class="bars">${list
    .map(
      (c) => `<li><button type="button" data-type="${c.other}">${model.types[c.other].name}</button>
      <div class="bar ${sign(c.sign)}" style="width:${Math.max(4, (100 * c.count) / max)}%"
        title="${c.sign > 0 ? "excitatory" : "inhibitory"}"></div>
      <span class="n">${c.count < 10 ? c.count.toFixed(1) : Math.round(c.count)}</span></li>`,
    )
    .join("")}</ul>`;
}

function selectType(k) {
  selected = k;
  panels.forEach((p, i) => p.button.classList.toggle("selected", i === k));
  const ct = model.types[k];
  const family = FAMILIES.find(([re]) => re.test(ct.name))?.[1] ?? "";
  const detail = $("detail");
  detail.innerHTML = `
    <h3>${ct.name}</h3>
    <p class="family">${family}${tuningNote(ct.name)}</p>
    <p class="muted small">${fmt.format(ct.n)} cells · time constant ${Math.round(ct.tau * 1000)} ms</p>
    <h4>Center cell, change from rest</h4>
    <canvas id="trace" width="600" height="260" aria-label="Activity of the center ${ct.name} cell over time"></canvas>
    <h4>Strongest inputs <span class="muted">(synapses per cell)</span></h4>
    ${barList(k, "in")}
    <h4>Strongest outputs <span class="muted">(synapses per target cell)</span></h4>
    ${barList(k, "out")}
    <p class="muted small"><span style="color:var(--warm)">■</span> excitatory
      <span style="color:var(--cool)">■</span> inhibitory</p>`;
  detail.querySelectorAll("[data-type]").forEach((b) =>
    b.addEventListener("click", () => selectType(Number(b.dataset.type))),
  );
  drawTrace();
}

function drawTrace() {
  const canvas = $("trace");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const { width: W, height: H } = canvas;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#243140";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();
  if (!job || job.t === 0) {
    ctx.fillStyle = "#8d9bab";
    ctx.font = "24px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Show the fly an email to see this cell respond", W / 2, H / 2 - 16);
    return;
  }
  // Scrolling window over the last TRACE_WINDOW frames; fixed y-scale per run.
  const nTypes = model.types.length;
  const trace = (t) => job.traces[t * nTypes + selected];
  let peak = model.types[selected].scale * 0.1;
  for (let t = 0; t < job.t; t++) peak = Math.max(peak, Math.abs(trace(t)));
  const start = Math.max(0, job.t - TRACE_WINDOW);
  ctx.strokeStyle = "#ffb454";
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let t = start; t < job.t; t++) {
    const x = ((t - start) / TRACE_WINDOW) * W;
    const y = H / 2 - (trace(t) / peak) * (H / 2 - 30);
    t > start ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#8d9bab";
  ctx.font = "22px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`${(start / FPS).toFixed(1)} s`, 8, H - 8);
  ctx.textAlign = "right";
  ctx.fillText(`${(job.t / FPS).toFixed(1)} s`, W - 8, H - 8);
}

// Motion evidence per direction from the direction-selective T4/T5 cells:
// each type's mean depolarization, normalized by its response to a
// full-contrast edge moving its preferred way. `meanDepolarization(k)` gives
// the value for cell type k.
function motionEvidence(meanDepolarization) {
  const evidence = { left: 0, right: 0, up: 0, down: 0 };
  for (const [name, tu] of Object.entries(model.data.direction_tuning)) {
    const pol = name.startsWith("T4") ? "on" : "off";
    evidence[tu.preferred] += meanDepolarization(model.typeIndex[name]) / tu[`${tu.preferred}_${pol}`];
  }
  return evidence;
}

let meter = 0;
function drawMeter() {
  const { left, right } = motionEvidence((k) => {
    const [a, b] = model.typeRange[k];
    let sum = 0;
    for (let i = a; i < b; i++) sum += Math.max(job.state[i] - rest[i], 0);
    return sum / (b - a);
  });
  const target = (right - left) / (right + left + 0.05);
  meter += 0.25 * (target - meter);
  const fill = $("meter-fill");
  fill.style.left = `${50 + Math.min(meter, 0) * 50}%`;
  fill.style.width = `${Math.abs(meter) * 50}%`;
  const word = Math.abs(meter) < 0.1 ? "no clear direction" : meter < 0 ? "leftward" : "rightward";
  $("meter").setAttribute("aria-label", `Motion sensed by the T4/T5 cells: ${word}`);
}

// ---------------------------------------------------------------- the reply

function finish(stats) {
  job.done = true;
  $("skip-btn").disabled = true;
  $("play-btn").disabled = true;
  const { types } = model;

  const evidence = motionEvidence((k) => stats.meanDepolarization[k]);
  const dir = evidence.left >= evidence.right ? "left" : "right";
  const other = dir === "left" ? "right" : "left";
  const ratio = evidence[dir] / Math.max(evidence[other], 1e-9);

  const busiest = types
    .map((ct, k) => ({ name: ct.name, activity: stats.meanActivity[k] }))
    .filter((c) => !/^R\d$/.test(c.name))
    .sort((a, b) => b.activity - a.activity)
    .slice(0, 3)
    .map((c) => c.name);

  const seconds = (job.nFrames / FPS).toFixed(1);
  const chars = job.strip.chars;
  const busy = busiestMoment();
  const motion =
    ratio >= 1.15
      ? `My ${dir}ward-motion detectors (T4 and T5 cells) were ${ratio.toFixed(1)}× as active as the ${other}ward ones, so your words were clearly moving ${dir}. A real fly would probably try to turn with them.`
      : `My motion detectors couldn't agree on a direction (${dir}ward won by only ${ratio.toFixed(2)}×).`;
  const reply = `Bzzz! Your message scrolled past my eye for ${seconds} seconds (${chars} characters).

I have to be honest: I can't read. Each of my 721 eye facets sees one blurry patch of the world, so your words reached me as dark shapes sliding ${dir}.

Here's what my brain did with them:
• ${motion}
• ${busy}
• Relative to their usual range, the cell types that reacted most were ${busiest[0]}, ${busiest[1]} and ${busiest[2]}.

Buzz,
A fruit fly (${fmt.format(model.type.length)} simulated neurons)`;

  $("reply-text").textContent = reply;
  const subject = job.subject ? `Re: ${job.subject}` : "Re: your email";
  $("mail-btn").href = `mailto:${encodeURIComponent(job.from)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(reply)}`;
  $("reply").hidden = false;
  status("Done. The fly has written a reply (below the viewer).");
}

// When was the whole brain most active, and which word was crossing the
// middle of the eye at that moment?
function busiestMoment() {
  let peak = 0;
  for (let t = 1; t < job.nFrames; t++) if (job.activity[t] > job.activity[peak]) peak = t;
  const { text, font } = job.strip;
  const center = peak * job.speed + eye.size / 2 - eye.size; // x within the text
  const ctx = document.createElement("canvas").getContext("2d");
  ctx.font = font;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ctx.measureText(text.slice(0, mid + 1)).width <= center) lo = mid + 1;
    else hi = mid;
  }
  const when = `My brain was busiest ${(peak / FPS).toFixed(1)} s in`;
  if (center < 0 || lo >= text.length || /\s/.test(text[lo])) return `${when}.`;
  const start = text.lastIndexOf(" ", lo) + 1;
  const end = text.indexOf(" ", lo);
  const word = text.slice(start, end === -1 ? undefined : end);
  return `${when}, while “${word}” was crossing the middle of my eye.`;
}

// ---------------------------------------------------------------- wiring

function status(text) {
  $("status").textContent = text;
}

$("email-form").addEventListener("submit", startRun);
$("play-btn").addEventListener("click", () => setPlaying(!playing));
$("skip-btn").addEventListener("click", skipToEnd);
$("gain").addEventListener("input", setGain);
document.querySelectorAll(".rate button").forEach((b) =>
  b.addEventListener("click", () => {
    rate = Number(b.dataset.rate);
    document.querySelectorAll(".rate button").forEach((x) => x.classList.toggle("on", x === b));
  }),
);
$("copy-btn").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("reply-text").textContent);
    $("copy-btn").textContent = "Copied";
    setTimeout(() => ($("copy-btn").textContent = "Copy"), 1500);
  } catch {
    status("Couldn't copy automatically. Select the text and copy it instead.");
  }
});

requestAnimationFrame(tick);
boot().catch((e) => status(`Couldn't load the fly: ${e.message}`));
