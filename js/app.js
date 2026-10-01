// The email page: an email scrolls past the fly's eye while the network runs
// in a worker; afterwards the fly "replies" with what its neurons did.
import { HexEye, StripSampler } from "./sim.js";
import { loadModel } from "./model.js";
import { BrainView, FPS, VIEW, drawHexEye, frameRect } from "./brain-view.js";
import { meanDepolarization, motionBalance } from "./motion.js";

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

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en-US");

const eye = new HexEye();
let model;
let brain;
let worker;
let rest;
let job = null; // the email currently being shown
let runCounter = 0;
let playing = true;
let rate = 1;
let meter = 0;

// ---------------------------------------------------------------- boot

async function boot() {
  model = await loadModel();
  $("model-name").textContent = model.data.meta.model;
  $("fact-neurons").textContent = fmt.format(model.type.length);

  buildExamples();
  brain = new BrainView(model, {
    groups: $("groups"),
    detail: $("detail"),
    gain: $("gain"),
    emptyTrace: "Show the fly an email to see this cell respond",
  });
  drawHexEye($("eye-view"), eye, new Float32Array(eye.n).fill(0.5));
  drawScreen();

  worker = new Worker("js/worker.js", { type: "module" });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => status(`The simulation failed to start: ${e.message}`);
  worker.postMessage({ type: "init", data: model.data });
}

function onWorkerMessage({ data: msg }) {
  if (msg.type === "ready") {
    rest = msg.rest;
    $("fact-synapses").textContent = fmt.format(msg.nEdges);
    brain.setRest(rest);
    $("show-btn").disabled = false;
    $("show-btn").textContent = "Show it to the fly";
    status("Ready. Pick an example or write your own email, then press the button.");
  } else if (msg.type === "progress" && job && msg.run === job.run) {
    job.activity.set(msg.activity, job.t);
    job.t = msg.t;
    job.state = msg.state;
    job.pending = false;
    brain.appendTraces(msg.traces);
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
    activity: new Float32Array(nFrames),
    done: false,
  };
  worker.postMessage({ type: "load", run: job.run, frames, nFrames }, [frames.buffer]);
  brain.resetTraces();
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
  drawHexEye($("eye-view"), eye, job.sampler.frame(shown * job.speed));
  brain.draw(job.state);
  drawMeter();
  $("progress-bar").style.width = `${(100 * job.t) / job.nFrames}%`;
  $("clock").textContent = `${(job.t / FPS).toFixed(1)} s`;
}

function drawScreen(strip, offset = 0) {
  const ctx = $("screen-view").getContext("2d");
  const { origin, size } = frameRect(eye);
  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(0, 0, VIEW, VIEW);
  ctx.fillStyle = "#000";
  ctx.fillRect(origin, origin, size, size);
  ctx.globalAlpha = 0.5; // what the fly gets: paper is 50% grey
  if (strip) {
    ctx.drawImage(strip, offset, 0, eye.size, eye.size, origin, origin, size, size);
  } else {
    ctx.fillStyle = "#fff";
    ctx.fillRect(origin, origin, size, size);
  }
  ctx.globalAlpha = 1;
}

// Live left/right balance of the direction-selective T4/T5 cells.
function drawMeter() {
  const depol = model.detectors.map((d) => meanDepolarization(job.state, rest, model.typeRange[d.k]));
  meter += 0.25 * (motionBalance(model.detectors, depol).balance - meter);
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

  const { evidence } = motionBalance(
    model.detectors,
    model.detectors.map((d) => stats.meanDepolarization[d.k]),
  );
  const dir = evidence.left >= evidence.right ? "left" : "right";
  const other = dir === "left" ? "right" : "left";
  const ratio = evidence[dir] / Math.max(evidence[other], 1e-9);

  const busiest = model.types
    .map((ct, k) => ({ name: ct.name, activity: stats.meanActivity[k] }))
    .filter((c) => !/^R\d$/.test(c.name))
    .sort((a, b) => b.activity - a.activity)
    .slice(0, 3)
    .map((c) => c.name);

  const seconds = (job.nFrames / FPS).toFixed(1);
  const motion =
    ratio >= 1.15
      ? `My ${dir}ward-motion detectors (T4 and T5 cells) were ${ratio.toFixed(1)}× as active as the ${other}ward ones, so your words were clearly moving ${dir}. A real fly would probably try to turn with them.`
      : `My motion detectors couldn't agree on a direction (${dir}ward won by only ${ratio.toFixed(2)}×).`;
  const reply = `Bzzz! Your message scrolled past my eye for ${seconds} seconds (${job.strip.chars} characters).

I have to be honest: I can't read. Each of my 721 eye facets sees one blurry patch of the world, so your words reached me as dark shapes sliding ${dir}.

Here's what my brain did with them:
• ${motion}
• ${busiestMoment()}
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
