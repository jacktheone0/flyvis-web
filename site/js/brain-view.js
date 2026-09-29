// Shared page UI: the 65 cell-type panels and their color scale, the detail
// panel (wiring and a rolling trace of one cell), and the hexagonal eye view.

export const FPS = 50; // simulation frames per second (dt = 20 ms)
const CELL = 4; // panel pixels per lattice step
const PANEL = 31 * CELL; // panel canvas size
const TRACE_SECONDS = 6;

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

// ---------------------------------------------------------------- text

const GROUPS = {
  retina: {
    title: "Photoreceptors",
    blurb: "Light hits these first. Every frame the eye sees is fed into R1–R8.",
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

// ---------------------------------------------------------------- panels

export class BrainView {
  // groups: container for the panels; detail: aside for the selected type;
  // gain: the contrast range input; emptyTrace: text shown before any data.
  constructor(model, { groups, detail, gain, emptyTrace }) {
    this.model = model;
    this.detailEl = detail;
    this.gainInput = gain;
    this.emptyTrace = emptyTrace;
    this.nTypes = model.types.length;
    this.window = TRACE_SECONDS * FPS;
    this.ring = new Float32Array(this.window * this.nTypes);
    this.peak = new Float32Array(this.nTypes);
    this.frames = 0;
    this.buildPanels(groups);
    gain.addEventListener("input", () => this.setGain());
    this.setGain();
    this.select(model.typeIndex.T4a);
  }

  buildPanels(container) {
    const { types, type, u, v } = this.model;
    const byGroup = {};
    for (const [key, g] of Object.entries(GROUPS)) {
      const section = document.createElement("div");
      section.className = "group";
      section.innerHTML = `<h3>${g.title}</h3><p>${g.blurb}</p><div class="panels"></div>`;
      container.append(section);
      byGroup[key] = section.querySelector(".panels");
    }
    this.panels = types.map((ct, k) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "panel";
      button.setAttribute("aria-label", `${ct.name}, ${ct.n} cells`);
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = PANEL;
      const label = document.createElement("span");
      label.textContent = ct.name;
      button.append(canvas, label);
      button.addEventListener("click", () => this.select(k));
      byGroup[ct.group].append(button);
      const ctx = canvas.getContext("2d");
      const image = ctx.createImageData(PANEL, PANEL);
      return { button, ctx, image, pixels: new Uint32Array(image.data.buffer) };
    });
    // Top-left pixel of each neuron's square, with columns offset by half a step.
    this.nodePixel = new Int32Array(type.length);
    for (let i = 0; i < type.length; i++) {
      const x = (v[i] + 15) * CELL;
      const y = (u[i] + 15) * CELL + v[i] * (CELL / 2);
      this.nodePixel[i] = y * PANEL + x;
    }
  }

  setRest(rest) {
    this.rest = rest;
    this.draw(rest);
  }

  draw(state) {
    if (!this.rest) return;
    this.state = state;
    const { type, scale } = this.model;
    const { rest, gain, panels, nodePixel } = this;
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

  setGain() {
    this.gain = 2 ** Number(this.gainInput.value);
    if (this.state) this.draw(this.state);
  }

  // ------------------------------------------------------------ traces

  resetTraces() {
    this.frames = 0;
    this.peak.fill(0);
    this.drawTrace();
  }

  // rows: n frames x nTypes changes from rest of each type's center cell.
  appendTraces(rows) {
    const n = rows.length / this.nTypes;
    for (let r = 0; r < n; r++) {
      const slot = (this.frames % this.window) * this.nTypes;
      for (let k = 0; k < this.nTypes; k++) {
        const value = rows[r * this.nTypes + k];
        this.ring[slot + k] = value;
        if (Math.abs(value) > this.peak[k]) this.peak[k] = Math.abs(value);
      }
      this.frames++;
    }
    this.drawTrace();
  }

  drawTrace() {
    const canvas = this.detailEl.querySelector("canvas");
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
    if (this.frames === 0) {
      ctx.fillStyle = "#8d9bab";
      ctx.font = "24px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(this.emptyTrace, W / 2, H / 2 - 16);
      return;
    }
    const k = this.selected;
    const peak = Math.max(this.peak[k], this.model.types[k].scale * 0.1);
    const start = Math.max(0, this.frames - this.window);
    ctx.strokeStyle = "#ffb454";
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let f = start; f < this.frames; f++) {
      const x = ((f - start) / this.window) * W;
      const y = H / 2 - (this.ring[(f % this.window) * this.nTypes + k] / peak) * (H / 2 - 30);
      f > start ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#8d9bab";
    ctx.font = "22px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${(start / FPS).toFixed(1)} s`, 8, H - 8);
    ctx.textAlign = "right";
    ctx.fillText(`${(this.frames / FPS).toFixed(1)} s`, W - 8, H - 8);
  }

  // ------------------------------------------------------------ detail

  select(k) {
    this.selected = k;
    this.panels.forEach((p, i) => p.button.classList.toggle("selected", i === k));
    const ct = this.model.types[k];
    const family = FAMILIES.find(([re]) => re.test(ct.name))?.[1] ?? "";
    this.detailEl.innerHTML = `
      <h3>${ct.name}</h3>
      <p class="family">${family}${this.tuningNote(ct.name)}</p>
      <p class="muted small">${fmt.format(ct.n)} cells · time constant ${Math.round(ct.tau * 1000)} ms</p>
      <h4>Center cell, change from rest</h4>
      <canvas width="600" height="260" aria-label="Activity of the center ${ct.name} cell over time"></canvas>
      <h4>Strongest inputs <span class="muted">(synapses per cell)</span></h4>
      ${this.barList(k, "in")}
      <h4>Strongest outputs <span class="muted">(synapses per target cell)</span></h4>
      ${this.barList(k, "out")}
      <p class="muted small"><span style="color:var(--warm)">■</span> excitatory
        <span style="color:var(--cool)">■</span> inhibitory</p>`;
    this.detailEl.querySelectorAll("[data-type]").forEach((b) =>
      b.addEventListener("click", () => this.select(Number(b.dataset.type))),
    );
    this.drawTrace();
  }

  tuningNote(name) {
    const tu = this.model.data.direction_tuning[name];
    if (!tu) return "";
    const pol = name.startsWith("T4") ? "on" : "off";
    const resp = ["right", "left", "up", "down"].map((d) => tu[`${d}_${pol}`]).sort((a, b) => b - a);
    const weak = resp[0] < 1.2 * resp[1];
    return ` In this model it responds most to motion toward the <strong>${tu.preferred}</strong> of the screen${weak ? " (weakly tuned)" : ""}.`;
  }

  connections(k, side) {
    const f = this.model.data.filters;
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

  barList(k, side) {
    const list = this.connections(k, side);
    if (!list.length) return `<p class="muted small">None in this model.</p>`;
    const max = list[0].count;
    return `<ul class="bars">${list
      .map(
        (c) => `<li><button type="button" data-type="${c.other}">${this.model.types[c.other].name}</button>
        <div class="bar ${c.sign > 0 ? "exc" : "inh"}" style="width:${Math.max(4, (100 * c.count) / max)}%"
          title="${c.sign > 0 ? "excitatory" : "inhibitory"}"></div>
        <span class="n">${c.count < 10 ? c.count.toFixed(1) : Math.round(c.count)}</span></li>`,
      )
      .join("")}</ul>`;
  }
}

// ---------------------------------------------------------------- eye views

// The 380 px views use 12 px per lattice step (a facet is 13 px of image).
export const VIEW = 380;
const STEP = 12;

// Where a size x size camera frame lands on a view, aligned with the hex eye.
export function frameRect(eye) {
  const s = STEP / eye.kernel;
  return { origin: VIEW / 2 - (eye.size / 2) * s, size: eye.size * s };
}

// Exact hexal rows (eye.y is truncated like flyvis' BoxEye).
function hexRows(extent) {
  const rows = [];
  for (let u = -extent; u <= extent; u++)
    for (let v = Math.max(-extent, -extent - u); v <= Math.min(extent, extent - u); v++) rows.push(u + v / 2);
  return rows;
}

// Draw the eye's 721 facets, shaded by the luminance each one receives.
export function drawHexEye(canvas, eye, values) {
  const rows = (drawHexEye.rows ??= hexRows(15));
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0a0e13";
  ctx.fillRect(0, 0, VIEW, VIEW);
  const a = (2 * STEP) / 3;
  const b = STEP / 2;
  for (let h = 0; h < eye.n; h++) {
    const cx = 10 + (eye.x[h] / eye.kernel + 15) * STEP;
    const cy = 10 + (rows[h] + 15) * STEP;
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
