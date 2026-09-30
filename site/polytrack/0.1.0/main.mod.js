// PolyModLoader mod: let the flyvis fruit fly drive PolyTrack.
//
// - Adds a "Fly driver" panel: teach the fly (an autopilot or you drive while
//   its neurons are recorded), train a decoder on its output neurons, then let
//   the fly drive from its neurons alone. Shows what the fly's eye sees.
// - Patches the physics worker's clock so the game can run in lockstep with
//   the fly's brain (slow motion). Real time resumes when the fly stops.
// - Blocks leaderboard submissions, record verification, profile writes and
//   multiplayer while the mod is loaded: slow-motion laps are not real laps.
const { MixinType, PolyMod } = await import(new URL("PolyTypes.js", document.baseURI).href).catch(
  () => import("https://cdn.polymodloader.com/pml/PolyModLoader/0.6.3/PolyTypes.js"),
);

const CLOCK_CHANNEL = "flyvis-polytrack-clock";

// Injected into the simulation worker right after the physics library loads
// (inside a comma expression, hence the leading comma). While locked, the
// worker's clock only moves when the page says so.
const WORKER_CLOCK = `, (() => {
  const realNow = performance.now.bind(performance);
  let locked = false, virtual = 0, offset = 0;
  const channel = new BroadcastChannel("${CLOCK_CHANNEL}");
  channel.onmessage = ({ data }) => {
    if (data?.type === "lock") {
      if (data.on && !locked) { virtual = realNow() + offset; locked = true; }
      else if (!data.on && locked) { offset = virtual - realNow(); locked = false; }
    } else if (data?.type === "advance" && locked) {
      virtual += data.ms;
    }
  };
  performance.now = () => (locked ? virtual : realNow() + offset);
})()`;

class FlyDriverMod extends PolyMod {
  touchingPhysics = true; // game time is altered, so this is not a vanilla session

  preInit = (pml) => {
    pml.registerSimWorkerMixin({
      type: MixinType.INSERT,
      token: 'importScripts("lib/polytrack_physics.js")',
      func: WORKER_CLOCK,
    });
    for (const token of [
      "submitLeaderboard(e, t, n, i, r, a, s, o) {",
      "submitUserProfile(e, t, n, i) {",
      "verifyRecordings(e, t, n, i, r) {",
      "getIceServers() {",
    ]) {
      pml.registerGlobalMixin({
        type: MixinType.INSERT,
        token,
        func: 'return Promise.reject(new Error("Fly Driver mod: offline only"));',
      });
    }
    for (const token of ["createMultiplayerHostWebSocket() {", "createMultiplayerJoinWebSocket() {"]) {
      pml.registerGlobalMixin({
        type: MixinType.INSERT,
        token,
        func: 'throw new Error("Fly Driver mod: offline only");',
      });
    }
  };

  // postInit always runs; onGameLoad is triggered by PML's core mod. Mount
  // the panel from whichever comes first.
  postInit = () => this.mount();
  onGameLoad = () => this.mount();

  mount() {
    if (document.getElementById("fly-driver-panel")) return;
    const base = (this.baseUrl ?? new URL("..", import.meta.url).href).replace(/\/+$/, "");
    const siteRoot = base.replace(/\/polytrack$/, "");
    const tryMount = () => (document.body ? mountPanel(siteRoot) : requestAnimationFrame(tryMount));
    tryMount();
  }
}

const STORE = "flyvis-polytrack-decoder";
const BUTTON = "width:100%;margin-top:6px;padding:7px;border:0;border-radius:6px;font-weight:700;cursor:pointer;";

function mountPanel(siteRoot) {
  const panel = document.createElement("div");
  panel.id = "fly-driver-panel";
  panel.style.cssText =
    "position:fixed;right:12px;top:64px;z-index:1000;width:200px;padding:10px;border-radius:10px;" +
    "max-height:calc(100vh - 76px);overflow-y:auto;box-sizing:border-box;" +
    "background:rgba(10,14,19,.88);color:#e8eef4;font:12px/1.4 system-ui,sans-serif;pointer-events:auto";
  panel.innerHTML = `
    <div style="font-weight:700;font-size:14px;margin-bottom:6px">🪰 Fly driver</div>
    <canvas width="380" height="380" style="width:100%;border-radius:6px;background:#0a0e13"></canvas>
    <div data-role="steer" style="position:relative;height:8px;margin:8px 0 2px;background:#18212b;border-radius:99px">
      <div style="position:absolute;top:0;bottom:0;left:50%;width:0;background:#ffb454;border-radius:99px"></div>
    </div>
    <div data-role="text" style="color:#c4cfda;min-height:34px;white-space:pre-line">Start a race, then teach the fly one lap.</div>
    <div style="margin-top:8px;color:#8d9bab;font-size:11px">1 · Teach (the fly watches)</div>
    <div style="display:flex;gap:6px">
      <button data-act="auto" style="${BUTTON}background:#243140;color:#e8eef4">Autopilot drives</button>
      <button data-act="human" style="${BUTTON}background:#243140;color:#e8eef4">I drive</button>
    </div>
    <div style="margin-top:8px;color:#8d9bab;font-size:11px">2 · Learn</div>
    <button data-act="train" style="${BUTTON}background:#243140;color:#e8eef4" disabled>Train the fly</button>
    <button data-act="clear" style="${BUTTON}background:transparent;color:#8d9bab;font-weight:400;padding:3px" disabled>Forget recorded laps</button>
    <div style="margin-top:8px;color:#8d9bab;font-size:11px">3 · Race</div>
    <button data-act="fly" style="${BUTTON}background:#ffb454;color:#1d1305" disabled>Let the fly drive</button>
    <div style="color:#8d9bab;margin-top:8px;font-size:11px">The fly and the autopilot run in slow motion: game time waits for the fly's brain. Leaderboards and multiplayer are off while this mod is loaded.</div>`;
  document.body.append(panel);
  const $ = (sel) => panel.querySelector(sel);
  const buttons = Object.fromEntries([...panel.querySelectorAll("button")].map((b) => [b.dataset.act, b]));
  const eyeCanvas = $("canvas");
  const bar = $('[data-role="steer"] div');
  const text = $('[data-role="text"]');
  for (const type of ["mousedown", "mouseup", "click", "pointerdown", "pointerup", "keydown"]) {
    panel.addEventListener(type, (e) => e.stopPropagation());
  }

  let modules = null;
  let run = null; // { mode, driver }
  let recording = null;
  let decoder = loadDecoder();
  const LABELS = { auto: "Autopilot drives", human: "I drive", fly: "Let the fly drive" };

  const load = async () =>
    (modules ??= await Promise.all([
      import(`${siteRoot}/polytrack/fly-driver.js`),
      import(`${siteRoot}/polytrack/autopilot.js`),
      import(`${siteRoot}/js/brain-view.js`),
      import(`${siteRoot}/js/sim.js`),
    ]).then(([driver, autopilot, view, sim]) => ({ driver, autopilot, view, eye: new sim.HexEye() })));

  const refresh = () => {
    for (const act of ["auto", "human", "fly"]) {
      buttons[act].textContent = run?.mode === act ? "Stop" : LABELS[act];
      buttons[act].disabled = Boolean(run && run.mode !== act);
    }
    buttons.fly.disabled ||= !decoder;
    buttons.train.disabled = Boolean(run) || !recording || recording.features.length < 500;
    buttons.clear.disabled = Boolean(run) || !recording;
  };

  const stop = () => {
    run.driver.stop();
    if (run.mode !== "fly") {
      // Recordings add up, so several laps can be taught one at a time.
      const r = run.driver.recording;
      recording ??= { features: [], steer: [], throttle: [] };
      for (const key of ["features", "steer", "throttle"]) recording[key].push(...r[key]);
      text.textContent = `${recording.features.length} frames recorded in total. Teach more laps, or train the fly.`;
    } else {
      text.textContent = "Stopped. Real time is back.";
    }
    run = null;
    refresh();
  };

  const start = async (mode) => {
    if (run) return stop();
    text.textContent = "Loading the fly's brain…";
    const m = await load();
    const options =
      mode === "auto" ? { record: true, control: m.autopilot.makeAutopilot() }
      : mode === "human" ? { record: true, human: true, lockstep: false }
      : { decoder };
    const driver = await m.driver.startFlyDriver({
      siteRoot,
      canvas: document.getElementById("screen"),
      options,
      onUpdate: (s, input) => {
        if (s.frames % 2) return;
        m.view.drawHexEye(eyeCanvas, m.eye, input);
        bar.style.left = `${50 + Math.min(s.steer, 0) * 50}%`;
        bar.style.width = `${Math.abs(s.steer) * 50}%`;
        const who = mode === "fly" ? "The fly is driving" : mode === "auto" ? "Autopilot driving, fly watching" : "You're driving, fly watching";
        const turn = Math.abs(s.steer) < 0.05 ? "straight" : s.steer < 0 ? "left" : "right";
        text.textContent = `${who}\nSteering ${turn} · ${s.throttle ? "accelerating" : "coasting"}` +
          (mode === "fly" ? "" : `\nThis session: ${driver.recording.features.length} frames`);
      },
    });
    run = { mode, driver };
    refresh();
  };

  buttons.auto.addEventListener("click", () => start("auto"));
  buttons.human.addEventListener("click", () => start("human"));
  buttons.fly.addEventListener("click", () => start("fly"));
  buttons.clear.addEventListener("click", () => {
    recording = null;
    text.textContent = "Recording cleared.";
    refresh();
  });
  buttons.train.addEventListener("click", async () => {
    buttons.train.disabled = true;
    text.textContent = "Training…";
    const m = await load();
    await new Promise((r) => setTimeout(r, 30)); // let the text paint
    decoder = await m.driver.trainDecoder(recording, { siteRoot });
    saveDecoder(decoder);
    const pct = (x) => `${Math.round(Math.max(0, x) * 100)}%`;
    text.textContent =
      `Trained on ${decoder.nFrames} frames of driving.\n` +
      `On driving it didn't train on, it explains ${pct(decoder.score.steer)} of the steering and ${pct(decoder.score.throttle)} of the throttle.`;
    refresh();
  });
  if (decoder) text.textContent = "A trained fly is saved. Start a race and let it drive, or teach it again.";
  refresh();
}

function saveDecoder(d) {
  const plain = { ...d, mean: [...d.mean], std: [...d.std], steer: [...d.steer], throttle: [...d.throttle] };
  try {
    localStorage.setItem(STORE, JSON.stringify(plain));
  } catch {
    // storage full or blocked: the decoder still works this session
  }
}

function loadDecoder() {
  try {
    const d = JSON.parse(localStorage.getItem(STORE));
    return d && { ...d, mean: Float64Array.from(d.mean), std: Float64Array.from(d.std) };
  } catch {
    return null;
  }
}

export const polyMod = new FlyDriverMod();
