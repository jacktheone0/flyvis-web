// PolyModLoader mod: let the flyvis fruit fly drive PolyTrack.
//
// - Adds a "Fly driver" panel: you drive while the fly watches, it learns a
//   decoder and a route memory from its own neurons, then it drives. Correct
//   it with the arrow keys any time; reward it (G) or punish it (B).
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

const DECODER_KEY = "flyvis-polytrack-decoder-v2";
const BRAIN_KEY = "flyvis-polytrack-brain";
const BTN = "padding:7px;border:0;border-radius:6px;font-weight:700;cursor:pointer;background:#243140;color:#e8eef4;";
const SMALL = "color:#8d9bab;font-size:11px;margin-top:8px";

function mountPanel(siteRoot) {
  const panel = document.createElement("div");
  panel.id = "fly-driver-panel";
  panel.style.cssText =
    "position:fixed;right:12px;top:64px;z-index:1000;width:210px;padding:10px;border-radius:10px;" +
    "max-height:calc(100vh - 76px);overflow-y:auto;box-sizing:border-box;transition:box-shadow .2s;" +
    "background:rgba(10,14,19,.9);color:#e8eef4;font:12px/1.4 system-ui,sans-serif;pointer-events:auto";
  panel.innerHTML = `
    <div style="font-weight:700;font-size:14px;margin-bottom:6px">🪰 Fly driver</div>
    <canvas width="380" height="380" style="width:100%;border-radius:6px;background:#0a0e13"></canvas>
    <div data-role="steer" style="position:relative;height:8px;margin:8px 0 2px;background:#18212b;border-radius:99px">
      <div style="position:absolute;top:0;bottom:0;left:50%;width:0;background:#ffb454;border-radius:99px"></div>
    </div>
    <div data-role="text" style="color:#c4cfda;min-height:48px;white-space:pre-line">Loading…</div>
    <div style="${SMALL}">1 · Teach: drive while the fly watches</div>
    <button data-act="watch" style="${BTN}width:100%;margin-top:4px">I drive</button>
    <div style="${SMALL}">2 · Learn</div>
    <button data-act="train" style="${BTN}width:100%;margin-top:4px">Train the fly</button>
    <div style="${SMALL}">Brain</div>
    <div style="display:flex;gap:4px;margin-top:4px">
      <button data-brain="decoder" style="${BTN}flex:1;font-weight:400">Decoder</button>
      <button data-brain="memory" style="${BTN}flex:1;font-weight:400">Route memory</button>
    </div>
    <div style="${SMALL}">3 · Race</div>
    <button data-act="fly" style="${BTN}width:100%;margin-top:4px;background:#ffb454;color:#1d1305">Let the fly drive (F)</button>
    <div style="display:flex;gap:4px;margin-top:6px">
      <button data-act="good" style="${BTN}flex:1;background:#1f4d33">😊 Good (G)</button>
      <button data-act="bad" style="${BTN}flex:1;background:#5a2323">😖 Bad (B)</button>
    </div>
    <div data-role="stats" style="${SMALL}"></div>
    <div style="${SMALL}">While the fly drives, just use the arrow keys to correct it. It gives you control at once and takes it back when you let go. Checkpoints reward it automatically; crashes punish it.</div>
    <button data-act="clear" style="${BTN}width:100%;margin-top:8px;background:transparent;color:#8d9bab;font-weight:400;padding:3px">Forget everything it learned</button>
    <div style="${SMALL}">Slow motion while the fly drives. Leaderboards and multiplayer are off while this mod is loaded.</div>`;
  document.body.append(panel);
  const $ = (sel) => panel.querySelector(sel);
  const buttons = Object.fromEntries([...panel.querySelectorAll("[data-act]")].map((b) => [b.dataset.act, b]));
  const text = $('[data-role="text"]');
  const stats = $('[data-role="stats"]');
  const bar = $('[data-role="steer"] div');
  const eyeCanvas = $("canvas");
  for (const type of ["mousedown", "mouseup", "click", "pointerdown", "pointerup", "keydown"]) {
    panel.addEventListener(type, (e) => e.stopPropagation());
  }

  let core;
  let driverModule;
  let view;
  let memory;
  let learner;
  let decoder = loadDecoder();
  let brain = localStorage.getItem(BRAIN_KEY) || "memory";
  let run = null; // the running driver
  let frames = 0; // frames taught so far
  let sinceTrain = 0;
  let training = false;
  let trainAgain = false;
  let pending = [];
  let flashUntil = 0;

  const say = (msg) => (text.textContent = msg);
  const refresh = () => {
    buttons.watch.textContent = run?.mode === "watch" ? "Stop" : "I drive";
    buttons.fly.textContent = run?.mode === "fly" ? "Stop the fly (F)" : "Let the fly drive (F)";
    buttons.watch.disabled = run?.mode === "fly";
    buttons.fly.disabled = run?.mode === "watch" || !decoder;
    buttons.train.disabled = Boolean(run?.mode === "watch") || training || frames < 300;
    buttons.good.disabled = buttons.bad.disabled = run?.mode !== "fly";
    for (const b of panel.querySelectorAll("[data-brain]")) {
      b.style.outline = b.dataset.brain === brain ? "2px solid #ffb454" : "none";
    }
    const s = run?.state;
    stats.textContent =
      `Taught: ${frames} frames` + (memory ? ` · memory ${memory.size} views` : "") +
      (s && run.mode === "fly" ? `\nCorrections ${s.corrections} · 😊 ${s.good} · 😖 ${s.bad} · resets ${s.resets}` : "");
  };

  // ---- learning
  const flush = () => {
    if (!pending.length) return;
    const d = pending[0].x.length;
    const X = new Float32Array(pending.length * d);
    pending.forEach((s, i) => X.set(s.x, i * d));
    const msg = {
      type: "add", d, X,
      steer: Float32Array.from(pending, (s) => s.steer),
      throttle: Float32Array.from(pending, (s) => s.throttle),
      weight: Float32Array.from(pending, (s) => s.weight),
    };
    learner.postMessage(msg, [X.buffer, msg.steer.buffer, msg.throttle.buffer, msg.weight.buffer]);
    pending = [];
  };
  const learn = (sample) => {
    pending.push(sample);
    frames++;
    sinceTrain++;
    // Route memory keeps every other frame of good driving: yours, and the
    // fly's own when it was rewarded.
    if (sample.source !== "bad" && frames % 2 === 0) memory.add(sample.x, sample.steer, sample.throttle === 1);
    if (pending.length >= 100) flush();
  };
  const train = () => {
    if (training) {
      trainAgain = true;
      return;
    }
    flush();
    training = true;
    sinceTrain = 0;
    learner.postMessage({ type: "train" });
    refresh();
  };
  const policy = (x) => {
    if (!decoder) return null;
    const fromMemory = brain === "memory" ? memory.act(x) : null;
    return fromMemory ?? core.decode(decoder, x);
  };

  const onLearner = ({ data: msg }) => {
    if (msg.type === "loaded" && msg.data) {
      const { X, d, n, steer, throttle } = msg.data;
      frames = n;
      for (let i = 0; i < n; i += 2) memory.add(X.subarray(i * d, (i + 1) * d), steer[i], throttle[i] > 0.5);
      if (decoder?.mean.length === d) memory.setNorm(decoder);
      say(decoder ? "The fly remembers its training. Start a race and press F." : `${n} frames taught. Train the fly.`);
    } else if (msg.type === "loaded") {
      say(decoder ? "Start a race and press F to let the fly drive." : "Start a race, press “I drive” and drive a few clean laps.");
    } else if (msg.type === "trained") {
      decoder = msg.decoder;
      saveDecoder(decoder);
      memory.setNorm(decoder);
      training = false;
      const pct = (v) => `${Math.round(Math.max(0, v) * 100)}%`;
      say(`Trained on ${decoder.nFrames} frames. On driving it didn't train on it predicts ${pct(decoder.score.steer)} of the steering and ${pct(decoder.score.throttle)} of the throttle.`);
      if (trainAgain) {
        trainAgain = false;
        train();
      }
    } else if (msg.type === "error") {
      training = false;
      say(msg.message);
    } else if (msg.type === "cleared") {
      say("Forgotten. Teach it again with “I drive”.");
    }
    refresh();
  };

  // ---- rewards
  const onReward = (r, why, recentFrames) => {
    const examples = core.rewardExamples(recentFrames, r);
    for (const ex of examples) learn(ex);
    if (decoder) core.nudgeDecoder(decoder, recentFrames, r);
    if (r < 0) memory.penalize(recentFrames.flatMap((f) => f.indices ?? []));
    panel.style.boxShadow = r > 0 ? "0 0 0 3px #3ddc84" : "0 0 0 3px #ff5a5a";
    flashUntil = performance.now() + 600;
    const label = { checkpoint: "Checkpoint! Dopamine 😊", crash: "Crash. Pain 😖", stuck: "Stuck. Pain 😖", good: "You: good 😊", bad: "You: bad 😖" }[why] ?? why;
    say(`${label}: ${examples.length} recent moments ${r > 0 ? "reinforced" : "weakened"}.`);
    if (sinceTrain >= 150) train();
  };

  // ---- running
  const stop = () => {
    run.stop();
    const was = run.mode;
    run = null;
    flush();
    if (was === "watch") say(`${frames} frames taught. Drive more laps, or train the fly.`);
    else say("Stopped. Real time is back.");
    if (was === "fly" && sinceTrain >= 25) train();
    refresh();
  };
  const start = async (mode) => {
    if (run) return stop();
    say("Starting the fly's brain…");
    let wasTakeover = false;
    run = await driverModule.startFlyDriver({
      siteRoot,
      canvas: document.getElementById("screen"),
      mode,
      policy,
      onSample: learn,
      onReward,
      onUpdate: (s, input) => {
        if (s.frames % 2) return;
        view.drawHexEye(eyeCanvas, view.eye, input);
        bar.style.left = `${50 + Math.min(s.steer, 0) * 50}%`;
        bar.style.width = `${Math.abs(s.steer) * 50}%`;
        if (performance.now() > flashUntil) panel.style.boxShadow = "none";
        // Retrain in the background after each correction.
        if (wasTakeover && !s.takeover && sinceTrain >= 25) train();
        wasTakeover = s.takeover;
        if (performance.now() > flashUntil) {
          const who = mode === "watch" ? "You're driving. The fly is watching." : s.takeover ? "You've taken over. Correcting…" : `The fly is driving (${brain === "memory" ? "route memory" : "decoder"}).`;
          const turn = Math.abs(s.steer) < 0.05 ? "straight" : s.steer < 0 ? "left" : "right";
          say(`${who}\nSteering ${turn} · ${s.throttle ? "accelerating" : "coasting"}${training ? "\nLearning in the background…" : ""}`);
        }
        if (s.frames % 30 === 0) refresh();
      },
    });
    refresh();
  };

  // ---- controls
  buttons.watch.addEventListener("click", () => start("watch"));
  buttons.fly.addEventListener("click", () => start("fly"));
  buttons.train.addEventListener("click", train);
  buttons.good.addEventListener("click", () => run?.reward(1, "good"));
  buttons.bad.addEventListener("click", () => run?.reward(-1, "bad"));
  buttons.clear.addEventListener("click", () => {
    if (run || !confirm("Forget all teaching laps and the trained fly?")) return;
    learner.postMessage({ type: "clear" });
    memory.clear();
    decoder = null;
    localStorage.removeItem(DECODER_KEY);
    frames = 0;
    refresh();
  });
  for (const b of panel.querySelectorAll("[data-brain]")) {
    b.addEventListener("click", () => {
      brain = b.dataset.brain;
      localStorage.setItem(BRAIN_KEY, brain);
      refresh();
    });
  }
  // In-game hotkeys, so you never need to leave the game: F, G, B.
  window.addEventListener(
    "keydown",
    (e) => {
      if (!e.isTrusted || e.repeat || e.target?.closest?.("input, textarea")) return;
      const act = { KeyF: "fly", KeyG: "good", KeyB: "bad" }[e.code];
      if (!act || buttons[act].disabled) return;
      e.stopPropagation();
      e.preventDefault();
      buttons[act].click();
    },
    true,
  );

  // ---- load the fly
  Promise.all([
    import(`${siteRoot}/polytrack/learner-core.js`),
    import(`${siteRoot}/polytrack/fly-driver.js`),
    import(`${siteRoot}/js/brain-view.js`),
    import(`${siteRoot}/js/sim.js`),
  ]).then(([c, d, v, sim]) => {
    core = c;
    driverModule = d;
    view = { drawHexEye: v.drawHexEye, eye: new sim.HexEye() };
    memory = new core.RouteMemory();
    // Workers must come from the page's own origin: wrap the module in a blob.
    const src = `import "${siteRoot}/polytrack/learner-worker.js";`;
    learner = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })), { type: "module" });
    learner.onmessage = onLearner;
    learner.postMessage({ type: "load" });
    refresh();
  }).catch((err) => say(`Couldn't load the fly: ${err.message}`));
  refresh();
}

function saveDecoder(d) {
  const plain = { ...d, mean: [...d.mean], std: [...d.std], steer: [...d.steer], throttle: [...d.throttle] };
  try {
    localStorage.setItem(DECODER_KEY, JSON.stringify(plain));
  } catch {
    // storage full or blocked: the decoder still works this session
  }
}

function loadDecoder() {
  try {
    const d = JSON.parse(localStorage.getItem(DECODER_KEY));
    return d?.steer && { ...d, mean: Float64Array.from(d.mean), std: Float64Array.from(d.std) };
  } catch {
    return null;
  }
}

export const polyMod = new FlyDriverMod();
