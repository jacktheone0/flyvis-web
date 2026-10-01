// Holds everything the fly has been taught and trains its decoder in the
// background, so the game never freezes. The data is kept in IndexedDB so it
// survives reloads. Messages:
//   load {}                         -> loaded {data | null}
//   add {X, d, steer, throttle, weight}  appends frames
//   train {}                        -> trained {decoder} | error {message}
//   clear {}                        -> cleared
import { fitDecoder } from "./learner-core.js";

const DB = "flyvis-polytrack";
const STORE = "dataset";
const MAX_FRAMES = 30000; // about 8 minutes of driving at 60 fps

let d = 0;
let X = new Float32Array(0);
let steer = new Float32Array(0);
let throttle = new Float32Array(0);
let weight = new Float32Array(0);
let n = 0;

function grow(min) {
  if (X.length >= min * d) return;
  const cap = Math.max(min, Math.ceil(n * 1.5) + 1000);
  const copy = (a, len) => {
    const b = new Float32Array(len);
    b.set(a.subarray(0, Math.min(a.length, len)));
    return b;
  };
  X = copy(X, cap * d);
  steer = copy(steer, cap);
  throttle = copy(throttle, cap);
  weight = copy(weight, cap);
}

function append(msg) {
  if (d && msg.d !== d) {
    n = 0; // feature layout changed: start over
  }
  d = msg.d;
  const m = msg.steer.length;
  grow(n + m);
  X.set(msg.X, n * d);
  steer.set(msg.steer, n);
  throttle.set(msg.throttle, n);
  weight.set(msg.weight, n);
  n += m;
  if (n > MAX_FRAMES) {
    // Drop the oldest frames, keeping the most recent teaching.
    const drop = n - MAX_FRAMES;
    X.copyWithin(0, drop * d, n * d);
    for (const a of [steer, throttle, weight]) a.copyWithin(0, drop, n);
    n = MAX_FRAMES;
  }
}

const open = () =>
  new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

async function persist(value) {
  try {
    const db = await open();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      value ? tx.objectStore(STORE).put(value, "data") : tx.objectStore(STORE).delete("data");
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch {
    // storage unavailable: the data still lives for this session
  }
}

async function restore() {
  try {
    const db = await open();
    const value = await new Promise((resolve, reject) => {
      const req = db.transaction(STORE).objectStore(STORE).get("data");
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    db.close();
    return value ?? null;
  } catch {
    return null;
  }
}

const snapshot = () => ({
  d,
  n,
  X: X.slice(0, n * d),
  steer: steer.slice(0, n),
  throttle: throttle.slice(0, n),
  weight: weight.slice(0, n),
});

self.onmessage = async ({ data: msg }) => {
  if (msg.type === "load") {
    const saved = await restore();
    if (saved?.n) append({ ...saved, d: saved.d });
    self.postMessage({ type: "loaded", data: saved?.n ? snapshot() : null });
  } else if (msg.type === "add") {
    append(msg);
  } else if (msg.type === "train") {
    if (n < 300) {
      self.postMessage({ type: "error", message: `Only ${n} frames recorded. Drive a bit more first.` });
      return;
    }
    const decoder = fitDecoder({ X, n, d, steer, throttle, weight });
    self.postMessage({ type: "trained", decoder });
    persist(snapshot());
  } else if (msg.type === "clear") {
    n = 0;
    await persist(null);
    self.postMessage({ type: "cleared" });
  }
};
