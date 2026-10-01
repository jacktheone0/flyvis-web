// The flight simulator's world: a procedural 360° landscape that wraps around
// the fly, and the wind gusts that spin it. No DOM access (runs in workers
// and Node).

// Pixels for a full turn. An eye facet is 13 px apart, so one facet spans
// 5°, about the spacing of a real fruit fly's facets.
export const WORLD_WIDTH = 936;
export const DEG_PER_PX = 360 / WORLD_WIDTH;

export function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
}

// Smooth value noise on a coarse grid; periodic in x so the world wraps.
function valueNoise(random, width, height, cell) {
  const gw = Math.round(width / cell);
  const gh = Math.ceil(height / cell) + 2;
  const grid = Array.from({ length: gh }, () => Float32Array.from({ length: gw }, random));
  const smooth = (t) => t * t * (3 - 2 * t);
  return (x, y) => {
    const fx = (x / width) * gw;
    const fy = y / cell;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const tx = smooth(fx - x0);
    const ty = smooth(fy - y0);
    const row0 = grid[y0];
    const row1 = grid[y0 + 1];
    const a = row0[x0 % gw];
    const b = row0[(x0 + 1) % gw];
    const c = row1[x0 % gw];
    const d = row1[(x0 + 1) % gw];
    return (a + (b - a) * tx) * (1 - ty) + (c + (d - c) * tx) * ty;
  };
}

// Luminance (0..1) of a WORLD_WIDTH x height panorama: sky with clouds, rolling
// hills, textured ground and a scattering of trees.
export function makeWorld(seed, height) {
  const W = WORLD_WIDTH;
  const H = height;
  const random = rng(seed);
  const hills = [1, 2, 3, 5].map((f) => ({
    f,
    a: ((0.5 + 0.5 * random()) * 30) / f,
    p: random() * 2 * Math.PI,
  }));
  const horizon = (x) =>
    H * 0.56 - hills.reduce((s, h) => s + h.a * Math.sin((2 * Math.PI * h.f * x) / W + h.p), 0);
  const ground = valueNoise(random, W, H, 16);
  const clouds = valueNoise(random, W, H, 52);

  const img = new Float32Array(W * H);
  for (let x = 0; x < W; x++) {
    const hz = horizon(x);
    for (let y = 0; y < H; y++) {
      const sky = 0.9 - 0.18 * (y / hz) + 0.22 * (clouds(x, y) - 0.5);
      const land = 0.4 - 0.12 * ((y - hz) / (H - hz)) + 0.3 * (ground(x, y) - 0.5);
      const t = Math.min(1, Math.max(0, y - hz + 0.5)); // antialiased horizon
      img[y * W + x] = sky * (1 - t) + land * t;
    }
  }

  // Trees: dark trunks and round canopies, drawn with soft edges.
  const paint = (x, y, value, alpha) => {
    const i = y * W + (((x % W) + W) % W);
    img[i] += (value - img[i]) * alpha;
  };
  const nTrees = 10 + Math.floor(random() * 6);
  for (let n = 0; n < nTrees; n++) {
    const x0 = random() * W;
    const base = horizon(x0) + 6;
    const top = horizon(x0) - (35 + random() * 115);
    const radius = 12 + random() * 22;
    const trunk = 3 + random() * 3;
    const shade = 0.06 + random() * 0.12;
    for (let y = Math.max(0, Math.floor(top - radius)); y < Math.min(H, Math.ceil(base)); y++) {
      for (let dx = -Math.ceil(radius) - 1; dx <= Math.ceil(radius) + 1; dx++) {
        const x = Math.floor(x0) + dx;
        const cx = x + 0.5 - x0;
        const inTrunk = Math.min(1, Math.max(0, trunk - Math.abs(cx) + 0.5)) * (y >= top ? 1 : 0);
        const inCanopy = Math.min(1, Math.max(0, radius - Math.hypot(cx, y + 0.5 - top) + 0.5));
        const alpha = Math.max(inTrunk, inCanopy);
        if (alpha > 0) paint(x, y, shade, alpha);
      }
    }
  }
  return img;
}

// Wind: a smooth random rotation speed (px/frame, RMS about 0.8) made of slow
// drifts and faster gusts. `strength` scales it.
export function makeGusts(seed) {
  const random = rng(seed ^ 0x9e3779b9);
  const parts = [
    [0.9, 97],
    [0.6, 23],
    [0.35, 7.3],
    [0.2, 3.1],
  ].map(([a, period]) => ({ a, period: period * (0.8 + 0.4 * random()), phase: random() * 2 * Math.PI }));
  return (t) => parts.reduce((s, p) => s + p.a * Math.sin(t / p.period + p.phase), 0);
}
