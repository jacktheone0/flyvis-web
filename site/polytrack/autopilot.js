// A simple road-following autopilot, used only as a teacher: it drives a
// lap while the fly's neurons are recorded, so a decoder can learn to drive
// from them. It finds PolyTrack's asphalt colour in the rows just below the
// horizon and steers toward the middle of the road stretch it is following.

import { hudSpeed } from "./hud.js";

const isRoad = (r, g, b) => Math.abs(r - 95) < 28 && Math.abs(g - 112) < 28 && Math.abs(b - 132) < 30 && b > r;

export function makeAutopilot({ speed = 50, gain = 3.5 } = {}) {
  let aim = 0.5; // where the road is, as a fraction of the image width
  return (state, regions, { rgb, width: W, height: H }) => {
    let wsum = 0;
    let xsum = 0;
    for (let y = Math.floor(H * 0.515); y <= Math.floor(H * 0.64); y++) {
      // Runs of road pixels in this row; the car itself counts as road.
      const runs = [];
      let start = -1;
      for (let x = 0; x <= W; x++) {
        const car = y > H * 0.56 && x > W * 0.38 && x < W * 0.62;
        let road = false;
        if (x < W) {
          const i = 4 * (y * W + x);
          road = car || isRoad(rgb[i], rgb[i + 1], rgb[i + 2]);
        }
        if (road && start < 0) start = x;
        if (!road && start >= 0) {
          runs.push([start, x - 1]);
          start = -1;
        }
      }
      let best = null;
      let bestDist = Infinity;
      for (const [a, b] of runs) {
        const d = Math.max(0, a / W - aim, aim - b / W);
        if (d < bestDist) {
          bestDist = d;
          best = [a, b];
        }
      }
      if (!best || best[1] - best[0] < 2) continue;
      const weight = 1 + (H * 0.64 - y); // rows nearer the horizon look further ahead
      xsum += (weight * (best[0] + best[1])) / 2 / W;
      wsum += weight;
    }
    if (wsum > 0) aim += 0.5 * (xsum / wsum - aim);

    return {
      steer: Math.max(-1, Math.min(1, gain * (aim - 0.5))),
      throttle: state.frames > 30 && hudSpeed() < speed,
    };
  };
}
