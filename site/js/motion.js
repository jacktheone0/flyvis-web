// Readouts of the direction-selective T4/T5 motion detectors, shared by the
// email page, the flight simulator and the tests. No DOM access.

// [start, end) node index of every cell type (nodes are grouped by type).
export function typeRanges(type, nTypes) {
  const ranges = Array.from({ length: nTypes }, () => [type.length, 0]);
  for (let i = 0; i < type.length; i++) {
    const r = ranges[type[i]];
    if (i < r[0]) r[0] = i;
    if (i + 1 > r[1]) r[1] = i + 1;
  }
  return ranges;
}

// The T4/T5 subtypes, each with its measured preferred direction and its mean
// depolarization for a full-contrast edge moving that way (for normalizing).
export function motionDetectors(cellTypes, tuning) {
  return Object.entries(tuning).map(([name, tu]) => {
    const pol = name.startsWith("T4") ? "on" : "off";
    return {
      k: cellTypes.findIndex((c) => c.name === name),
      name,
      preferred: tu.preferred,
      norm: tu[`${tu.preferred}_${pol}`],
    };
  });
}

// Mean depolarization (positive change from rest) over nodes [a, b).
export function meanDepolarization(state, rest, [a, b]) {
  let sum = 0;
  for (let i = a; i < b; i++) {
    const d = state[i] - rest[i];
    if (d > 0) sum += d;
  }
  return sum / (b - a);
}

// Evidence for each direction, and the left/right balance in [-1, 1]
// (negative = the world is moving left across the eye). `depol[i]` is the
// mean depolarization of detectors[i].
export function motionBalance(detectors, depol) {
  const evidence = { left: 0, right: 0, up: 0, down: 0 };
  detectors.forEach((d, i) => (evidence[d.preferred] += depol[i] / d.norm));
  const balance = (evidence.right - evidence.left) / (evidence.right + evidence.left + 0.05);
  return { evidence, balance };
}
