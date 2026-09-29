// Loads network.json for the page's UI (the simulation itself runs in a worker).
import { decodeBase64 } from "./sim.js";
import { motionDetectors, typeRanges } from "./motion.js";

export async function loadModel(url = "data/network.json") {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} loading ${url}`);
  const data = await response.json();
  const type = decodeBase64(data.nodes.type, Uint8Array);
  return {
    data,
    type,
    types: data.cell_types,
    typeIndex: Object.fromEntries(data.cell_types.map((c, k) => [c.name, k])),
    u: decodeBase64(data.nodes.u, Int8Array),
    v: decodeBase64(data.nodes.v, Int8Array),
    scale: Float32Array.from(type, (k) => data.cell_types[k].scale),
    typeRange: typeRanges(type, data.cell_types.length),
    detectors: motionDetectors(data.cell_types, data.direction_tuning),
  };
}
