// In-browser port of the flyvis network dynamics (PPNeuronIGRSynapses) and of
// the BoxEye renderer. Plain ES module with no DOM access, so it runs in a Web
// Worker and in Node (tests/sim.test.mjs).
//
// Dynamics, integrated with forward Euler exactly as flyvis does:
//   tau_i dV_i/dt = -V_i + bias_i + sum_j w_ij * relu(V_j) + x_i(t)
// with tau clamped below at dt.

export function decodeBase64(text, Type) {
  const bin = atob(text);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

// Hexal (u, v) coordinates of the eye in flyvis order.
export function hexLattice(extent) {
  const u = [];
  const v = [];
  for (let uu = -extent; uu <= extent; uu++) {
    const vMin = Math.max(-extent, -extent - uu);
    const vMax = Math.min(extent, extent - uu);
    for (let vv = vMin; vv <= vMax; vv++) {
      u.push(uu);
      v.push(vv);
    }
  }
  return { u: Int8Array.from(u), v: Int8Array.from(v) };
}

export class FlyNetwork {
  constructor(data) {
    this.dt = data.dt;
    this.extent = data.extent;
    this.cellTypes = data.cellTypes ?? data.cell_types;
    const type = decodeBase64(data.nodes.type, Uint8Array);
    const u = decodeBase64(data.nodes.u, Int8Array);
    const v = decodeBase64(data.nodes.v, Int8Array);
    const n = type.length;
    this.n = n;
    this.type = type;
    this.u = u;
    this.v = v;

    // Per-type lookup from lattice position to node index.
    const side = 2 * this.extent + 1;
    const nTypes = this.cellTypes.length;
    this.side = side;
    this.lookup = new Int32Array(nTypes * side * side).fill(-1);
    this.typeStart = new Int32Array(nTypes + 1);
    for (let i = 0; i < n; i++) {
      this.lookup[this.cellIndex(type[i], u[i], v[i])] = i;
      this.typeStart[type[i] + 1] = i + 1;
    }

    this.bias = new Float64Array(n);
    this.gain = new Float64Array(n); // dt / max(tau, dt)
    for (let i = 0; i < n; i++) {
      const ct = this.cellTypes[type[i]];
      this.bias[i] = ct.bias;
      this.gain[i] = this.dt / Math.max(ct.tau, this.dt);
    }

    this.buildSynapses(data.filters);

    // Photoreceptor nodes that receive each eye hexal.
    const eye = hexLattice(this.extent);
    this.nHexals = eye.u.length;
    this.inputNodes = [];
    for (const t of data.inputTypes ?? data.input_types) {
      const nodes = new Int32Array(this.nHexals);
      for (let h = 0; h < this.nHexals; h++) {
        nodes[h] = this.lookup[this.cellIndex(t, eye.u[h], eye.v[h])];
      }
      this.inputNodes.push(nodes);
    }

    this.state = new Float64Array(n);
    this.next = new Float64Array(n);
    this.rectified = new Float64Array(n);
    this.input = new Float64Array(n);
    this.reset();
  }

  cellIndex(t, u, v) {
    const e = this.extent;
    return (t * this.side + (u + e)) * this.side + (v + e);
  }

  // Expand the shared filter table into a target-major (CSR) synapse list.
  buildSynapses(filters) {
    const nTypes = this.cellTypes.length;
    const byTarget = Array.from({ length: nTypes }, () => []);
    for (let k = 0; k < filters.weight.length; k++) {
      byTarget[filters.target[k]].push({
        source: filters.source[k],
        du: filters.du[k],
        dv: filters.dv[k],
        weight: filters.weight[k],
      });
    }
    const e = this.extent;
    const sourceOf = (f, i) => {
      const us = this.u[i] - f.du;
      const vs = this.v[i] - f.dv;
      if (us < -e || us > e || vs < -e || vs > e) return -1;
      return this.lookup[this.cellIndex(f.source, us, vs)];
    };
    const rowStart = new Int32Array(this.n + 1);
    for (let i = 0; i < this.n; i++) {
      let count = 0;
      for (const f of byTarget[this.type[i]]) if (sourceOf(f, i) >= 0) count++;
      rowStart[i + 1] = rowStart[i] + count;
    }
    const nEdges = rowStart[this.n];
    const source = new Int32Array(nEdges);
    const weight = new Float32Array(nEdges);
    for (let i = 0; i < this.n; i++) {
      let k = rowStart[i];
      for (const f of byTarget[this.type[i]]) {
        const j = sourceOf(f, i);
        if (j >= 0) {
          source[k] = j;
          weight[k] = f.weight;
          k++;
        }
      }
    }
    this.rowStart = rowStart;
    this.source = source;
    this.weight = weight;
    this.nEdges = nEdges;
    this.byTarget = byTarget;
  }

  reset() {
    this.state.set(this.bias);
  }

  // Present one frame of the 721 eye hexals (luminance 0..1) and advance dt.
  step(hexals) {
    const { n, state, next, rectified, input, rowStart, source, weight, bias, gain } =
      this;
    for (let i = 0; i < n; i++) rectified[i] = state[i] > 0 ? state[i] : 0;
    for (const nodes of this.inputNodes) {
      for (let h = 0; h < nodes.length; h++) input[nodes[h]] = hexals[h];
    }
    for (let i = 0; i < n; i++) {
      let current = 0;
      const end = rowStart[i + 1];
      for (let k = rowStart[i]; k < end; k++) current += weight[k] * rectified[source[k]];
      next[i] = state[i] + gain[i] * (-state[i] + bias[i] + current + input[i]);
    }
    this.state = next;
    this.next = state;
  }

  // flyvis' Network.steady_state: start at rest (bias), then show uniform grey.
  steadyState(value = 0.5, seconds = 1.0) {
    this.reset();
    const grey = new Float64Array(this.nHexals).fill(value);
    const steps = Math.floor(seconds / this.dt + 1e-9);
    for (let s = 0; s < steps; s++) this.step(grey);
    return Float64Array.from(this.state);
  }
}

// Port of flyvis.datasets.rendering.BoxEye(extent=15, kernel_size=13): each
// hexal is the mean of a kernel x kernel box around its receptor center, with
// zeros outside the frame.
export class HexEye {
  constructor(extent = 15, kernel = 13) {
    const { u, v } = hexLattice(extent);
    this.n = u.length;
    this.kernel = kernel;
    this.y = new Int32Array(this.n);
    this.x = new Int32Array(this.n);
    for (let h = 0; h < this.n; h++) {
      this.y[h] = Math.trunc(kernel * (u[h] + v[h] / 2));
      this.x[h] = kernel * v[h];
    }
    // Frame size that holds every receptor center (391 px for the defaults).
    this.size = 2 * Math.max(...this.x) + 1;
    this.rows = [...new Set(this.y)].sort((a, b) => a - b);
    this.rowOf = Int32Array.from(this.y, (y) => this.rows.indexOf(y));
  }

  // Sample a single size x size frame (Float32Array, row-major).
  sampleFrame(image, out = new Float32Array(this.n)) {
    const sampler = new StripSampler(this, image, this.size);
    return sampler.frame(0, out);
  }
}

// Efficient BoxEye sampling of a size-high strip that slides horizontally
// under the eye (the scrolling email). Vertical box sums for every receptor
// row are precomputed, then prefix-summed along x.
export class StripSampler {
  constructor(eye, strip, width) {
    this.eye = eye;
    this.width = width;
    const { size, kernel, rows } = eye;
    const half = size >> 1;
    const lo = Math.ceil((kernel - 1) / 2);
    const hi = Math.floor((kernel - 1) / 2);
    this.lo = lo;
    this.hi = hi;
    this.prefix = rows.map((y) => {
      const yc = y + half;
      const y0 = Math.max(0, yc - lo);
      const y1 = Math.min(size - 1, yc + hi);
      const p = new Float64Array(width + 1);
      for (let x = 0; x < width; x++) {
        let s = 0;
        for (let yy = y0; yy <= y1; yy++) s += strip[yy * width + x];
        p[x + 1] = p[x] + s;
      }
      return p;
    });
  }

  // Eye input when the frame's left edge sits at strip column `offset`.
  frame(offset, out = new Float32Array(this.eye.n)) {
    const { eye, lo, hi, prefix } = this;
    const half = eye.size >> 1;
    const area = eye.kernel * eye.kernel;
    const xMin = offset;
    const xMax = Math.min(offset + eye.size - 1, this.width - 1);
    for (let h = 0; h < eye.n; h++) {
      const xc = offset + eye.x[h] + half;
      const x0 = Math.max(xMin, xc - lo);
      const x1 = Math.min(xMax, xc + hi);
      const p = prefix[eye.rowOf[h]];
      out[h] = x1 >= x0 ? (p[x1 + 1] - p[x0]) / area : 0;
    }
    return out;
  }
}
