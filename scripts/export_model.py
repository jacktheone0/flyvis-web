"""Export a pretrained flyvis network for the in-browser simulator.

The flyvis network shares its parameters across the hexagonal lattice: every
cell of a type has the same resting potential and time constant, and every
synapse between two types at a given lattice offset (du, dv) has the same
weight. That makes the whole 45,669-neuron / 1.5M-synapse network expressible
as a table of ~2,400 "filter" entries, which the browser expands back into the
full synapse list at load time.

Outputs:
  site/data/network.json          network parameters + display calibration
  tests/fixtures/reference.json   flyvis simulations the JS port is tested against

Usage (after `pip install flyvis` and `flyvis download-pretrained`):
  python scripts/export_model.py [--model flow/0000/000]
"""

import argparse
import base64
import datetime
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import flyvis
from flyvis import NetworkView
from flyvis.datasets.rendering import BoxEye

ROOT = Path(__file__).resolve().parents[1]
DT = 1 / 50  # the integration step the models were trained with
FRAME = 391  # BoxEye(extent=15, kernel_size=13).min_frame_size


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def check_filter_expansion(connectome, types, filters):
    """Rebuild the synapse list from the filter table exactly as site/js/sim.js
    does, and assert it matches the connectome edge for edge."""
    node_type = connectome.nodes.type[:].astype(str)
    u, v = connectome.nodes.u[:], connectome.nodes.v[:]
    lookup = {(t, uu, vv): i for i, (t, uu, vv) in enumerate(zip(node_type, u, v))}
    by_target = defaultdict(list)
    for k, (s, t, du, dv, *_) in enumerate(filters):
        by_target[types[t]].append((types[s], du, dv, k))
    rebuilt = set()
    for i, (t, uu, vv) in enumerate(zip(node_type, u, v)):
        for s, du, dv, k in by_target[t]:
            j = lookup.get((s, uu - du, vv - dv))
            if j is not None:
                rebuilt.add((j, i, k))
    src = connectome.edges.source_index[:]
    tgt = connectome.edges.target_index[:]
    assert len(rebuilt) == len(src), (len(rebuilt), len(src))
    assert {(j, i) for j, i, _ in rebuilt} == set(zip(src.tolist(), tgt.tolist()))
    return len(src)


def moving_edge(direction: str, polarity: float, speed: float = 4.0) -> np.ndarray:
    """Cartesian frames of an edge sweeping across a grey (0.5) background."""
    n = int((FRAME + 40) / speed)
    frames = np.full((n, FRAME, FRAME), 0.5, dtype=np.float32)
    grid = np.arange(FRAME)
    for f in range(n):
        pos = -20 + f * speed
        if direction == "right":
            frames[f][:, grid < pos] = polarity
        elif direction == "left":
            frames[f][:, grid > FRAME - pos] = polarity
        elif direction == "down":
            frames[f][grid < pos, :] = polarity
        elif direction == "up":
            frames[f][grid > FRAME - pos, :] = polarity
    return frames


def procedural_image() -> np.ndarray:
    """Deterministic test image, reproduced in tests/sim.test.mjs."""
    y, x = np.mgrid[:FRAME, :FRAME]
    return (((x * 7 + y * 13) % 17) / 16).astype(np.float32)


def fixture_movie() -> np.ndarray:
    """A short cartesian movie: a dark bar crossing a textured background."""
    base = procedural_image() * 0.5 + 0.25
    frames = []
    for f in range(60):
        frame = base.copy()
        x0 = 40 + 5 * f
        frame[:, x0 : x0 + 30] = 0.0
        frames.append(frame)
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="flow/0000/000")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    view = NetworkView(args.model)
    net = view.init_network(checkpoint="best")
    connectome = net.connectome
    params = net._param_api()

    types = connectome.unique_cell_types[:].astype(str).tolist()
    layout = dict(connectome.layout[:].astype(str))
    node_type = connectome.nodes.type[:].astype(str)
    type_index = np.array([types.index(t) for t in node_type], dtype=np.uint8)
    assert np.all(np.diff(type_index.astype(int)) >= 0), "nodes must be grouped by type"

    bias = params.nodes.bias.numpy()
    tau = params.nodes.time_const.numpy()
    weight = params.edges.weight.numpy()
    syn_count = params.edges.syn_count.numpy()

    # One filter entry per (source type, target type, du, dv); verify sharing.
    e = connectome.edges
    keys = zip(
        e.source_type[:].astype(str), e.target_type[:].astype(str), e.du[:], e.dv[:]
    )
    entries = {}
    for key, w, n in zip(keys, weight, syn_count):
        prev = entries.setdefault(key, (w, n))
        assert prev == (w, n), f"parameters not shared for {key}"
    filters = sorted(
        (types.index(s), types.index(t), int(du), int(dv), float(w), float(n))
        for (s, t, du, dv), (w, n) in entries.items()
    )
    n_edges = check_filter_expansion(connectome, types, filters)

    # Reference simulations with the real flyvis network.
    eye = BoxEye()

    def simulate(frames: np.ndarray) -> np.ndarray:
        hex_movie = eye(torch.from_numpy(frames)[None])  # (1, T, 1, 721)
        return hex_movie, net.simulate(hex_movie, DT).numpy()[0]  # (T, nodes)

    rest = net.steady_state(1.0, DT, 1, value=0.5).nodes.activity.numpy()[0]
    central = connectome.central_cells_index[:]

    # Display calibration and direction tuning from moving ON/OFF edges.
    peak = np.zeros(len(types))
    tuning = {t: {} for t in types if t[:2] in ("T4", "T5")}
    for direction in ("right", "left", "up", "down"):
        for polarity in (1.0, 0.0):
            _, act = simulate(moving_edge(direction, polarity))
            delta = act - rest
            for k, t in enumerate(types):
                cells = type_index == k
                peak[k] = max(peak[k], np.percentile(np.abs(delta[:, cells]), 99.5))
                if t in tuning:
                    pol = "on" if polarity == 1.0 else "off"
                    mean_depol = np.maximum(delta[:, cells], 0).mean()
                    tuning[t][f"{direction}_{pol}"] = round(float(mean_depol), 6)
    for t, resp in tuning.items():
        pol = "on" if t.startswith("T4") else "off"
        dirs = {d: resp[f"{d}_{pol}"] for d in ("right", "left", "up", "down")}
        resp["preferred"] = max(dirs, key=dirs.get)
        print(f"{t}: prefers {resp['preferred']:5s} {dirs}")

    cell_types = [
        {
            "name": t,
            "group": layout[t],
            "n": int((type_index == k).sum()),
            "bias": float(bias[type_index == k][0]),
            "tau": float(tau[type_index == k][0]),
            "scale": round(float(max(peak[k], 1e-3)), 6),
            "central": int(central[k]),
        }
        for k, t in enumerate(types)
    ]

    network = {
        "meta": {
            "model": args.model,
            "checkpoint": "best",
            "flyvis_version": getattr(flyvis, "__version__", "unknown"),
            "exported": datetime.date.today().isoformat(),
            "source": "https://github.com/TuragaLab/flyvis",
            "paper": "Lappalainen et al., Connectome-constrained networks predict "
            "neural activity across the fly visual system. Nature (2024).",
            "license": "MIT, Copyright (c) 2023 Janne K. Lappalainen, Fabian D. "
            "Tschopp, Mason McGill, Jakob H. Macke, Srinivas C. Turaga",
        },
        "dt": DT,
        "extent": 15,
        "kernel_size": 13,
        "n_edges": n_edges,
        "input_types": [types.index(t) for t in connectome.input_cell_types[:].astype(str)],
        "cell_types": cell_types,
        "nodes": {
            "type": b64(type_index),
            "u": b64(connectome.nodes.u[:].astype(np.int8)),
            "v": b64(connectome.nodes.v[:].astype(np.int8)),
        },
        "filters": {
            "source": [f[0] for f in filters],
            "target": [f[1] for f in filters],
            "du": [f[2] for f in filters],
            "dv": [f[3] for f in filters],
            "weight": [float(np.float32(f[4])) for f in filters],
            # Average synapse count per target cell (weight = sign * count * strength).
            "count": [round(f[5], 3) for f in filters],
        },
        "direction_tuning": tuning,
    }
    out = ROOT / "site" / "data" / "network.json"
    out.write_text(json.dumps(network, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB, {len(filters)} filters)")

    hex_movie, act = simulate(fixture_movie())
    reference = {
        "about": "flyvis reference outputs for tests/sim.test.mjs",
        "model": args.model,
        "eye_image_expected": b64(eye(torch.from_numpy(procedural_image())[None, None])
                                  .numpy().astype(np.float32).ravel()),
        "stimulus": b64(hex_movie.numpy().astype(np.float32).ravel()),
        "n_frames": int(act.shape[0]),
        "rest": b64(rest.astype(np.float32)),
        "central_traces": b64(act[:, central].astype(np.float32).ravel()),
        "final_state": b64(act[-1].astype(np.float32)),
    }
    out = ROOT / "tests" / "fixtures" / "reference.json"
    out.write_text(json.dumps(reference, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
