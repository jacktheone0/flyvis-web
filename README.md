# Fly Brain, Live

**Live site: https://jacktheone0.github.io/flyvis-web/**

A model of the fruit fly's visual system, running entirely in your browser. Paste in an
email and it scrolls past the fly's eye, like a sign going by. You can watch all 45,669
simulated neurons respond, see which way the fly's motion detectors say the text is
moving, and get a "reply" from the fly that describes what its brain did.

The model is [flyvis](https://github.com/TuragaLab/flyvis), from Lappalainen et al.,
[*Connectome-constrained networks predict neural activity across the fly visual system*](https://www.nature.com/articles/s41586-024-07939-3)
(Nature, 2024). Its wiring comes from electron-microscopy reconstructions of the fly's
optic lobe. The network here is the pretrained model `flow/0000/000`, trained to detect
visual motion (optic flow).

## What it can and can't do

- It simulates the visual system only: 65 cell types, from photoreceptors to the T4/T5
  motion detectors. There is no memory, no language and no decision-making.
- The fly can't read. Its eye has 721 facets, far too coarse to resolve letters.
- The reply is a template filled in with measurements from the simulation: the direction
  of motion the T4/T5 cells reported, the moment of peak activity and the cell types that
  reacted most.
- Everything runs on your device. The email never leaves the page, and the site has no
  server.

## How it works

| Piece | What it does |
| --- | --- |
| `scripts/export_model.py` | Loads the pretrained flyvis network and writes `site/data/network.json`. Every synapse between two cell types at the same lattice offset shares one weight, so the full 1.5M-synapse network compresses to a ~2,400-row table (≈280 kB). The script checks that the table rebuilds the connectome edge for edge. |
| `site/js/sim.js` | A JavaScript port of the flyvis dynamics (leaky non-spiking neurons, graded synapses, Euler steps of 20 ms) and of its `BoxEye` renderer. |
| `site/js/worker.js` | Runs the network in a Web Worker at about 3 ms per 20 ms step, so it can keep up in real time. |
| `site/js/app.js` | Draws the email onto a strip that scrolls across the eye, renders every neuron, and writes the reply. |
| `tests/sim.test.mjs` | Compares the JS port with flyvis outputs saved in `tests/fixtures/reference.json`: eye sampling, resting state and 60 frames of full-network responses. They agree to about 1e-6. |

## Run locally

```sh
python3 -m http.server -d site 8000   # then open http://localhost:8000
node --test                           # check the simulator against flyvis
```

## Regenerate the model export

```sh
pip install flyvis
flyvis download-pretrained --skip_large_files
python scripts/export_model.py --model flow/0000/000
node --test
```

## Deployment

Each push to `main` runs the tests, then publishes `site/` to the `gh-pages` branch, which
GitHub Pages serves. If the site doesn't appear after the first deploy, go to
**Settings → Pages** and set **Source** to *Deploy from a branch*, branch `gh-pages`,
folder `/ (root)`.

## Credits

Model, pretrained weights and connectome data are from flyvis, © 2023 Janne K. Lappalainen,
Fabian D. Tschopp, Mason McGill, Jakob H. Macke and Srinivas C. Turaga, under the MIT
License (`site/data/LICENSE-flyvis.txt`). This project isn't affiliated with the authors.
