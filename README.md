# Fly Brain, Live

**Live site: https://jacktheone0.github.io/flyvis-web/**

A model of the fruit fly's visual system, running entirely in your browser, with two demos:

- **[Email demo](https://jacktheone0.github.io/flyvis-web/):** paste in an email and it
  scrolls past the fly's eye, like a sign going by. You can watch all 45,669 simulated
  neurons respond, see which way the fly's motion detectors say the text is moving, and get
  a "reply" from the fly that describes what its brain did.
- **[Flight simulator](https://jacktheone0.github.io/flyvis-web/flight.html):** wind keeps
  spinning the fly around a 360° world. Switch its brain on and it steers back using its
  motion detectors, the reflex real flies use to fly straight. A **Train** button fits a
  decoder on the fly's motion detectors in your browser and runs a test flight comparing
  brain off, a hand-made reflex and the trained decoder.

- **[PolyTrack mod](https://jacktheone0.github.io/flyvis-web/polytrack.html):** a
  [PolyModLoader](https://web.polymodloader.com/) mod for Kodub's PolyTrack. Add the mod URL
  `https://jacktheone0.github.io/flyvis-web/polytrack`. Drive a few laps while the fly
  watches, train a decoder on its output neurons, then let the fly drive, in slow motion
  (game time waits for the fly's brain). Leaderboards and multiplayer are disabled while it's
  loaded. Whether the fly learns to drive well is an open experiment.

The model is [flyvis](https://github.com/TuragaLab/flyvis), from Lappalainen et al.,
[*Connectome-constrained networks predict neural activity across the fly visual system*](https://www.nature.com/articles/s41586-024-07939-3)
(Nature, 2024). Its wiring comes from electron-microscopy reconstructions of the fly's
optic lobe. The network here is the pretrained model `flow/0000/000`, trained to detect
visual motion (optic flow).

## What it can and can't do

- It simulates the visual system only: 65 cell types, from photoreceptors to the T4/T5
  motion detectors. There is no memory, no language and no decision-making.
- The fly can't read. Its eye has 721 facets, far too coarse to resolve letters. The
  email reply is a template filled in with measurements from the simulation.
- In the flight simulator, the world, the wind and the steering rules are added around
  the model. Only turning (yaw) is simulated. Training fits 8 decoder weights on top of the
  motion detectors and never changes the published network.
- Typical flight results in breezy wind: brain off about 80°/s of spin, simple reflex
  about 43°/s, trained decoder about 38°/s. The decoder explains about 93% of the spin
  on worlds it never saw.
- Everything runs on your device. Nothing is sent to a server, and the site has none.

## How it works

| Piece | What it does |
| --- | --- |
| `scripts/export_model.py` | Loads the pretrained flyvis network and writes `site/data/network.json`. Every synapse between two cell types at the same lattice offset shares one weight, so the full 1.5M-synapse network compresses to a ~2,400-row table (≈280 kB). The script checks that the table rebuilds the connectome edge for edge, and measures each T4/T5 type's preferred direction. |
| `site/js/sim.js` | A JavaScript port of the flyvis dynamics (leaky non-spiking neurons, graded synapses, Euler steps of 20 ms) and of its `BoxEye` renderer. |
| `site/js/worker.js`, `app.js` | Email demo: the network runs in a Web Worker (about 3 ms per 20 ms step, so it keeps up in real time); the page draws the scrolling email, every neuron and the reply. |
| `site/js/world.js`, `flight-sim.js` | Flight simulator: the procedural 360° world and wind, the closed loop (wind + steering → eye → network → steering), and the decoder training (ridge regression on the 8 T4/T5 types' activity). |
| `site/js/flight-worker.js`, `flight.js` | Runs the flight and training in a worker; the page draws the world, the eye, a compass, the heading plot and the brain. |
| `site/js/brain-view.js`, `model.js`, `motion.js` | Shared by both pages: the cell-type panels and detail view, model loading, and the T4/T5 motion readouts. |
| `tests/sim.test.mjs` | Compares the JS port with flyvis outputs saved in `tests/fixtures/reference.json`: eye sampling, resting state and 60 frames of full-network responses. They agree to about 1e-6. |
| `site/polytrack/` | The PolyTrack mod: `manifest.json` and `0.1.0/main.mod.js` (panel, physics-clock patch, online-write blocks), `fly-driver.js` (frame capture → fly eye → network → keys; recording, decoder training), `autopilot.js` (a simple teacher). |
| `tests/polytrack.test.mjs` | Checks the mod's decoder training on synthetic data. |
| `tests/flight.test.mjs` | Checks that the fly senses which way the world turns, that the reflex and a trained decoder both reduce spin, and that worlds wrap seamlessly. |

## Run locally

```sh
python3 -m http.server -d site 8000   # then open http://localhost:8000
node --test                           # simulator accuracy and flight behaviour
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
