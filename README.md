# Waste Segregation Reinforcement Learning Summative

A custom Gymnasium environment simulating a single sorting station on a
municipal solid waste (MSW) conveyor line, trained with four reinforcement
learning algorithms -- DQN, REINFORCE, A2C, and PPO -- and compared under a
uniform hyperparameter-sweep and analysis pipeline.

![Facility render preview](assets/render_preview.png)

## Table of contents

- [Problem statement](#problem-statement)
- [Environment design](#environment-design)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Running locally](#running-locally)
- [Training on Google Colab](#training-on-google-colab)
- [Reproducing the analysis](#reproducing-the-analysis)
- [Testing](#testing)
- [Design notes and limitations](#design-notes-and-limitations)

## Problem statement

On every timestep, a new waste item arrives at a sorting station. An agent
observes a noisy sensor reading of the item's material composition
(organic, plastic, metal, glass, or non-recyclable contaminant) and must
decide what to do with it: sort it into one of four bins, reject it to
landfill, hold it for another step, or trigger a closer sensor scan to
reduce uncertainty at an energy cost.

This is deliberately **not** a disguised classification problem. Six
interacting mechanics give the environment genuine sequential structure, so
past decisions causally affect future reward and future action
availability:

1. **Bin capacity with soft value decay** -- a bin's value per correctly
   sorted item decreases as it approaches capacity, so a rational policy
   diverts material or plans ahead rather than greedily filling one bin.
2. **Contamination cascades** -- a bin's true value is only realized at
   periodic "ship-out" events and is discounted convexly by accumulated
   contamination. A string of small mistakes several steps earlier produces
   a delayed, amplified penalty -- the environment's primary source of
   temporal credit assignment.
3. **Sensor noise proportional to item ambiguity** -- items with no clearly
   dominant material are observed with more noise, forcing decisions under
   uncertainty.
4. **Conveyor time pressure** -- a superlinear penalty on consecutive
   hold/scan actions means indefinite hesitation is always worse than a
   fast, imperfect commitment.
5. **Equipment jamming caused by the agent's own history** -- a bin that has
   recently received excess contamination (beyond what is realistically
   unavoidable) becomes increasingly likely to jam and go offline, so the
   agent must anticipate and route around consequences of its own earlier
   decisions.
6. **A finite energy budget for close sensor scans** -- scanning is the only
   way to reduce observation noise, but it is a scarce resource that must be
   rationed across the episode.

See `environment/dynamics.py` for the full mathematical specification of
each mechanic and `environment/custom_env.py` for how they compose into the
step/reward function.

## Environment design

**Observation** (`Box(21,)`, `float32`, all components normalized to
`[0, 1]`):

| Indices | Meaning |
|---|---|
| `0:4`   | Bin fill fractions (organic, plastic, metal, glass) |
| `4:8`   | Bin contamination-so-far fractions (same bin order) |
| `8:12`  | Bin jam-cooldown remaining, normalized |
| `12`    | Energy remaining, normalized |
| `13`    | Consecutive non-committal (hold/scan) actions, normalized |
| `14:19` | Noisy observed item composition (organic, plastic, metal, glass, contaminant) |
| `19`    | Current item mass, normalized |
| `20`    | Steps remaining in episode, normalized |

**Action** (`Discrete(7)`): `sort_organic`, `sort_plastic`, `sort_metal`,
`sort_glass`, `reject`, `hold`, `scan_closely`.

**Reward**: dominated by value realized at periodic bin ship-out events
(discounted by accumulated contamination), with smaller immediate shaping
terms and penalties for overflow, jamming, forced rejection from stalling,
and wasting valuable material by rejecting it. A capacity-aware heuristic
policy scores roughly two orders of magnitude higher than a uniform-random
policy under this reward design (see `tests/test_environment.py`), which we
use as a basic sanity check that the reward function rewards genuine
competence rather than being arbitrary or trivially gameable.

Material composition ranges are grounded in published estimates of African
municipal solid waste composition (organic fraction roughly 30-61%
depending on city), and sensor noise ranges reflect reported optical/NIR
sorting-sensor accuracy (>95% on well-separated single-material items,
degrading toward noisy/unreliable on mixed or contaminated items). Bin base
values are illustrative relative weights (metal > plastic > organic >
glass) reflecting the well-known relative ordering of recyclate market
value; they are not claimed to be precise market prices.

## Repository structure

```
tawe_kelvin_rl_summative/
├── pyproject.toml              # dependencies and package metadata
├── README.md                   # this file
├── main.py                     # auto-selects and demos the best trained model
├── play.py                     # renders a specific trained agent, live or headless
│
├── environment/
│   ├── custom_env.py           # WasteSegregationEnv(gym.Env) -- step/reset/spaces
│   ├── dynamics.py             # the five mathematical dynamics models
│   └── rendering.py            # Pygame facility dashboard renderer
│
├── training/
│   ├── hyperparameters.py      # 10 presets per algorithm, single source of truth
│   ├── utils.py                # env factory, CSV episode logger, model loader
│   ├── dqn_training.py         # SB3 DQN training entry point
│   ├── reinforce_training.py   # from-scratch REINFORCE (SB3 has no REINFORCE)
│   ├── a2c_training.py         # SB3 A2C training entry point
│   └── ppo_training.py         # SB3 PPO training entry point
│
├── models/{dqn,reinforce,a2c,ppo}/   # saved model checkpoints per run
├── logs/{dqn,reinforce,a2c,ppo}/     # per-episode CSV logs + TensorBoard logs
│
├── notebooks/                  # one self-contained Colab notebook per algorithm
│   ├── 01_train_dqn.ipynb
│   ├── 02_train_reinforce.ipynb
│   ├── 03_train_a2c.ipynb
│   └── 04_train_ppo.ipynb
│
├── results/
│   ├── analysis.py             # builds hyperparameter tables + reward plots
│   ├── hyperparameter_tables/  # 4 required 10-row tables, as CSV
│   └── plots/                  # reward curves, algorithm comparison
│
├── assets/                     # report figures, rendering preview
└── tests/                      # pytest suite: dynamics + environment
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) as its canonical
dependency manager, with `pip` as a fully supported fallback (used by the
Colab notebooks, which do not have `uv` preinstalled).

### Option A: uv (recommended for local development)

```bash
uv sync
uv run pytest tests/ -v
```

### Option B: pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pytest tests/ -v
```

## Running locally

### Train a single algorithm, all ten hyperparameter presets

```bash
python -m training.dqn_training --run all
python -m training.reinforce_training --run all
python -m training.a2c_training --run all
python -m training.ppo_training --run all
```

Each command trains all ten presets for that algorithm sequentially, saving
a model checkpoint to `models/<algo>/<run_id>.{zip|pt}` and a per-episode
CSV log to `logs/<algo>/<run_id>.csv`.

### Train a single specific run

```bash
python -m training.ppo_training --run ppo_07
```

### Generate hyperparameter tables and reward plots

```bash
python -m results.analysis
```

Writes the four hyperparameter comparison tables to
`results/hyperparameter_tables/` and reward-curve plots (per algorithm, plus
a best-run-per-algorithm comparison) to `results/plots/`.

### Watch a trained agent

```bash
# Auto-select the best model across all algorithms (requires results/ to be populated)
python main.py

# Or specify an exact model
python play.py --algo ppo --model models/ppo/ppo_07.zip

# Headless (no Pygame window), useful over SSH / CI
python play.py --algo ppo --model models/ppo/ppo_07.zip --no-render
```

## Training on Google Colab

Each notebook in `notebooks/` is fully self-contained and follows an
identical six-cell pattern, so any one can be opened and run top to bottom
independently of the others:

1. **Clone + install** -- clones this repository and `pip install`s it.
2. **Imports** -- pulls in `WasteSegregationEnv` and the relevant algorithm.
3. **Hyperparameter presets** -- the same ten presets from
   `training/hyperparameters.py`, versioned in the notebook for
   reproducibility.
4. **Training loop** -- trains all ten presets, logging to
   `logs/<algo>/run_i.csv` and saving checkpoints to
   `models/<algo>/run_i.{zip|pt}`.
5. **Persist results** -- mounts Google Drive (or pushes back to the GitHub
   repository with a token) so results survive Colab session resets.
6. **Quick eval** -- loads the best of the ten runs and runs a short
   evaluation episode as a sanity check.

## Reproducing the analysis

After training (locally or via the notebooks, with results synced back into
`logs/` and `models/`):

```bash
python -m results.analysis
```

This regenerates:
- `results/hyperparameter_tables/{dqn,reinforce,a2c,ppo}_hyperparameters.csv`
  -- each preset's hyperparameters joined with its final training
  performance (mean reward over the last 10% of episodes, best episode
  reward, final episode reward).
- `results/plots/{dqn,reinforce,a2c,ppo}_reward_curves.png` -- reward
  learning curves for all ten runs of each algorithm.
- `results/plots/dqn_objective_curves.png` -- DQN TD-loss (objective) curves.
- `results/plots/{reinforce,a2c,ppo}_entropy_curves.png` -- policy-entropy
  (exploration) curves for the three policy-gradient methods.
- `results/plots/convergence_subplots.png` -- best-run reward curve per
  algorithm, as a 2x2 subplot grid.
- `results/plots/algorithm_comparison.png` -- the single best run from each
  algorithm, overlaid for direct comparison.

The extra diagnostics (DQN loss, policy entropy, value loss) are logged
per-episode by the training scripts into the same CSVs, so they survive the
Colab -> zip -> local round-trip and no TensorBoard parsing is needed.

### Generalization test

```bash
python -m results.generalization
```

Loads the best model per algorithm and evaluates it under demand profiles it
never trained on (`high_contam`, `plastic_surge`, `uniform`), writing
`results/hyperparameter_tables/generalization_results.csv` and
`results/plots/generalization.png`.

### Serving the environment as an API (web/mobile integration)

The environment serializes its full state to JSON via
`WasteSegregationEnv.to_json()`, and `serve.py` exposes it over a FastAPI REST
service (reset/step/predict), so a browser or mobile app can render and drive
the facility with no Python or Pygame dependency on the client. A Three.js/WebGL
dashboard is included at `web/index.html` and served from the root URL:

**One command (recommended)** — start the backend and open the 3D frontend
together. Because the frontend uses same-origin relative API calls, a single
uvicorn process serves both; `run.py` launches it and auto-opens the dashboard
once the server is healthy:

```bash
uv run --extra serve python run.py
# -> http://127.0.0.1:8000/ opens automatically (dashboard)
# -> http://127.0.0.1:8000/docs for the interactive API explorer
# Ctrl+C to stop. Flags: --port <n>, --no-browser, --reload
```

**Manual equivalent** (if you prefer to drive uvicorn yourself):

```bash
uv sync --extra serve
uv run uvicorn serve:app --reload
# open http://127.0.0.1:8000 for the 3D dashboard, or /docs for the API explorer
```

## Testing

```bash
pytest tests/ -v
```

The suite covers:
- `tests/test_dynamics.py` -- unit tests for each of the five mathematical
  dynamics models in isolation (monotonicity, boundary behavior, convexity).
- `tests/test_environment.py` -- integration tests for the full
  `WasteSegregationEnv`: observation/action space validity, reset
  reproducibility, bin capacity and overflow, jamming behavior, ship-out
  reward realization, reject logic, and an end-to-end sanity check that a
  capacity-aware heuristic policy substantially outperforms a random one.

## Design notes and limitations

- **Why value is realized at ship-out, not per-item.** Giving full reward
  immediately on sorting would make this closer to a repeated single-step
  classification task. Delaying most of the reward to periodic bin
  ship-outs (discounted by accumulated contamination) means the
  consequences of a sort several steps earlier are only felt later --
  requiring the agent to reason about accumulated state, not just the
  current item.
- **Why jam risk is based on *excess* contamination, not raw
  contamination.** Real single-stream recyclables are rarely 100% pure;
  penalizing all contamination equally would make bins jam even under a
  correct, well-run policy. Jam risk is instead driven by contamination
  *above* a realistic baseline, so it reflects genuinely poor decisions
  rather than ordinary, expected impurity.
- **Single active item, not a queue.** For tractability, the environment
  models one item at a time rather than a queue of simultaneously visible
  items. Multi-item prioritization is a natural extension (see "Future
  work" below) but was scoped out to keep the six core mechanics
  implementable and debuggable within the project timeline.
- **Illustrative economics.** Bin base values reflect the correct *relative
  ordering* of recyclate value (metal > plastic > organic > glass) but are
  not calibrated to real market prices, which fluctuate by region and time.
- **Future work.** Natural extensions include: multiple simultaneously
  visible items requiring prioritization, composite/mixed items requiring a
  pre-sort "clean" action, seasonal/weather-driven contamination-detection
  noise, and downstream demand fluctuation (bin value changing episode to
  episode) as a generalization test -- train under one demand profile, test
  under another.
