"""
generate_notebooks.py

One-off script that programmatically builds the four Colab training
notebooks (notebooks/01_train_dqn.ipynb .. 04_train_ppo.ipynb) so that their
content stays byte-for-byte consistent with training/hyperparameters.py and
the six-cell pattern documented in the README. Not part of the runtime
package; re-run this script if the hyperparameter presets change.

Usage:
    python generate_notebooks.py
"""

from __future__ import annotations

import json
import os

import nbformat as nbf

# Point at the actual repository so the Colab clone works out of the box.
# NOTE: the assignment asks for a repo named "<student_name>_rl_summative"
# (i.e. tawe_kelvin_rl_summative). If/when the GitHub repo is renamed to that,
# update REPO_URL/REPO_DIR here and re-run `python generate_notebooks.py`.
REPO_URL = "https://github.com/kelvintawe12/tawe_kelvin_rl_summative.git"
REPO_DIR = "tawe_kelvin_rl_summative"

ALGO_CONFIG = {
    "dqn": {
        "title": "DQN",
        "notebook_name": "01_train_dqn.ipynb",
        "module": "training.dqn_training",
        "preset_var": "DQN_PRESETS",
        "model_ext": "zip",
        "sb3_class": "DQN",
    },
    "reinforce": {
        "title": "REINFORCE",
        "notebook_name": "02_train_reinforce.ipynb",
        "module": "training.reinforce_training",
        "preset_var": "REINFORCE_PRESETS",
        "model_ext": "pt",
        "sb3_class": None,
    },
    "a2c": {
        "title": "A2C",
        "notebook_name": "03_train_a2c.ipynb",
        "module": "training.a2c_training",
        "preset_var": "A2C_PRESETS",
        "model_ext": "zip",
        "sb3_class": "A2C",
    },
    "ppo": {
        "title": "PPO",
        "notebook_name": "04_train_ppo.ipynb",
        "module": "training.ppo_training",
        "preset_var": "PPO_PRESETS",
        "model_ext": "zip",
        "sb3_class": "PPO",
    },
}


def load_presets_source(algo: str) -> str:
    """Extract the literal Python source of one PRESETS list from
    training/hyperparameters.py, so the notebook's hardcoded copy is
    generated from (and stays consistent with) the single source of truth,
    rather than retyped by hand."""
    import training.hyperparameters as hp

    var_name = ALGO_CONFIG[algo]["preset_var"]
    presets = getattr(hp, var_name)
    return f"{var_name} = " + json.dumps(presets, indent=4)


def build_notebook(algo: str) -> nbf.NotebookNode:
    cfg = ALGO_CONFIG[algo]
    nb = nbf.v4.new_notebook()
    cells = []

    # --- Cell 0: title / intro (markdown) ---
    cells.append(nbf.v4.new_markdown_cell(
        f"# Train {cfg['title']} on WasteSegregationEnv\n\n"
        "This notebook is fully self-contained: run every cell top to bottom "
        "and it will clone the repository, install dependencies, train all "
        f"ten {cfg['title']} hyperparameter presets, and save models/logs "
        "back into the repository structure.\n\n"
        "See `README.md` in the repository root for the full project "
        "writeup, environment design, and the six-mechanic reward "
        "specification."
    ))

    # --- Cell 1: clone + install ---
    cells.append(nbf.v4.new_markdown_cell("## 1. Clone the repository and install dependencies"))
    cells.append(nbf.v4.new_code_cell(
        f"!git clone {REPO_URL}\n"
        f"%cd {REPO_DIR}\n"
        "!pip install -q -e .\n"
        "!pip install -q tensorboard nbformat"
    ))

    # --- Cell 2: imports ---
    cells.append(nbf.v4.new_markdown_cell("## 2. Imports"))
    if cfg["sb3_class"] is not None:
        import_lines = (
            f"from stable_baselines3 import {cfg['sb3_class']}\n"
            "from stable_baselines3.common.monitor import Monitor\n"
            "from environment.custom_env import WasteSegregationEnv\n"
            f"from {cfg['module']} import train_single_run\n"
            "import os, pandas as pd"
        )
    else:
        import_lines = (
            "from environment.custom_env import WasteSegregationEnv\n"
            f"from {cfg['module']} import ReinforceAgent, train_single_run\n"
            "import os, pandas as pd, torch"
        )
    cells.append(nbf.v4.new_code_cell(import_lines))

    # --- Cell 3: hyperparameter presets ---
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Hyperparameter presets\n\n"
        f"The ten {cfg['title']} configurations required for the "
        "hyperparameter-sweep deliverable, generated from "
        "`training/hyperparameters.py` so this notebook stays in sync with "
        "the canonical source of truth. Each preset is fully versioned here "
        "for reproducibility."
    ))
    cells.append(nbf.v4.new_code_cell(load_presets_source(algo)))

    # --- Cell 4: training loop ---
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Train all ten runs\n\n"
        f"Trains each preset in sequence, saving a model checkpoint to "
        f"`models/{algo}/<run_id>.{cfg['model_ext']}` and a per-episode "
        f"reward CSV log to `logs/{algo}/<run_id>.csv`."
    ))
    cells.append(nbf.v4.new_code_cell(
        f"log_dir = \"logs/{algo}\"\n"
        f"model_dir = \"models/{algo}\"\n"
        f"os.makedirs(log_dir, exist_ok=True)\n"
        f"os.makedirs(model_dir, exist_ok=True)\n\n"
        f"saved_paths = []\n"
        f"for preset in {cfg['preset_var']}:\n"
        f"    print(f\"Training {{preset['run_id']}} ...\")\n"
        f"    path = train_single_run(preset, log_dir, model_dir, seed=0)\n"
        f"    saved_paths.append(path)\n"
        f"    print(f\"  saved -> {{path}}\")\n\n"
        f"print(\"\\nAll {cfg['title']} runs complete.\")"
    ))

    # --- Cell 4b: visualize training ---
    cells.append(nbf.v4.new_markdown_cell(
        f"## 5. Visualize training\n\n"
        f"A 4-panel dashboard across all ten {cfg['title']} runs (raw reward, "
        f"smoothed reward, the {cfg['title']} diagnostic curve, and episode "
        f"length), followed by the hyperparameter + performance table with the "
        f"best run highlighted. All plotting lives in "
        f"`results/notebook_viz.py` so it stays consistent across notebooks."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from results.notebook_viz import training_dashboard, show_hyperparameter_table\n\n"
        f"training_dashboard('{algo}', save_to='results/plots/{algo}_dashboard.png')\n"
        f"table = show_hyperparameter_table('{algo}')\n"
        "table"
    ))

    # --- Cell 5: persist results ---
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Persist results\n\n"
        "Colab sessions are ephemeral, so models and logs must be persisted "
        "somewhere durable. Two options are provided below -- use whichever "
        "fits your workflow. **Only run one of the two cells.**"
    ))
    cells.append(nbf.v4.new_code_cell(
        "# Option A: mount Google Drive and copy results there\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "!mkdir -p /content/drive/MyDrive/tawe_kelvin_rl_summative_results\n"
        f"!cp -r {{log_dir}} /content/drive/MyDrive/tawe_kelvin_rl_summative_results/\n"
        f"!cp -r {{model_dir}} /content/drive/MyDrive/tawe_kelvin_rl_summative_results/\n"
        "print(\"Copied logs/ and models/ to Google Drive.\")"
    ))
    cells.append(nbf.v4.new_code_cell(
        "# Option B: commit and push results directly back to the GitHub repository\n"
        "# Requires a GitHub personal access token with repo write access.\n"
        "GITHUB_TOKEN = \"\"  # paste your token here (do not commit this notebook with a real token)\n"
        "if GITHUB_TOKEN:\n"
        f"    !git add {{log_dir}} {{model_dir}}\n"
        f"    !git commit -m \"Add {cfg['title']} training results\"\n"
        f"    !git push https://{{GITHUB_TOKEN}}@github.com/kelvintawe12/{REPO_DIR}.git HEAD:main\n"
        "else:\n"
        "    print(\"Skipped: no GitHub token provided.\")"
    ))

    # --- Cell 6: quick eval ---
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Quick evaluation\n\n"
        "Loads the best of the ten runs (by mean reward over the last 10% "
        "of episodes) and runs one evaluation episode as a fast sanity "
        "check before moving to the full analysis pipeline locally "
        "(`python -m results.analysis`)."
    ))
    if cfg["sb3_class"] is not None:
        eval_code = (
            "best_run_id, best_mean = None, -float('inf')\n"
            f"for preset in {cfg['preset_var']}:\n"
            f"    csv_path = os.path.join(log_dir, preset['run_id'] + '.csv')\n"
            "    if not os.path.exists(csv_path):\n"
            "        continue\n"
            "    df = pd.read_csv(csv_path)\n"
            "    tail = df.tail(max(1, len(df) // 10))\n"
            "    mean_r = tail['episode_reward'].mean()\n"
            "    if mean_r > best_mean:\n"
            "        best_mean, best_run_id = mean_r, preset['run_id']\n\n"
            "print(f'Best run: {best_run_id} (mean reward last 10%: {best_mean:.2f})')\n\n"
            f"best_model_path = os.path.join(model_dir, best_run_id + '.zip')\n"
            f"model = {cfg['sb3_class']}.load(best_model_path)\n\n"
            "eval_env = WasteSegregationEnv(seed=999)\n"
            "obs, info = eval_env.reset(seed=999)\n"
            "done, total_reward = False, 0.0\n"
            "while not done:\n"
            "    action, _ = model.predict(obs, deterministic=True)\n"
            "    obs, reward, terminated, truncated, info = eval_env.step(int(action))\n"
            "    total_reward += reward\n"
            "    done = terminated or truncated\n\n"
            "print(f'Evaluation episode reward: {total_reward:.2f}')\n"
            "print('Episode stats:', info['episode_stats'])"
        )
    else:
        eval_code = (
            "best_run_id, best_mean = None, -float('inf')\n"
            f"for preset in {cfg['preset_var']}:\n"
            f"    csv_path = os.path.join(log_dir, preset['run_id'] + '.csv')\n"
            "    if not os.path.exists(csv_path):\n"
            "        continue\n"
            "    df = pd.read_csv(csv_path)\n"
            "    tail = df.tail(max(1, len(df) // 10))\n"
            "    mean_r = tail['episode_reward'].mean()\n"
            "    if mean_r > best_mean:\n"
            "        best_mean, best_run_id = mean_r, preset['run_id']\n\n"
            "print(f'Best run: {best_run_id} (mean reward last 10%: {best_mean:.2f})')\n\n"
            "eval_env = WasteSegregationEnv(seed=999)\n"
            "agent = ReinforceAgent(\n"
            "    obs_dim=eval_env.observation_space.shape[0],\n"
            "    n_actions=eval_env.action_space.n,\n"
            ")\n"
            f"best_model_path = os.path.join(model_dir, best_run_id + '.pt')\n"
            "agent.load(best_model_path)\n\n"
            "obs, info = eval_env.reset(seed=999)\n"
            "done, total_reward = False, 0.0\n"
            "while not done:\n"
            "    action, _ = agent.predict(obs, deterministic=True)\n"
            "    obs, reward, terminated, truncated, info = eval_env.step(int(action))\n"
            "    total_reward += reward\n"
            "    done = terminated or truncated\n\n"
            "print(f'Evaluation episode reward: {total_reward:.2f}')\n"
            "print('Episode stats:', info['episode_stats'])"
        )
    cells.append(nbf.v4.new_code_cell(eval_code))

    # --- Cell 8: bundle all results into a downloadable zip ---
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Compile everything into a downloadable zip\n\n"
        "Regenerates the full analysis (hyperparameter tables + every plot: "
        "reward curves, DQN objective / policy-entropy curves, convergence "
        "subplots, best-run comparison) and packages `logs/`, `models/`, and "
        "`results/` into a single archive. In Colab this triggers a browser "
        "download so you can drop the results straight into the local repo "
        "(and into the report) after the session ends."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from results.notebook_viz import bundle_results\n\n"
        f"bundle_results(out_path='{algo}_results_bundle.zip')"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "name": cfg["notebook_name"]},
    }
    return nb


def main() -> None:
    out_dir = "notebooks"
    os.makedirs(out_dir, exist_ok=True)
    for algo, cfg in ALGO_CONFIG.items():
        nb = build_notebook(algo)
        out_path = os.path.join(out_dir, cfg["notebook_name"])
        with open(out_path, "w") as f:
            nbf.write(nb, f)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
