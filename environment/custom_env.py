"""
custom_env.py

WasteSegregationEnv: a custom Gymnasium environment simulating a single
sorting station on a municipal solid waste (MSW) conveyor line.

Design summary
---------------
On every timestep a new waste item arrives at the sorting station. The agent
observes a *noisy* estimate of the item's material composition (organic,
plastic, metal, glass, contaminant) and must choose one of seven actions:
sort into one of four bins, reject the item to landfill, hold it for another
step, or trigger a closer sensor scan that reduces observation noise at an
energy cost.

The environment is deliberately built to avoid degenerating into a one-shot
classification task. Six interacting mechanics give it genuine sequential
structure, meaning past actions causally affect future reward and future
action availability:

    1. Bin capacity with soft value decay (`dynamics.value_multiplier`) -
       sorting into a nearly-full bin is worth less, encouraging the agent
       to plan ahead rather than greedily pick the "correct" bin.
    2. Contamination cascades (`dynamics.contamination_penalty_factor`) -
       a bin's true value is only realized at periodic ship-out events, and
       is discounted convexly by accumulated contamination, so a string of
       small mistakes several steps earlier produces a delayed, amplified
       penalty. This is the environment's main source of temporal credit
       assignment.
    3. Sensor noise proportional to item ambiguity
       (`dynamics.sensor_noise_sigma`) - forces decisions under uncertainty.
    4. Conveyor time pressure via a superlinear stalling penalty
       (`dynamics.stall_penalty`) - indefinite hesitation is always worse
       than a fast, imperfect commitment.
    5. Equipment jamming caused by the agent's own sorting history
       (`dynamics.jam_probability`) - bins that have recently received a lot
       of contaminant mass become increasingly likely to jam and go
       offline, so the agent must anticipate and route around consequences
       of its own earlier decisions.
    6. A finite energy budget for close sensor scans - scanning is the only
       way to reduce observation noise, but it is a scarce resource that
       must be rationed across the episode.

Reward is intentionally NOT a simple per-item classification signal. Most of
the reward mass for a correct sort is only released when a bin is "shipped
out" (periodically, or at episode end), which is what makes this a genuine
Markov Decision Process rather than a repeated single-step classification
problem.

Material and composition ranges are grounded in published estimates of
African municipal solid waste composition (organic fraction roughly
30-61% depending on city) and in reported optical/NIR sorting-sensor
accuracy (>95% on well-separated single-material items, degrading toward
noisy/unreliable on mixed or contaminated items). Bin base values are
illustrative relative weights (metal > plastic > organic > glass), chosen to
reflect the well-known relative ordering of recyclate market value; they are
not claimed to be precise market prices.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.dynamics import (
    DynamicsConfig,
    RandomizationConfig,
    RewardConfig,
    value_multiplier,
    sensor_noise_sigma,
    contamination_penalty_factor,
    stall_penalty,
    jam_probability,
    update_ewma,
)

# Material categories. The first four each have a dedicated bin; the fifth
# ("contaminant") represents non-recyclable / unsortable material and has no
# bin of its own -- it should be rejected.
MATERIALS = ["organic", "plastic", "metal", "glass", "contaminant"]
BINS = ["organic", "plastic", "metal", "glass"]
N_BINS = len(BINS)

# Relative base value per unit mass when correctly sorted and fully realized
# (i.e. zero contamination, bin not near capacity). Illustrative ordering
# reflecting real recyclate market value (metal >> plastic > organic > glass).
BASE_VALUES = {"organic": 18.0, "plastic": 28.0, "metal": 55.0, "glass": 14.0}

# Action indices.
ACTION_SORT_ORGANIC = 0
ACTION_SORT_PLASTIC = 1
ACTION_SORT_METAL = 2
ACTION_SORT_GLASS = 3
ACTION_REJECT = 4
ACTION_HOLD = 5
ACTION_SCAN = 6
N_ACTIONS = 7

ACTION_NAMES = [
    "sort_organic", "sort_plastic", "sort_metal", "sort_glass",
    "reject", "hold", "scan_closely",
]

_SORT_ACTION_TO_BIN = {
    ACTION_SORT_ORGANIC: "organic",
    ACTION_SORT_PLASTIC: "plastic",
    ACTION_SORT_METAL: "metal",
    ACTION_SORT_GLASS: "glass",
}


class WasteSegregationEnv(gym.Env):
    """A custom Gymnasium environment for RL-based waste segregation.

    Observation (Box(21 + 6*queue_lookahead,), float32, normalized to [0, 1]):
        [0:4]   bin fill fractions (organic, plastic, metal, glass)
        [4:8]   bin contamination-so-far fractions (same bin order)
        [8:12]  bin jam-cooldown remaining, normalized
        [12]    energy remaining, normalized
        [13]    consecutive non-committal (hold/scan) actions, normalized
        [14:19] noisy observed item composition (organic, plastic, metal,
                glass, contaminant), approximately sums to 1
        [19]    current item mass, normalized
        [20]    steps remaining in episode, normalized
        [21:]   (only if queue_lookahead > 0) for each of the next
                `queue_lookahead` upcoming items: its noisy observed
                composition (5) and normalized mass (1) -- a conveyor buffer
                that lets the agent plan ahead instead of reacting item by item.

    Action (Discrete(7)): see ACTION_NAMES / module-level ACTION_* constants.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        max_steps: int = 200,
        bin_capacity: float = 50.0,
        ship_out_interval: int = 40,
        energy_budget: float = 60.0,
        scan_energy_cost: float = 3.0,
        item_mass_range: tuple[float, float] = (0.5, 3.0),
        category_probs: Optional[np.ndarray] = None,
        dynamics_config: Optional[DynamicsConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        queue_lookahead: int = 0,
        domain_randomize: bool = False,
        randomization_config: Optional[RandomizationConfig] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert max_steps >= 1, "max_steps must be >= 1"
        assert bin_capacity > 0.0, "bin_capacity must be positive"
        assert ship_out_interval >= 1, "ship_out_interval must be >= 1"
        assert energy_budget >= 0.0, "energy_budget must be non-negative"
        assert queue_lookahead >= 0, "queue_lookahead must be >= 0"

        self.max_steps = max_steps
        self.bin_capacity = bin_capacity
        self.ship_out_interval = ship_out_interval
        self.energy_budget_max = energy_budget
        self.scan_energy_cost = scan_energy_cost
        self.item_mass_range = item_mass_range
        self.cfg = dynamics_config if dynamics_config is not None else DynamicsConfig()
        self.rcfg = reward_config if reward_config is not None else RewardConfig()
        self.queue_lookahead = queue_lookahead
        self.domain_randomize = domain_randomize
        self.rand_cfg = (
            randomization_config if randomization_config is not None else RandomizationConfig()
        )
        self.render_mode = render_mode

        # Per-episode active dynamics parameters. With domain randomization off
        # these stay equal to the config defaults; with it on they are
        # resampled each reset() (see _sample_episode_regime).
        self._active_sigma_min = self.cfg.sigma_min
        self._active_sigma_max = self.cfg.sigma_max
        self._active_jam_baseline = self.cfg.jam_baseline_contamination

        self.action_space = spaces.Discrete(N_ACTIONS)
        obs_dim = 21 + 6 * queue_lookahead
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Item generation is two-stage: first a "dominant material" category
        # is drawn (giving the mix of item types on the line, calibrated so
        # organic dominates on average, ~30-61% of MSW by mass in African
        # city studies), then a Dirichlet distribution concentrated on that
        # category produces the actual composition vector (giving realistic
        # per-item purity, since real single-stream items are usually mostly
        # one material but rarely 100% pure). This decouples "what kind of
        # item is this" from "how ambiguous/contaminated is it", both of
        # which vary independently on a real sorting line.
        # Demand profile: probability that each material category is the
        # item's dominant one. Overridable so a generalization test can train
        # on one profile and evaluate on a shifted one (see
        # results/generalization.py). Default is calibrated so organic
        # dominates (~30-61% of MSW by mass in African city studies).
        if category_probs is None:
            self._base_category_probs = np.array([0.42, 0.20, 0.10, 0.10, 0.18])
        else:
            cp = np.asarray(category_probs, dtype=np.float64)
            assert cp.shape == (5,), "category_probs must have shape (5,)"
            self._base_category_probs = cp / cp.sum()
        # Active demand profile for the current episode. Equal to the base
        # profile unless domain randomization resamples it each reset().
        self._category_probs = self._base_category_probs.copy()
        self._dominant_alpha = 5.0
        self._other_alpha = 0.8

        self._rng = np.random.default_rng(seed)
        self._pygame_renderer = None  # lazily constructed, see rendering.py

        # Episode state, populated in reset().
        self.bin_fill = np.zeros(N_BINS, dtype=np.float64)
        self.bin_raw_value = np.zeros(N_BINS, dtype=np.float64)
        self.bin_contam_mass = np.zeros(N_BINS, dtype=np.float64)
        self.bin_total_mass = np.zeros(N_BINS, dtype=np.float64)
        self.bin_jam_cooldown = np.zeros(N_BINS, dtype=np.int32)
        self.bin_contam_ewma = np.zeros(N_BINS, dtype=np.float64)
        self.energy = self.energy_budget_max
        self.consecutive_holds = 0
        self.step_count = 0
        self.current_item_true = np.zeros(5, dtype=np.float64)
        self.current_item_obs = np.zeros(5, dtype=np.float64)
        self.current_item_mass = 0.0
        # Conveyor lookahead buffer: the next `queue_lookahead` items, each a
        # dict with true/observed composition and mass. Empty when disabled.
        self._item_queue: list[dict] = []
        self.last_action_name = ""
        self.last_reward_breakdown: dict[str, float] = {}
        self.total_reward = 0.0
        self.episode_stats = {"correct_sorts": 0, "incorrect_sorts": 0,
                               "jams_triggered": 0, "shipouts": 0}

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.bin_fill[:] = 0.0
        self.bin_raw_value[:] = 0.0
        self.bin_contam_mass[:] = 0.0
        self.bin_total_mass[:] = 0.0
        self.bin_jam_cooldown[:] = 0
        self.bin_contam_ewma[:] = 0.0
        self.energy = self.energy_budget_max
        self.consecutive_holds = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.last_action_name = ""
        self.last_reward_breakdown = {}
        self.episode_stats = {"correct_sorts": 0, "incorrect_sorts": 0,
                               "jams_triggered": 0, "shipouts": 0}

        # `options={"domain_randomize": bool}` lets a caller flip randomization
        # per-reset (e.g. randomized training resets, fixed evaluation resets)
        # without constructing a new env.
        randomize = self.domain_randomize
        if options is not None and "domain_randomize" in options:
            randomize = bool(options["domain_randomize"])
        self._sample_episode_regime(randomize)

        # Prime the conveyor: the current item plus the lookahead buffer.
        self._item_queue = [self._generate_item() for _ in range(self.queue_lookahead)]
        self._spawn_item()
        obs = self._get_observation()
        return obs, self._get_info()

    def _sample_episode_regime(self, randomize: bool) -> None:
        """Set the episode's active dynamics parameters.

        With `randomize` False, restore the fixed config defaults. With it
        True, sample a fresh demand profile, sensor-noise scale, and jam
        baseline from `RandomizationConfig` using the seeded RNG (so a given
        reset seed still reproduces the same randomized episode exactly).
        """
        rc = self.rand_cfg
        if not randomize:
            self._category_probs = self._base_category_probs.copy()
            self._active_sigma_min = self.cfg.sigma_min
            self._active_sigma_max = self.cfg.sigma_max
            self._active_jam_baseline = self.cfg.jam_baseline_contamination
            return

        if rc.randomize_demand:
            alpha = self._base_category_probs * rc.demand_concentration
            self._category_probs = self._rng.dirichlet(alpha)
        else:
            self._category_probs = self._base_category_probs.copy()

        if rc.randomize_sensor:
            scale = self._rng.uniform(*rc.sensor_sigma_scale_range)
            self._active_sigma_min = self.cfg.sigma_min * scale
            self._active_sigma_max = self.cfg.sigma_max * scale
        else:
            self._active_sigma_min = self.cfg.sigma_min
            self._active_sigma_max = self.cfg.sigma_max

        if rc.randomize_jam:
            self._active_jam_baseline = float(self._rng.uniform(*rc.jam_baseline_range))
        else:
            self._active_jam_baseline = self.cfg.jam_baseline_contamination

    def action_masks(self) -> np.ndarray:
        """Boolean validity mask over the action space (shape ``(7,)``).

        A jammed bin physically cannot accept material this step, so its
        corresponding ``sort_*`` action is masked out (invalid). The
        non-committal / diversion actions (``reject``, ``hold``,
        ``scan_closely``) are always valid, so the mask is never all-False and
        the agent always has a legal move.

        This is the method name expected by ``sb3_contrib.MaskablePPO`` and by
        the ``ActionMasker`` wrapper, so the environment can be dropped into a
        maskable-policy training loop unchanged. The same array is also
        surfaced in ``info["action_mask"]`` and ``to_json()`` so a
        rule-based/greedy controller or a web frontend can consume it too.
        """
        mask = np.ones(N_ACTIONS, dtype=bool)
        for action, bin_name in _SORT_ACTION_TO_BIN.items():
            if self.bin_jam_cooldown[BINS.index(bin_name)] > 0:
                mask[action] = False
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"invalid action {action}"
        reward = 0.0
        breakdown: dict[str, float] = {}
        terminated = False

        # Bins that jammed on a previous step count down toward availability.
        self._decrement_jam_cooldowns()

        if action in _SORT_ACTION_TO_BIN:
            r, bd = self._do_sort(_SORT_ACTION_TO_BIN[action])
            reward += r
            breakdown.update(bd)
            self.consecutive_holds = 0
            self._spawn_item()
        elif action == ACTION_REJECT:
            r, bd = self._do_reject()
            reward += r
            breakdown.update(bd)
            self.consecutive_holds = 0
            self._spawn_item()
        elif action == ACTION_HOLD:
            r, bd = self._do_hold()
            reward += r
            breakdown.update(bd)
        elif action == ACTION_SCAN:
            r, bd = self._do_scan()
            reward += r
            breakdown.update(bd)
        else:  # pragma: no cover - guarded by assert above
            raise ValueError(f"Unhandled action {action}")

        self.last_action_name = ACTION_NAMES[action]
        self.step_count += 1

        # Periodic ship-out: realize accumulated bin value, discounted by
        # accumulated contamination, and empty the bin.
        if self.step_count % self.ship_out_interval == 0:
            r_ship, bd_ship = self._ship_out_all_bins()
            reward += r_ship
            breakdown.update(bd_ship)

        # Facility shutdown: every bin simultaneously jammed.
        if np.all(self.bin_jam_cooldown > 0):
            reward -= self.rcfg.shutdown_penalty
            breakdown["shutdown_penalty"] = -self.rcfg.shutdown_penalty
            terminated = True

        truncated = self.step_count >= self.max_steps
        if (terminated or truncated) and not terminated:
            # Final ship-out so unrealized value at episode end is not lost.
            r_ship, bd_ship = self._ship_out_all_bins()
            reward += r_ship
            for k, v in bd_ship.items():
                breakdown[k] = breakdown.get(k, 0.0) + v

        self.total_reward += reward
        self.last_reward_breakdown = breakdown
        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    def _do_sort(self, bin_name: str) -> tuple[float, dict]:
        b = BINS.index(bin_name)
        breakdown: dict[str, float] = {}

        if self.bin_jam_cooldown[b] > 0:
            # Jammed bins reject any material routed to them.
            self.episode_stats["incorrect_sorts"] += 1
            penalty = -self.rcfg.jam_failure_penalty
            breakdown["jam_failure_penalty"] = penalty
            return penalty, breakdown

        mass = self.current_item_mass
        fill_before = self.bin_fill[b] / self.bin_capacity
        remaining_capacity = max(self.bin_capacity - self.bin_fill[b], 0.0)
        accepted_mass = min(mass, remaining_capacity)
        overflow_mass = mass - accepted_mass

        reward = 0.0
        if overflow_mass > 1e-9:
            overflow_penalty = -self.rcfg.overflow_coef * overflow_mass
            reward += overflow_penalty
            breakdown["overflow_penalty"] = overflow_penalty

        if accepted_mass > 1e-9:
            correct_fraction = float(self.current_item_true[b])
            correct_mass = accepted_mass * correct_fraction
            contaminant_mass = accepted_mass - correct_mass

            vmult = value_multiplier(
                fill_before,
                theta=self.cfg.decay_start_fraction,
                k=self.cfg.decay_strength,
                floor=self.cfg.min_value_multiplier,
            )
            value_added = BASE_VALUES[bin_name] * correct_mass * vmult
            self.bin_raw_value[b] += value_added
            self.bin_fill[b] += accepted_mass
            self.bin_contam_mass[b] += contaminant_mass
            self.bin_total_mass[b] += accepted_mass

            # Immediate shaping signal; the dominant reward for a good sort is
            # still realized later, at ship-out, but this term is sized so a
            # correct sort of a valuable item beats the safe reject reward
            # even before ship-out (removing the "reject everything" trap).
            shaping = self.rcfg.sort_shaping_coef * value_added
            reward += shaping
            breakdown["sort_shaping_reward"] = shaping

            if correct_fraction >= 0.5:
                self.episode_stats["correct_sorts"] += 1
            else:
                self.episode_stats["incorrect_sorts"] += 1

            # Equipment jamming: driven by an EWMA of *per-item contamination
            # fraction* routed into this bin, so repeated bad decisions
            # compound while routine, expected impurity does not.
            item_contam_fraction = contaminant_mass / accepted_mass if accepted_mass > 1e-9 else 0.0
            self.bin_contam_ewma[b] = update_ewma(
                self.bin_contam_ewma[b], item_contam_fraction, alpha=self.cfg.jam_ewma_alpha
            )
            p_jam = jam_probability(
                self.bin_contam_ewma[b],
                baseline=self._active_jam_baseline,
                kappa=self.cfg.jam_kappa,
            )
            if self._rng.random() < p_jam:
                self.bin_jam_cooldown[b] = self.cfg.jam_cooldown_steps
                self.episode_stats["jams_triggered"] += 1
                reward -= self.rcfg.jam_triggered_penalty
                breakdown["jam_triggered_penalty"] = -self.rcfg.jam_triggered_penalty

        return reward, breakdown

    def _do_reject(self) -> tuple[float, dict]:
        contaminant_frac = float(self.current_item_true[4])
        valuable_frac = float(np.sum(self.current_item_true[:4]))
        breakdown: dict[str, float] = {}
        if contaminant_frac >= 0.5:
            reward = self.rcfg.reject_correct_reward
            breakdown["correct_reject_reward"] = reward
        else:
            reward = -self.rcfg.reject_wasted_coef * self.current_item_mass * valuable_frac
            breakdown["wasted_material_penalty"] = reward
        return reward, breakdown

    def _do_hold(self) -> tuple[float, dict]:
        self.consecutive_holds += 1
        breakdown: dict[str, float] = {}
        if self.consecutive_holds > self.cfg.max_consecutive_holds:
            # Conveyor cannot wait indefinitely: item is force-rejected.
            penalty = -self.rcfg.forced_reject_penalty - stall_penalty(
                self.consecutive_holds, beta=self.cfg.stall_beta
            )
            breakdown["forced_reject_penalty"] = penalty
            self.consecutive_holds = 0
            self._spawn_item()
            return penalty, breakdown
        penalty = -stall_penalty(self.consecutive_holds, beta=self.cfg.stall_beta)
        breakdown["stall_penalty"] = penalty
        return penalty, breakdown

    def _do_scan(self) -> tuple[float, dict]:
        self.consecutive_holds += 1
        breakdown: dict[str, float] = {}
        reward = -stall_penalty(self.consecutive_holds, beta=self.cfg.stall_beta)
        breakdown["stall_penalty"] = reward

        if self.energy >= self.scan_energy_cost:
            self.energy -= self.scan_energy_cost
            ambiguity = 1.0 - float(np.max(self.current_item_true))
            reduced_sigma = sensor_noise_sigma(
                ambiguity, self._active_sigma_min, self._active_sigma_max
            ) * self.cfg.scan_noise_reduction
            self.current_item_obs = self._noisy_observation(
                self.current_item_true, reduced_sigma
            )
            # Scanning now costs reward (was free), so it is only worth it when
            # the reduced uncertainty genuinely improves the next decision.
            energy_cost = -self.rcfg.scan_cost_coef * self.scan_energy_cost
            reward += energy_cost
            breakdown["scan_energy_cost"] = energy_cost
        else:
            reward -= self.rcfg.scan_unavailable_penalty
            breakdown["scan_unavailable_penalty"] = -self.rcfg.scan_unavailable_penalty

        if self.consecutive_holds > self.cfg.max_consecutive_holds:
            penalty = -self.rcfg.forced_reject_penalty
            reward += penalty
            breakdown["forced_reject_penalty"] = penalty
            self.consecutive_holds = 0
            self._spawn_item()

        return reward, breakdown

    def _ship_out_all_bins(self) -> tuple[float, dict]:
        total = 0.0
        breakdown: dict[str, float] = {}
        for i, name in enumerate(BINS):
            if self.bin_total_mass[i] <= 1e-9:
                continue
            contamination_fraction = self.bin_contam_mass[i] / self.bin_total_mass[i]
            factor = contamination_penalty_factor(
                contamination_fraction, p=self.cfg.contamination_exponent
            )
            realized = self.bin_raw_value[i] * factor
            total += realized
            breakdown[f"shipout_{name}"] = realized
            self.episode_stats["shipouts"] += 1

            # Empty the bin; jam-risk history partially carries over,
            # representing imperfect maintenance between ship-outs.
            self.bin_fill[i] = 0.0
            self.bin_raw_value[i] = 0.0
            self.bin_contam_mass[i] = 0.0
            self.bin_total_mass[i] = 0.0
            self.bin_contam_ewma[i] *= 0.5
        return total, breakdown

    # ------------------------------------------------------------------
    # Item generation and observation
    # ------------------------------------------------------------------
    def _generate_item(self) -> dict:
        """Draw one fresh conveyor item under the active demand profile.

        Returns a dict of (true composition, noisy observed composition, mass)
        rather than mutating state, so it can be used both for the current item
        and to fill the lookahead queue.
        """
        dominant = self._rng.choice(5, p=self._category_probs)
        alpha = np.full(5, self._other_alpha, dtype=np.float64)
        alpha[dominant] = self._dominant_alpha
        true_comp = self._rng.dirichlet(alpha)
        mass = self._rng.uniform(*self.item_mass_range)
        ambiguity = 1.0 - float(np.max(true_comp))
        sigma = sensor_noise_sigma(ambiguity, self._active_sigma_min, self._active_sigma_max)
        return {
            "true": true_comp,
            "obs": self._noisy_observation(true_comp, sigma),
            "mass": mass,
        }

    def _spawn_item(self) -> None:
        """Advance the conveyor: promote the head of the lookahead queue to the
        current item and append a freshly generated item to the queue tail.
        With `queue_lookahead == 0` the queue is empty and every call simply
        generates a new current item (original behavior)."""
        if self.queue_lookahead > 0:
            item = self._item_queue.pop(0)
            self._item_queue.append(self._generate_item())
        else:
            item = self._generate_item()

        self.current_item_true = item["true"]
        self.current_item_obs = item["obs"]
        self.current_item_mass = item["mass"]

    def _noisy_observation(self, true_comp: np.ndarray, sigma: float) -> np.ndarray:
        noise = self._rng.normal(0.0, sigma, size=true_comp.shape)
        noisy = np.clip(true_comp + noise, 0.0, None)
        total = noisy.sum()
        if total <= 1e-9:
            return true_comp.copy()
        return noisy / total

    def _decrement_jam_cooldowns(self) -> None:
        self.bin_jam_cooldown = np.maximum(self.bin_jam_cooldown - 1, 0)

    # ------------------------------------------------------------------
    # Observation / info assembly
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        fill_frac = np.clip(self.bin_fill / self.bin_capacity, 0.0, 1.0)
        contam_frac = np.where(
            self.bin_total_mass > 1e-9,
            self.bin_contam_mass / np.maximum(self.bin_total_mass, 1e-9),
            0.0,
        )
        jam_frac = np.clip(
            self.bin_jam_cooldown / max(self.cfg.jam_cooldown_steps, 1), 0.0, 1.0
        )
        energy_frac = np.clip(self.energy / self.energy_budget_max, 0.0, 1.0)
        hold_frac = np.clip(
            self.consecutive_holds / max(self.cfg.max_consecutive_holds, 1), 0.0, 1.0
        )
        mass_max = self.item_mass_range[1]
        mass_frac = np.clip(self.current_item_mass / mass_max, 0.0, 1.0)
        steps_remaining_frac = np.clip(
            (self.max_steps - self.step_count) / max(self.max_steps, 1), 0.0, 1.0
        )

        parts = [
            fill_frac.astype(np.float32),
            contam_frac.astype(np.float32),
            jam_frac.astype(np.float32),
            np.array([energy_frac], dtype=np.float32),
            np.array([hold_frac], dtype=np.float32),
            self.current_item_obs.astype(np.float32),
            np.array([mass_frac], dtype=np.float32),
            np.array([steps_remaining_frac], dtype=np.float32),
        ]

        # Conveyor lookahead: append (observed composition, mass) for each of
        # the next `queue_lookahead` items so the policy can plan ahead.
        for item in self._item_queue:
            parts.append(item["obs"].astype(np.float32))
            parts.append(
                np.array([np.clip(item["mass"] / mass_max, 0.0, 1.0)], dtype=np.float32)
            )

        obs = np.concatenate(parts)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "step": self.step_count,
            "last_action": self.last_action_name,
            "reward_breakdown": dict(self.last_reward_breakdown),
            "total_reward": self.total_reward,
            "bin_fill": self.bin_fill.copy(),
            "bin_jam_cooldown": self.bin_jam_cooldown.copy(),
            "energy": self.energy,
            "episode_stats": dict(self.episode_stats),
            "current_item_true": self.current_item_true.copy(),
            "current_item_obs": self.current_item_obs.copy(),
            "action_mask": self.action_masks(),
            "active_regime": {
                "category_probs": self._category_probs.copy(),
                "sigma_min": self._active_sigma_min,
                "sigma_max": self._active_sigma_max,
                "jam_baseline": self._active_jam_baseline,
            },
        }

    def to_json(self) -> dict:
        """Serialize the full render-relevant environment state to a plain,
        JSON-safe dict (only Python floats/ints/strings/lists -- no numpy).

        This is what makes the environment directly consumable by a web or
        mobile frontend: the REST API in `serve.py` returns exactly this
        object on every reset/step, so a browser or app can render the
        facility (bins, conveyor item, energy, stats) without any Python or
        Pygame dependency. See README "Serving the environment as an API".
        """
        return {
            "step": int(self.step_count),
            "max_steps": int(self.max_steps),
            "total_reward": float(self.total_reward),
            "last_action": self.last_action_name or None,
            "last_reward_breakdown": {k: float(v) for k, v in self.last_reward_breakdown.items()},
            "energy": float(self.energy),
            "energy_max": float(self.energy_budget_max),
            "consecutive_holds": int(self.consecutive_holds),
            "bins": [
                {
                    "name": name,
                    "fill_fraction": float(min(self.bin_fill[i] / self.bin_capacity, 1.0)),
                    "contamination_fraction": float(
                        self.bin_contam_mass[i] / self.bin_total_mass[i]
                        if self.bin_total_mass[i] > 1e-9 else 0.0
                    ),
                    "jammed": bool(self.bin_jam_cooldown[i] > 0),
                    "jam_cooldown": int(self.bin_jam_cooldown[i]),
                }
                for i, name in enumerate(BINS)
            ],
            "current_item": {
                "observed_composition": {
                    m: float(self.current_item_obs[j]) for j, m in enumerate(MATERIALS)
                },
                "mass": float(self.current_item_mass),
            },
            "upcoming_items": [
                {
                    "observed_composition": {
                        m: float(item["obs"][j]) for j, m in enumerate(MATERIALS)
                    },
                    "mass": float(item["mass"]),
                }
                for item in self._item_queue
            ],
            "episode_stats": {k: int(v) for k, v in self.episode_stats.items()},
            "action_names": ACTION_NAMES,
            "action_mask": [bool(v) for v in self.action_masks()],
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        from environment.rendering import PygameRenderer  # local import: optional dep path

        if self._pygame_renderer is None:
            self._pygame_renderer = PygameRenderer(render_mode=self.render_mode)
        return self._pygame_renderer.render(self)

    def close(self):
        if self._pygame_renderer is not None:
            self._pygame_renderer.close()
            self._pygame_renderer = None
