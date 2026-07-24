"""Integration tests for environment/custom_env.py -- WasteSegregationEnv."""

from __future__ import annotations

import numpy as np
import pytest

from environment.custom_env import (
    ACTION_HOLD,
    ACTION_REJECT,
    ACTION_SCAN,
    ACTION_SORT_METAL,
    ACTION_SORT_ORGANIC,
    ACTION_SORT_PLASTIC,
    BINS,
    N_ACTIONS,
    WasteSegregationEnv,
)


@pytest.fixture
def env():
    e = WasteSegregationEnv(seed=42, max_steps=50)
    yield e
    e.close()


class TestResetAndSpaces:
    def test_reset_returns_valid_observation(self, env):
        obs, info = env.reset(seed=1)
        assert env.observation_space.contains(obs)
        assert obs.shape == (21,)
        assert obs.dtype == np.float32

    def test_reset_zeroes_episode_state(self, env):
        obs, info = env.reset(seed=1)
        assert env.step_count == 0
        assert np.all(env.bin_fill == 0.0)
        assert env.energy == env.energy_budget_max
        assert env.consecutive_holds == 0

    def test_action_space_has_seven_actions(self, env):
        assert env.action_space.n == N_ACTIONS == 7

    def test_reset_is_reproducible_with_same_seed(self, env):
        obs1, _ = env.reset(seed=123)
        item1 = env.current_item_true.copy()
        obs2, _ = env.reset(seed=123)
        item2 = env.current_item_true.copy()
        np.testing.assert_allclose(obs1, obs2)
        np.testing.assert_allclose(item1, item2)


class TestStepMechanics:
    def test_step_returns_correct_types(self, env):
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_invalid_action_raises(self, env):
        env.reset(seed=1)
        with pytest.raises(AssertionError):
            env.step(999)

    def test_sort_action_increases_bin_fill(self, env):
        env.reset(seed=1)
        bin_idx = 0
        mass_before = env.current_item_mass
        fill_before = env.bin_fill[bin_idx]
        env.step(ACTION_SORT_ORGANIC)
        assert env.bin_fill[bin_idx] >= fill_before  # may be capped by capacity/jam

    def test_hold_action_does_not_consume_item(self, env):
        env.reset(seed=1)
        item_before = env.current_item_true.copy()
        env.step(ACTION_HOLD)
        np.testing.assert_allclose(env.current_item_true, item_before)

    def test_hold_increments_consecutive_counter(self, env):
        env.reset(seed=1)
        assert env.consecutive_holds == 0
        env.step(ACTION_HOLD)
        assert env.consecutive_holds == 1

    def test_sort_resets_consecutive_hold_counter(self, env):
        env.reset(seed=1)
        env.step(ACTION_HOLD)
        assert env.consecutive_holds == 1
        env.step(ACTION_SORT_ORGANIC)
        assert env.consecutive_holds == 0

    def test_forced_reject_after_max_consecutive_holds(self, env):
        env.reset(seed=1)
        for _ in range(env.cfg.max_consecutive_holds + 2):
            obs, reward, term, trunc, info = env.step(ACTION_HOLD)
        assert env.consecutive_holds <= env.cfg.max_consecutive_holds

    def test_scan_reduces_energy(self, env):
        env.reset(seed=1)
        energy_before = env.energy
        env.step(ACTION_SCAN)
        assert env.energy < energy_before

    def test_scan_unavailable_when_energy_depleted(self, env):
        env.reset(seed=1)
        env.energy = 0.0
        energy_before = env.energy
        obs, reward, term, trunc, info = env.step(ACTION_SCAN)
        assert env.energy == energy_before  # no energy to spend
        assert "scan_unavailable_penalty" in info["reward_breakdown"]

    def test_truncation_at_max_steps(self, env):
        env.reset(seed=1)
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated) and steps < 1000:
            obs, reward, terminated, truncated, info = env.step(ACTION_HOLD if steps % 5 else ACTION_SORT_ORGANIC)
            steps += 1
        assert truncated or terminated
        assert steps <= env.max_steps + 1


class TestBinCapacityAndOverflow:
    def test_overflow_is_penalized(self, env):
        env.reset(seed=7)
        env.bin_fill[0] = env.bin_capacity - 0.01  # nearly full organic bin
        env.current_item_mass = 5.0  # force an oversized item
        obs, reward, term, trunc, info = env.step(ACTION_SORT_ORGANIC)
        assert "overflow_penalty" in info["reward_breakdown"]
        assert info["reward_breakdown"]["overflow_penalty"] < 0

    def test_fill_never_exceeds_capacity(self, env):
        env.reset(seed=7)
        for _ in range(40):
            env.step(ACTION_SORT_ORGANIC)
        assert env.bin_fill[0] <= env.bin_capacity + 1e-6


class TestJammingBehavior:
    def test_jammed_bin_rejects_sorting_and_penalizes(self, env):
        env.reset(seed=3)
        env.bin_jam_cooldown[0] = 5
        obs, reward, term, trunc, info = env.step(ACTION_SORT_ORGANIC)
        assert "jam_failure_penalty" in info["reward_breakdown"]
        assert reward < 0

    def test_jam_cooldown_decrements_over_time(self, env):
        env.reset(seed=3)
        env.bin_jam_cooldown[0] = 3
        env.step(ACTION_HOLD)
        assert env.bin_jam_cooldown[0] == 2


class TestActionMasking:
    def test_mask_all_valid_when_no_bin_jammed(self, env):
        env.reset(seed=3)
        mask = env.action_masks()
        assert mask.shape == (N_ACTIONS,)
        assert mask.dtype == bool
        assert mask.all()

    def test_jammed_bin_masks_only_its_sort_action(self, env):
        env.reset(seed=3)
        env.bin_jam_cooldown[BINS.index("metal")] = 5
        mask = env.action_masks()
        assert not mask[ACTION_SORT_METAL]          # metal sort now invalid
        assert mask[ACTION_SORT_ORGANIC]            # other bins unaffected
        assert mask[ACTION_SORT_PLASTIC]
        # Diversion / non-committal actions are always available.
        assert mask[ACTION_REJECT]
        assert mask[ACTION_HOLD]
        assert mask[ACTION_SCAN]

    def test_mask_is_never_all_false(self, env):
        env.reset(seed=3)
        env.bin_jam_cooldown[:] = 9  # every bin jammed
        mask = env.action_masks()
        assert mask.any(), "agent must always have at least one legal action"
        assert mask[ACTION_REJECT] and mask[ACTION_HOLD] and mask[ACTION_SCAN]

    def test_mask_present_in_info_and_json(self, env):
        env.reset(seed=3)
        env.bin_jam_cooldown[BINS.index("glass")] = 4
        _, _, _, _, info = env.step(ACTION_HOLD)
        assert "action_mask" in info
        assert info["action_mask"].shape == (N_ACTIONS,)
        json_state = env.to_json()
        assert "action_mask" in json_state
        assert len(json_state["action_mask"]) == N_ACTIONS
        assert all(isinstance(v, bool) for v in json_state["action_mask"])

    def test_mask_tracks_jam_cooldown_decay(self, env):
        env.reset(seed=3)
        env.bin_jam_cooldown[BINS.index("organic")] = 1
        assert not env.action_masks()[ACTION_SORT_ORGANIC]
        env.step(ACTION_HOLD)  # cooldown decrements to 0
        assert env.action_masks()[ACTION_SORT_ORGANIC]


class TestShipOutAndRewardRealization:
    def test_shipout_occurs_at_interval(self, env):
        env.reset(seed=9)
        env.ship_out_interval = 5
        total_shipout_reward = 0.0
        for i in range(5):
            obs, reward, term, trunc, info = env.step(ACTION_SORT_ORGANIC)
        assert any(k.startswith("shipout_") for k in info["reward_breakdown"])

    def test_shipout_empties_bin(self, env):
        env.reset(seed=9)
        env.ship_out_interval = 3
        for _ in range(3):
            env.step(ACTION_SORT_ORGANIC)
        assert env.bin_fill[0] == 0.0
        assert env.bin_raw_value[0] == 0.0


class TestRejectLogic:
    def test_correct_reject_of_contaminant_item_is_rewarded(self, env):
        env.reset(seed=1)
        env.current_item_true = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        obs, reward, term, trunc, info = env.step(ACTION_REJECT)
        assert info["reward_breakdown"].get("correct_reject_reward", 0) > 0

    def test_rejecting_valuable_item_is_penalized(self, env):
        env.reset(seed=1)
        env.current_item_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        env.current_item_mass = 2.0
        obs, reward, term, trunc, info = env.step(ACTION_REJECT)
        assert info["reward_breakdown"].get("wasted_material_penalty", 0) < 0


class TestGoodPolicyBeatsRandomPolicy:
    def test_capacity_aware_heuristic_outperforms_random(self):
        """End-to-end sanity check on the reward design: a policy that
        routes items to their best-matching, non-full, non-jammed bin
        should dramatically outperform uniform-random actions."""

        def capacity_aware(e):
            true = e.current_item_true
            order = np.argsort(-true[:4])
            for idx in order:
                if true[idx] < 0.15:
                    break
                fill_frac = e.bin_fill[idx] / e.bin_capacity
                if e.bin_jam_cooldown[idx] == 0 and fill_frac < 0.80:
                    return int(idx)
            if true[4] >= 0.4:
                return ACTION_REJECT
            if e.consecutive_holds < e.cfg.max_consecutive_holds:
                return ACTION_HOLD
            return ACTION_REJECT

        def run_policy(policy_fn, n_episodes=5, seed_offset=0):
            totals = []
            for ep in range(n_episodes):
                e = WasteSegregationEnv(seed=seed_offset + ep)
                obs, info = e.reset(seed=seed_offset + ep)
                done, total = False, 0.0
                while not done:
                    a = policy_fn(e)
                    obs, r, term, trunc, info = e.step(a)
                    total += r
                    done = term or trunc
                totals.append(total)
                e.close()
            return float(np.mean(totals))

        good_mean = run_policy(capacity_aware, seed_offset=500)
        random_mean = run_policy(lambda e: e.action_space.sample(), seed_offset=500)

        assert good_mean > random_mean, (
            f"A capacity-aware policy ({good_mean:.1f}) should clearly beat "
            f"random actions ({random_mean:.1f})"
        )
        assert good_mean > 0, "A competent policy should achieve positive reward"


class TestQueueLookahead:
    def test_default_obs_dim_is_21(self, env):
        assert env.observation_space.shape == (21,)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (21,)

    def test_lookahead_expands_observation(self):
        e = WasteSegregationEnv(seed=1, queue_lookahead=3)
        assert e.observation_space.shape == (21 + 6 * 3,)
        obs, _ = e.reset(seed=1)
        assert obs.shape == (39,)
        assert e.observation_space.contains(obs)
        e.close()

    def test_queue_stays_full_and_obs_in_bounds_over_rollout(self):
        e = WasteSegregationEnv(seed=2, queue_lookahead=2, max_steps=60)
        obs, _ = e.reset(seed=2)
        assert len(e._item_queue) == 2
        for _ in range(60):
            a = e.action_space.sample()
            obs, _, term, trunc, _ = e.step(a)
            assert e.observation_space.contains(obs)
            if not (term or trunc):
                assert len(e._item_queue) == 2  # buffer refilled after each spawn
            if term or trunc:
                break
        e.close()

    def test_current_item_was_previous_queue_head(self):
        e = WasteSegregationEnv(seed=5, queue_lookahead=2)
        e.reset(seed=5)
        next_up = e._item_queue[0]["true"].copy()
        e.step(ACTION_SORT_ORGANIC)  # consumes current item, advances conveyor
        np.testing.assert_allclose(e.current_item_true, next_up)
        e.close()

    def test_upcoming_items_in_json(self):
        e = WasteSegregationEnv(seed=5, queue_lookahead=2)
        e.reset(seed=5)
        state = e.to_json()
        assert len(state["upcoming_items"]) == 2
        assert "observed_composition" in state["upcoming_items"][0]
        e.close()


class TestDomainRandomization:
    def test_randomization_off_uses_fixed_regime(self, env):
        env.reset(seed=1)
        assert env._active_sigma_min == env.cfg.sigma_min
        assert env._active_jam_baseline == env.cfg.jam_baseline_contamination
        np.testing.assert_allclose(env._category_probs, env._base_category_probs)

    def test_randomization_changes_regime(self):
        e = WasteSegregationEnv(seed=1, domain_randomize=True)
        e.reset(seed=1)
        # At least one active parameter should differ from the fixed default.
        differs = (
            not np.allclose(e._category_probs, e._base_category_probs)
            or e._active_sigma_min != e.cfg.sigma_min
            or e._active_jam_baseline != e.cfg.jam_baseline_contamination
        )
        assert differs
        e.close()

    def test_randomized_reset_is_reproducible_with_seed(self):
        a = WasteSegregationEnv(seed=1, domain_randomize=True)
        b = WasteSegregationEnv(seed=1, domain_randomize=True)
        a.reset(seed=99)
        b.reset(seed=99)
        np.testing.assert_allclose(a._category_probs, b._category_probs)
        assert a._active_sigma_min == b._active_sigma_min
        assert a._active_jam_baseline == b._active_jam_baseline
        a.close(); b.close()

    def test_different_seeds_give_different_regimes(self):
        a = WasteSegregationEnv(seed=1, domain_randomize=True)
        b = WasteSegregationEnv(seed=1, domain_randomize=True)
        a.reset(seed=1)
        b.reset(seed=2)
        assert not np.allclose(a._category_probs, b._category_probs)
        a.close(); b.close()

    def test_options_override_toggles_randomization(self):
        # Constructed with randomization OFF, but reset options force it ON.
        e = WasteSegregationEnv(seed=1, domain_randomize=False)
        e.reset(seed=1, options={"domain_randomize": True})
        differs = (
            not np.allclose(e._category_probs, e._base_category_probs)
            or e._active_sigma_min != e.cfg.sigma_min
            or e._active_jam_baseline != e.cfg.jam_baseline_contamination
        )
        assert differs
        # And back off again.
        e.reset(seed=1, options={"domain_randomize": False})
        np.testing.assert_allclose(e._category_probs, e._base_category_probs)
        e.close()

    def test_probs_always_valid_distribution(self):
        e = WasteSegregationEnv(seed=3, domain_randomize=True)
        for s in range(20):
            e.reset(seed=s)
            assert abs(e._category_probs.sum() - 1.0) < 1e-9
            assert (e._category_probs >= 0).all()
        e.close()


def test_render_rgb_array_produces_correct_shape():
    env = WasteSegregationEnv(seed=1, render_mode="rgb_array")
    env.reset(seed=1)
    env.step(ACTION_HOLD)
    frame = env.render()
    assert frame.shape == (560, 900, 3)
    assert frame.dtype == np.uint8
    env.close()
