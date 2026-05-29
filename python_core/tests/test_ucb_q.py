"""Tests verifying the implementation of the Optimistic UCB-Q router.

These tests ensure that:
1. Configuring MDPConfig with `use_ucb_q` works as expected.
2. States are initialized optimistically to self.n_actions when UCB-Q is active.
3. Action selection is pure greedy on the optimistic Q-values (no random ε-exploration).
4. Q-table updates apply learning rate decay and the UCB exploration bonus correctly.
"""

import math
import numpy as np
import pytest

from python_core.router.learned_router import MDPRouter, MDPConfig, MDPState
from python_core.tests.conftest import FakeEngine


def test_ucbq_config():
    """Verify that MDPConfig defaults are correct and can be configured."""
    cfg = MDPConfig()
    assert not cfg.use_ucb_q
    assert cfg.ucb_bonus_coeff == 0.1
    assert cfg.ucb_delta == 0.05

    cfg_custom = MDPConfig(use_ucb_q=True, ucb_bonus_coeff=0.5, ucb_delta=0.01)
    assert cfg_custom.use_ucb_q
    assert cfg_custom.ucb_bonus_coeff == 0.5
    assert cfg_custom.ucb_delta == 0.01


def test_ucbq_optimistic_initialization():
    """Verify that Q-values are initialized optimistically to n_actions under UCB-Q."""
    engines = [FakeEngine("t1", 1), FakeEngine("t2", 2)]
    
    # 1. Standard Q-learning: should initialize to 0.5
    router_standard = MDPRouter(engines=engines, config=MDPConfig(use_ucb_q=False))
    state_key = (0, 4, 0, 0, 0)  # arbitrary state tuple
    q_standard = router_standard._get_q_values(state_key)
    assert np.allclose(q_standard, 0.5)

    # 2. UCB-Q: should initialize to self.n_actions (which is 3 since len(engines)=2 + STOP)
    router_ucb = MDPRouter(engines=engines, config=MDPConfig(use_ucb_q=True))
    assert router_ucb.n_actions == 3
    q_ucb = router_ucb._get_q_values(state_key)
    assert np.allclose(q_ucb, 3.0)


def test_ucbq_action_selection():
    """Verify that UCB-Q selects greedily and does not perform random ε-exploration."""
    engines = [FakeEngine("t1", 1), FakeEngine("t2", 2)]
    # Set high epsilon to trigger exploration if epsilon-greedy was active
    router = MDPRouter(engines=engines, config=MDPConfig(use_ucb_q=True, epsilon=0.9))
    
    state = MDPState(complexity_bin=0, budget_remaining=1.0, n_tiers_tried=0, last_confidence=0.0, last_failed=False)
    state_key = state.to_tuple()
    
    # Set Q-values to make action 1 strictly superior to others
    q_values = router._get_q_values(state_key)
    q_values[0] = 1.0
    q_values[1] = 5.0  # action 1 is the best
    q_values[2] = 0.5
    
    # Selection should be deterministic (always action 1) despite high epsilon
    selections = [router._select_action(state, tried=set()) for _ in range(100)]
    assert all(a == 1 for a in selections), "UCB-Q did not select greedily on Q-values (random exploration occurred)"


def test_ucbq_update_decay_and_bonus():
    """Verify that Q-table updates apply learning rate decay and UCB bonuses correctly."""
    engines = [FakeEngine("t1", 1), FakeEngine("t2", 2)]
    router = MDPRouter(engines=engines, config=MDPConfig(use_ucb_q=True, ucb_bonus_coeff=0.2, discount_factor=0.9))
    
    # Setup transition: state_key -> action 1 -> reward 0.8 -> terminal state (None)
    state_key = (0, 4, 0, 0, 0)
    action = 1
    reward = 0.8
    transitions = [(state_key, action, reward, None)]
    
    H = float(router.n_actions)  # H = 3.0
    assert H == 3.0
    
    # Before update: Q-value is initialized to H = 3.0
    q_init = float(router._get_q_values(state_key)[action])
    assert q_init == 3.0
    
    # Run the update (first visit to state-action pair)
    router._update_q_table(transitions)
    
    # Visit count should be incremented to 1
    assert router._visit_counts[(state_key, action)] == 1
    
    # Verify values mathematically:
    # t = 1
    # alpha_t = (H + 1) / (H + t) = (3 + 1) / (3 + 1) = 1.0
    # b_t = coeff * sqrt(H^3 / t) = 0.2 * sqrt(27 / 1) = 0.2 * 5.1961524... = 1.03923...
    # target = min(H, reward + b_t) = min(3.0, 0.8 + 1.03923...) = 1.83923...
    # new_Q = (1 - alpha_t) * q_init + alpha_t * target = 0 + 1 * 1.83923... = 1.83923...
    
    b_t_expected = 0.2 * math.sqrt(27.0)
    target_expected = min(3.0, 0.8 + b_t_expected)
    q_val_first = router._get_q_values(state_key)[action]
    assert math.isclose(q_val_first, target_expected, rel_tol=1e-5)

    # Run the update again (second visit to state-action pair)
    # Transitions: state_key -> action 1 -> reward 0.5 -> next_state (None)
    transitions2 = [(state_key, action, 0.5, None)]
    router._update_q_table(transitions2)
    
    # Visit count should be incremented to 2
    assert router._visit_counts[(state_key, action)] == 2
    
    # t = 2
    # alpha_t = (H + 1) / (H + t) = 4 / 5 = 0.8
    # b_t = 0.2 * sqrt(27 / 2) = 0.2 * 3.67423... = 0.73484...
    # target = min(3.0, 0.5 + 0.73484...) = 1.23484...
    # Q_new = (1 - 0.8) * Q_prev + 0.8 * target = 0.2 * 1.83923... + 0.8 * 1.23484...
    # Q_new = 0.367846... + 0.98787... = 1.3557...
    
    b_t_2 = 0.2 * math.sqrt(27.0 / 2.0)
    target_expected_2 = min(3.0, 0.5 + b_t_2)
    alpha_t_2 = 4.0 / 5.0
    q_val_expected_2 = (1.0 - alpha_t_2) * q_val_first + alpha_t_2 * target_expected_2
    
    q_val_second = router._get_q_values(state_key)[action]
    assert math.isclose(q_val_second, q_val_expected_2, rel_tol=1e-5)
