"""
Learned Router — Paper 2 Contribution.

Replaces static confidence thresholds with adaptive policies that learn
optimal routing from experience. Two approaches implemented:

1. Thompson Sampling (Multi-Armed Bandit):
   - Each (input_complexity, engine) pair is an arm.
   - Reward = quality_score - λ·cost - μ·latency.
   - Samples from posterior Beta distributions to select engines.
   - Pro: Simple, fast convergence. Con: No sequential state modeling.

2. Contextual MDP (Markov Decision Process):
   - State: (input_features, budget_remaining, engines_tried_so_far)
   - Action: which engine to try next, or STOP (accept current best)
   - Reward: accuracy - λ·cost - μ·latency
   - Solved via Q-learning with function approximation.
   - Pro: Models sequential decisions. Con: Slower to converge.

Key research contribution: modeling STOCHASTIC RELIABILITY — engine failure
rates vary with time-of-day, load, and recent history. The learned router
adapts to non-stationary environments.
"""

import asyncio
import math
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..engines.base import (
    BaseEngine,
    EngineStatus,
    FailureMode,
    InferenceRequest,
    InferenceResponse,
)
from .cascade_router import RoutingDecision


# ═══════════���═══════════════════════════════════════════════════════════════════
# THOMPSON SAMPLING ROUTER (Multi-Armed Bandit)
# ══════════════════��═══════════════════════���════════════════════════════════════


@dataclass
class ArmStats:
    """Beta posterior for one (context_bin, engine) arm under CD-TS.

    Algorithm 1 of paper/theory.tex; references Agrawal & Goyal (2012, §2) for
    the Bernoulli trick and Russac et al. (2019) for the discounting analysis.
    """
    alpha: float = 1.0  # Beta(1,1) = Uniform prior
    beta: float = 1.0
    total_reward: float = 0.0
    n_pulls: int = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.n_pulls, 1)

    def sample(self) -> float:
        """Sample from Beta posterior."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: float, decay: float = 1.0, floor: float = 1.0):
        """Bernoulli-trick discounted update.

        Preserves conjugacy under continuous rewards by drawing y ~ Bernoulli(r)
        and updating α,β with the integer outcome. The Bernoulli trick is the
        unbiased estimator required by the regret bound in Theorem 1 of
        paper/theory.tex. The heuristic `α += r if r > 0.5 else β += 1-r`
        previously used here is biased and the bound does not apply to it.

        Discounting (decay < 1) is applied to this arm only — global decay over
        all arms is not needed for the analysis and accelerates information loss
        unnecessarily.
        """
        self.n_pulls += 1
        self.total_reward += reward
        r = 0.0 if reward < 0.0 else (1.0 if reward > 1.0 else reward)
        y = 1.0 if random.random() < r else 0.0
        self.alpha = max(floor, decay * self.alpha + y)
        self.beta = max(floor, decay * self.beta + (1.0 - y))


@dataclass
class ThompsonConfig:
    """Configuration for the Thompson Sampling router."""
    # Reward function weights
    cost_penalty: float = 10.0          # λ: penalty per dollar spent
    latency_penalty: float = 0.001      # μ: penalty per ms
    quality_weight: float = 1.0         # Weight on confidence/quality

    # Context binning
    n_complexity_bins: int = 5          # Discretize input complexity into N bins

    # Exploration
    exploration_bonus: float = 0.1      # UCB-style bonus for under-explored arms
    min_samples_before_exploit: int = 10 # Explore at least this many times per arm

    # Non-stationarity adaptation
    decay_factor: float = 0.995         # Decay old observations (handles drift)


class ThompsonSamplingRouter:
    """
    Multi-Armed Bandit router using Thompson Sampling.

    Arms: (complexity_bin, engine_id) pairs.
    Reward: quality - λ·cost - μ·latency (clipped to [0, 1]).

    The router bins each request by estimated complexity, then samples
    from the posterior for each engine and picks the one with highest sample.
    """

    def __init__(self, engines: List[BaseEngine], config: ThompsonConfig = None):
        self.config = config or ThompsonConfig()
        self.engines = sorted(engines, key=lambda e: e.tier)

        # Initialize arms: one per (complexity_bin, engine) pair
        self._arms: Dict[Tuple[int, str], ArmStats] = {}
        for bin_idx in range(self.config.n_complexity_bins):
            for engine in self.engines:
                self._arms[(bin_idx, engine.engine_id)] = ArmStats()

        # Track time-varying performance (sliding window)
        self._recent_rewards: Dict[str, List[float]] = {
            e.engine_id: [] for e in self.engines
        }

    async def route(self, request: InferenceRequest) -> Tuple[InferenceResponse, RoutingDecision]:
        """Select best engine via Thompson Sampling, then execute."""
        decision = RoutingDecision(request_id=request.request_id)
        start = time.perf_counter()

        # Step 1: Estimate input complexity
        complexity_bin = self._get_complexity_bin(request)

        # Step 2: Sample from posterior for each available engine
        candidates = []
        for engine in self.engines:
            if engine.status == EngineStatus.UNAVAILABLE:
                continue
            if request.min_tier and engine.tier < request.min_tier:
                continue

            arm_key = (complexity_bin, engine.engine_id)
            arm = self._arms[arm_key]

            # Thompson sample + exploration bonus for under-sampled arms
            sample = arm.sample()
            if arm.n_pulls < self.config.min_samples_before_exploit:
                sample += self.config.exploration_bonus

            candidates.append((sample, engine, arm_key))

        if not candidates:
            return self._empty_response(request, decision, start)

        # Step 3: Sort by sampled value (highest first) and try in order
        candidates.sort(key=lambda x: x[0], reverse=True)

        best_response: Optional[InferenceResponse] = None
        for sample_val, engine, arm_key in candidates:
            # Budget check
            est_cost = engine.estimated_cost(request)
            if request.max_cost and decision.total_cost_usd + est_cost > request.max_cost:
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: budget exceeded"
                )
                continue

            decision.engines_tried.append(engine.engine_id)
            decision.tiers_attempted.append(engine.tier)

            # Execute inference
            response = await engine.infer(request)
            decision.total_cost_usd += response.cost_usd

            # Compute reward
            reward = self._compute_reward(response)

            # Update played arm's posterior with Bernoulli-trick + discount.
            # Per CD-TS (Algorithm 1, paper/theory.tex) we decay only the played
            # arm, not all arms — this is what the regret analysis requires.
            arm = self._arms[arm_key]
            arm.update(reward, decay=self.config.decay_factor, floor=1.0)

            # Track recent rewards
            self._recent_rewards[engine.engine_id].append(reward)
            if len(self._recent_rewards[engine.engine_id]) > 100:
                self._recent_rewards[engine.engine_id].pop(0)

            if response.success and response.confidence > 0.5:
                decision.final_engine = engine.engine_id
                decision.final_tier = engine.tier
                decision.success = True
                decision.total_latency_ms = (time.perf_counter() - start) * 1000
                return response, decision
            else:
                reason = f"{engine.engine_id}: reward={reward:.2f}"
                if not response.success:
                    reason += f" (failed: {response.failure_mode.value})"
                decision.escalation_reasons.append(reason)
                best_response = response

        # Exhausted all candidates
        decision.total_latency_ms = (time.perf_counter() - start) * 1000
        if best_response:
            return best_response, decision
        return self._empty_response(request, decision, start)

    def _get_complexity_bin(self, request: InferenceRequest) -> int:
        """
        Bin request complexity into discrete categories.

        Complexity signals:
        - Prompt length (longer = more complex)
        - Task type
        - Vocabulary richness (unique words / total words)
        """
        prompt = request.prompt
        word_count = len(prompt.split())
        unique_ratio = len(set(prompt.lower().split())) / max(word_count, 1)

        # Simple composite score
        length_score = min(word_count / 200, 1.0)  # Normalize to [0,1]
        complexity_score = (length_score * 0.6 + unique_ratio * 0.4)

        # Map to bin
        bin_idx = int(complexity_score * (self.config.n_complexity_bins - 1))
        return min(bin_idx, self.config.n_complexity_bins - 1)

    def _compute_reward(self, response: InferenceResponse) -> float:
        """
        Reward function: quality - ��·cost - μ·latency, clipped to [0, 1].

        This is the core function to tune for your research experiments.
        """
        if not response.success:
            return 0.0

        quality = response.confidence * self.config.quality_weight
        cost_term = response.cost_usd * self.config.cost_penalty
        latency_term = response.latency_ms * self.config.latency_penalty

        reward = quality - cost_term - latency_term
        return max(0.0, min(1.0, reward))

    def _empty_response(self, request, decision, start):
        decision.total_latency_ms = (time.perf_counter() - start) * 1000
        return InferenceResponse(
            request_id=request.request_id,
            engine_id="none",
            tier=0,
            content="",
            success=False,
            failure_mode=FailureMode.INFRASTRUCTURE,
            error_message="No viable engine found",
        ), decision

    def get_arm_stats(self) -> Dict[str, dict]:
        """Export arm statistics for analysis."""
        result = {}
        for (bin_idx, engine_id), arm in self._arms.items():
            key = f"bin{bin_idx}_{engine_id}"
            result[key] = {
                "alpha": round(arm.alpha, 3),
                "beta": round(arm.beta, 3),
                "mean_reward": round(arm.mean_reward, 4),
                "n_pulls": arm.n_pulls,
                "expected_value": round(arm.alpha / (arm.alpha + arm.beta), 4),
            }
        return result

    def get_engine_preferences(self) -> Dict[int, str]:
        """For each complexity bin, which engine does the policy prefer?"""
        preferences = {}
        for bin_idx in range(self.config.n_complexity_bins):
            best_engine = None
            best_ev = -1
            for engine in self.engines:
                arm = self._arms.get((bin_idx, engine.engine_id))
                if arm:
                    ev = arm.alpha / (arm.alpha + arm.beta)
                    if ev > best_ev:
                        best_ev = ev
                        best_engine = engine.engine_id
            preferences[bin_idx] = best_engine
        return preferences


# ���══════════════════��═══════════════════════���═══════════════════════════════════
# CONTEXTUAL MDP ROUTER (Q-Learning)
# ═══════��════════════��══════════════════════════════════════════════════════════


@dataclass
class MDPState:
    """State representation for the routing MDP."""
    complexity_bin: int          # Input complexity category
    budget_remaining: float      # Fraction of budget left [0, 1]
    n_tiers_tried: int          # How many engines already tried
    last_confidence: float       # Confidence from last attempt (0 if first)
    last_failed: bool           # Did the last attempt fail?

    def to_tuple(self) -> tuple:
        """Discretized state for Q-table lookup."""
        budget_bin = int(self.budget_remaining * 4)  # 5 budget levels
        conf_bin = int(self.last_confidence * 4)     # 5 confidence levels
        return (
            self.complexity_bin,
            min(budget_bin, 4),
            min(self.n_tiers_tried, 3),
            min(conf_bin, 4),
            int(self.last_failed),
        )


@dataclass
class MDPConfig:
    """Configuration for the MDP router."""
    # Q-learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.15           # ε-greedy exploration
    epsilon_decay: float = 0.9995   # Decay exploration over time
    epsilon_min: float = 0.01

    # Reward shaping
    cost_penalty: float = 10.0
    latency_penalty: float = 0.001
    success_bonus: float = 0.5      # Bonus for finishing early (fewer tiers)

    # State space
    n_complexity_bins: int = 5
    max_budget: float = 0.05


class MDPRouter:
    """
    MDP-based router using Q-learning with ε-greedy exploration.

    Models the routing problem as sequential decision-making:
    - State: (complexity, budget_remaining, n_tried, last_confidence, last_failed)
    - Actions: [try_engine_0, try_engine_1, ..., try_engine_K, STOP]
    - Transitions: deterministic given engine response
    - Reward: quality - cost - latency at STOP; intermediate rewards shape exploration

    The STOP action accepts the best response seen so far.
    """

    def __init__(self, engines: List[BaseEngine], config: MDPConfig = None):
        self.config = config or MDPConfig()
        self.engines = sorted(engines, key=lambda e: e.tier)
        self.n_actions = len(self.engines) + 1  # +1 for STOP action
        self.STOP_ACTION = len(self.engines)

        # Q-table: state_tuple -> array of Q-values per action
        self._q_table: Dict[tuple, np.ndarray] = {}
        self._epsilon = self.config.epsilon
        self._episode_count = 0
        self._min_tier: Optional[int] = None

    async def route(self, request: InferenceRequest) -> Tuple[InferenceResponse, RoutingDecision]:
        """Run one episode of the MDP to route a request."""
        decision = RoutingDecision(request_id=request.request_id)
        start = time.perf_counter()

        complexity_bin = self._get_complexity_bin(request)
        budget = request.max_cost or self.config.max_budget
        budget_remaining = 1.0
        self._min_tier = request.min_tier  # consumed by _select_action

        state = MDPState(
            complexity_bin=complexity_bin,
            budget_remaining=budget_remaining,
            n_tiers_tried=0,
            last_confidence=0.0,
            last_failed=False,
        )

        best_response: Optional[InferenceResponse] = None
        episode_transitions = []  # (state, action, reward, next_state)
        engines_tried_set = set()

        for step in range(len(self.engines) + 1):
            # Select action (ε-greedy)
            action = self._select_action(state, engines_tried_set)

            if action == self.STOP_ACTION:
                # Accept best response so far
                if best_response and best_response.success:
                    decision.final_engine = best_response.engine_id
                    decision.final_tier = best_response.tier
                    decision.success = True
                    # Reward for stopping early
                    reward = self.config.success_bonus * (1 - state.n_tiers_tried / len(self.engines))
                    episode_transitions.append((state.to_tuple(), action, reward, None))
                break

            # Try the selected engine
            engine = self.engines[action]
            engines_tried_set.add(action)
            decision.engines_tried.append(engine.engine_id)
            decision.tiers_attempted.append(engine.tier)

            response = await engine.infer(request)
            decision.total_cost_usd += response.cost_usd
            budget_remaining = max(0, 1 - decision.total_cost_usd / budget)

            # Compute step reward
            step_reward = self._compute_step_reward(response)

            # Next state
            next_state = MDPState(
                complexity_bin=complexity_bin,
                budget_remaining=budget_remaining,
                n_tiers_tried=state.n_tiers_tried + 1,
                last_confidence=response.confidence if response.success else 0.0,
                last_failed=not response.success,
            )

            episode_transitions.append((state.to_tuple(), action, step_reward, next_state.to_tuple()))

            if response.success:
                if best_response is None or response.confidence > best_response.confidence:
                    best_response = response

                # If very high confidence, auto-stop (no point escalating)
                if response.confidence > 0.9:
                    decision.final_engine = response.engine_id
                    decision.final_tier = response.tier
                    decision.success = True
                    bonus = self.config.success_bonus
                    episode_transitions.append((next_state.to_tuple(), self.STOP_ACTION, bonus, None))
                    break
            else:
                decision.escalation_reasons.append(
                    f"{engine.engine_id}: {response.failure_mode.value}"
                )

            state = next_state

        # Learn from this episode
        self._update_q_table(episode_transitions)
        self._decay_epsilon()
        self._episode_count += 1

        decision.total_latency_ms = (time.perf_counter() - start) * 1000

        if best_response is None:
            best_response = InferenceResponse(
                request_id=request.request_id,
                engine_id="none",
                tier=0,
                content="",
                success=False,
                failure_mode=FailureMode.INFRASTRUCTURE,
                error_message="MDP exhausted all options",
            )

        return best_response, decision

    def _select_action(self, state: MDPState, tried: set) -> int:
        """ε-greedy action selection."""
        state_key = state.to_tuple()
        q_values = self._get_q_values(state_key)

        # Mask already-tried engines (can't try same engine twice)
        valid_actions = []
        for a in range(self.n_actions):
            if a == self.STOP_ACTION:
                valid_actions.append(a)
            elif a not in tried and self.engines[a].status != EngineStatus.UNAVAILABLE:
                if self._min_tier and self.engines[a].tier < self._min_tier:
                    continue
                valid_actions.append(a)

        if not valid_actions:
            return self.STOP_ACTION

        # ε-greedy
        if random.random() < self._epsilon:
            return random.choice(valid_actions)
        else:
            # Greedy: pick valid action with highest Q-value
            best_action = max(valid_actions, key=lambda a: q_values[a])
            return best_action

    def _compute_step_reward(self, response: InferenceResponse) -> float:
        """Intermediate reward for a single engine call."""
        if not response.success:
            return -0.1  # Small penalty for failed attempts

        quality = response.confidence
        cost_term = response.cost_usd * self.config.cost_penalty
        latency_term = response.latency_ms * self.config.latency_penalty

        return quality - cost_term - latency_term

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Get Q-values for a state, initializing if needed."""
        if state_key not in self._q_table:
            # Optimistic initialization encourages exploration
            self._q_table[state_key] = np.ones(self.n_actions) * 0.5
        return self._q_table[state_key]

    def _update_q_table(self, transitions: list):
        """Batch Q-learning update from one episode's transitions."""
        lr = self.config.learning_rate
        gamma = self.config.discount_factor

        # Reverse pass for proper TD updates
        for i in range(len(transitions) - 1, -1, -1):
            state_key, action, reward, next_state_key = transitions[i]
            q_values = self._get_q_values(state_key)

            if next_state_key is None:
                # Terminal state
                target = reward
            else:
                next_q = self._get_q_values(next_state_key)
                target = reward + gamma * np.max(next_q)

            # Q-learning update
            q_values[action] += lr * (target - q_values[action])

    def _decay_epsilon(self):
        """Reduce exploration over time."""
        self._epsilon = max(
            self.config.epsilon_min,
            self._epsilon * self.config.epsilon_decay,
        )

    def _get_complexity_bin(self, request: InferenceRequest) -> int:
        """Same binning as Thompson router for consistency."""
        prompt = request.prompt
        word_count = len(prompt.split())
        unique_ratio = len(set(prompt.lower().split())) / max(word_count, 1)
        length_score = min(word_count / 200, 1.0)
        complexity_score = length_score * 0.6 + unique_ratio * 0.4
        bin_idx = int(complexity_score * (self.config.n_complexity_bins - 1))
        return min(bin_idx, self.config.n_complexity_bins - 1)

    def get_policy_summary(self) -> dict:
        """Summarize learned policy for analysis."""
        return {
            "episode_count": self._episode_count,
            "epsilon": round(self._epsilon, 4),
            "q_table_size": len(self._q_table),
            "action_names": [e.engine_id for e in self.engines] + ["STOP"],
        }

    def get_q_table_snapshot(self) -> Dict[str, list]:
        """Export Q-table for visualization."""
        snapshot = {}
        for state_key, q_values in self._q_table.items():
            key_str = str(state_key)
            snapshot[key_str] = q_values.tolist()
        return snapshot

    def export_policy(self) -> dict:
        """Export the learned policy as a deterministic mapping (for deployment)."""
        policy = {}
        for state_key, q_values in self._q_table.items():
            best_action = int(np.argmax(q_values))
            action_names = [e.engine_id for e in self.engines] + ["STOP"]
            policy[str(state_key)] = {
                "action": action_names[best_action],
                "q_value": round(float(q_values[best_action]), 4),
                "all_q_values": {
                    name: round(float(v), 4)
                    for name, v in zip(action_names, q_values)
                },
            }
        return policy
