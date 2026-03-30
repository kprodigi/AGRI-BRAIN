"""Online adaptation of context modifier rule weights.

After each decision step, compares the reward received with and without
context modifier. If context improved the outcome (reward_with > reward_without),
the weights of rules that fired are increased. If context hurt, they decrease.

Uses a bandit-style exponential-weight update:
    w_i <- w_i * exp(eta * delta_reward)  if rule_i fired
    w_i <- w_i                             if rule_i did not fire

Weights are renormalized to mean 1.0 after each update to prevent drift.
All updates use the passed numpy RNG for deterministic reproducibility.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class ContextRuleLearner:
    """Online learner for context modifier rule weights.

    Parameters
    ----------
    n_rules : number of rules in MODIFIER_RULES.
    learning_rate : exponential weight update step size.
    rng : numpy random generator for deterministic behavior.
    """

    def __init__(
        self,
        n_rules: int = 5,
        learning_rate: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.weights = np.ones(n_rules, dtype=np.float64)
        self.lr = learning_rate
        self.rng = rng or np.random.default_rng(42)
        self._update_log: List[Dict[str, Any]] = []

    def get_weights(self) -> np.ndarray:
        """Current rule weights (passed to compute_context_modifier)."""
        return self.weights.copy()

    def update(
        self,
        rules_fired: List[int],
        reward_with_context: float,
        reward_without_context: float,
    ) -> None:
        """Update weights based on reward comparison.

        Parameters
        ----------
        rules_fired : indices of MODIFIER_RULES that were active this step.
        reward_with_context : actual reward with context modifier applied.
        reward_without_context : counterfactual reward without modifier.
        """
        delta = reward_with_context - reward_without_context

        for i in rules_fired:
            if 0 <= i < len(self.weights):
                self.weights[i] *= np.exp(self.lr * delta)

        # Renormalize to mean 1.0
        mean_w = self.weights.mean()
        if mean_w > 0:
            self.weights /= mean_w

        # Clamp to [0.2, 3.0] to prevent extreme weights
        self.weights = np.clip(self.weights, 0.2, 3.0)

        self._update_log.append({
            "rules_fired": list(rules_fired),
            "delta_reward": float(delta),
            "weights_after": self.weights.copy(),
        })

    def summary(self) -> Dict[str, Any]:
        """Statistics for paper reporting."""
        deltas = [e["delta_reward"] for e in self._update_log]
        return {
            "final_weights": self.weights.tolist(),
            "n_updates": len(self._update_log),
            "mean_delta_reward": float(np.mean(deltas)) if deltas else 0.0,
            "weight_range": [float(self.weights.min()), float(self.weights.max())],
        }

    def reset(self) -> None:
        """Reset weights and history."""
        self.weights = np.ones_like(self.weights)
        self._update_log.clear()
