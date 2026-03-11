"""Numpy-only PINN (Physics-Informed Neural Network) for spoilage residual correction.

Implements a small feedforward network that learns residual corrections
to the Arrhenius-Baranyi ODE baseline prediction.  The physics-informed
loss penalises violations of the first-order decay ODE, ensuring the
network output is consistent with known kinetics.

Architecture:
    PINN prediction = ODE baseline + NN residual correction

Input:  3 features [temp_norm, rh_norm, time_norm]
Hidden: 2 layers of 32 neurons, tanh activation
Output: scalar delta_C in [-0.05, 0.05]

Loss = MSE(C_ode + delta_C, C_target) + lambda_phys * mean(ODE_residual^2)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
TEMP_MEAN = 8.0
TEMP_STD = 10.0
RH_MEAN = 0.85
RH_STD = 0.10
TIME_MEAN = 36.0
TIME_STD = 24.0


class SpoilagePINN:
    """Numpy-only feedforward PINN for spoilage residual correction.

    Parameters
    ----------
    hidden_size : neurons per hidden layer.
    n_hidden : number of hidden layers.
    lambda_phys : weight of the physics-informed loss term.
    lr : learning rate for gradient descent.
    seed : random seed for reproducible Xavier initialization.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        n_hidden: int = 2,
        lambda_phys: float = 1.0,
        lr: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.lambda_phys = lambda_phys
        self.lr = lr

        rng = np.random.default_rng(seed)

        # Build layers: input(3) -> hidden(32) x n_hidden -> output(1)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        in_dim = 3
        for _ in range(n_hidden):
            # Xavier initialization
            scale = np.sqrt(2.0 / (in_dim + hidden_size))
            W = rng.normal(0.0, scale, (in_dim, hidden_size))
            b = np.zeros(hidden_size)
            self.weights.append(W)
            self.biases.append(b)
            in_dim = hidden_size

        # Output layer (zero init — starts with no correction, learns only
        # when data and physics losses provide a meaningful gradient signal)
        W = np.zeros((in_dim, 1))
        b = np.array([0.0])
        self.weights.append(W)
        self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _normalize(
        self,
        temp_C: np.ndarray,
        rh_frac: np.ndarray,
        dt_h: np.ndarray,
    ) -> np.ndarray:
        """Normalise inputs to zero-mean, unit-variance features."""
        t_norm = (temp_C - TEMP_MEAN) / TEMP_STD
        h_norm = (rh_frac - RH_MEAN) / RH_STD
        time_norm = (dt_h - TIME_MEAN) / TIME_STD
        return np.column_stack([t_norm, h_norm, time_norm])

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network.

        Parameters
        ----------
        X : shape (N, 3) normalised input features.

        Returns
        -------
        delta_C : shape (N,) residual corrections clamped to [-0.08, 0.08].
        """
        h = X
        for i in range(self.n_hidden):
            h = np.tanh(h @ self.weights[i] + self.biases[i])

        # Output layer with tanh scaled to [-0.08, 0.08]
        out = np.tanh(h @ self.weights[-1] + self.biases[-1])
        delta_C = 0.08 * out.ravel()
        return delta_C

    def _forward_with_cache(
        self, X: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Forward pass returning intermediate activations for backprop."""
        activations = [X]
        h = X
        for i in range(self.n_hidden):
            z = h @ self.weights[i] + self.biases[i]
            h = np.tanh(z)
            activations.append(h)

        z_out = h @ self.weights[-1] + self.biases[-1]
        out = np.tanh(z_out)
        delta_C = 0.08 * out.ravel()
        activations.append(out)
        return activations, delta_C

    # ------------------------------------------------------------------
    # Physics residual
    # ------------------------------------------------------------------

    def _physics_residual(
        self,
        temp_C: np.ndarray,
        rh_frac: np.ndarray,
        dt_h: np.ndarray,
        C_pred: np.ndarray,
        k_ref: float,
        Ea_R: float,
        T_ref_K: float,
        beta: float,
        lag_lambda: float,
    ) -> np.ndarray:
        """Compute ODE residual: dC/dt + k_eff * C at each point.

        The physics constraint is that dC/dt = -k_eff * C.
        So the residual dC/dt + k_eff * C should be zero when physics
        is perfectly satisfied.
        """
        n = len(C_pred)
        if n < 2:
            return np.zeros(1)

        # Approximate dC/dt with finite differences
        dC_dt = np.zeros(n)
        for i in range(1, n):
            delta_t = dt_h[i] - dt_h[i - 1]
            if delta_t > 0:
                dC_dt[i] = (C_pred[i] - C_pred[i - 1]) / delta_t

        # Compute k_eff at each point
        T_K = temp_C + 273.15
        k = k_ref * np.exp(Ea_R * (1.0 / T_ref_K - 1.0 / T_K))
        k = k * (1.0 + beta * rh_frac)

        alpha = np.where(
            (lag_lambda > 0) & (dt_h + lag_lambda > 0),
            dt_h / (dt_h + lag_lambda),
            1.0,
        )
        k_eff = k * alpha

        residual = dC_dt + k_eff * C_pred
        return residual[1:]  # skip first point (no dC/dt available)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        temp_C: np.ndarray,
        rh_frac: np.ndarray,
        dt_h: np.ndarray,
        C_target: np.ndarray,
        k_ref: float,
        Ea_R: float,
        T_ref_K: float,
        beta: float,
        lag_lambda: float,
        epochs: int = 200,
    ) -> Dict[str, List[float]]:
        """Train the PINN using numpy-based gradient descent.

        Parameters
        ----------
        temp_C : temperature trajectory (Celsius).
        rh_frac : relative humidity trajectory [0, 1].
        dt_h : time trajectory (hours from start).
        C_target : ODE baseline solution (quality fraction).
        k_ref, Ea_R, T_ref_K, beta, lag_lambda : ODE parameters.
        epochs : number of training iterations.

        Returns
        -------
        Training history dict with ``"loss"`` key.
        """
        X = self._normalize(temp_C, rh_frac, dt_h)
        n = len(X)
        history: Dict[str, List[float]] = {"loss": []}

        for epoch in range(epochs):
            # Forward
            activations, delta_C = self._forward_with_cache(X)
            C_pred = C_target + delta_C

            # Data loss: MSE(C_pred, C_target)
            data_err = delta_C  # C_pred - C_target = delta_C
            data_loss = float(np.mean(data_err ** 2))

            # Physics loss
            phys_res = self._physics_residual(
                temp_C, rh_frac, dt_h, C_pred,
                k_ref, Ea_R, T_ref_K, beta, lag_lambda,
            )
            phys_loss = float(np.mean(phys_res ** 2))

            total_loss = data_loss + self.lambda_phys * phys_loss
            history["loss"].append(total_loss)

            # Backprop (gradient of data loss w.r.t. weights)
            # d(loss)/d(delta_C) = 2 * delta_C / n
            d_delta = (2.0 / n) * data_err

            # Through output tanh and 0.05 scale
            out_pre_tanh = activations[-1]  # tanh output
            d_out = d_delta.reshape(-1, 1) * 0.08 * (1.0 - out_pre_tanh ** 2)

            # Output layer gradients
            h_last = activations[-2]
            dW_out = h_last.T @ d_out / n
            db_out = d_out.mean(axis=0)

            d_h = d_out @ self.weights[-1].T

            # Hidden layer gradients (reverse order)
            for i in range(self.n_hidden - 1, -1, -1):
                # tanh derivative
                h_i = activations[i + 1]
                d_pre = d_h * (1.0 - h_i ** 2)

                h_prev = activations[i]
                dW = h_prev.T @ d_pre / n
                db = d_pre.mean(axis=0)

                # Update
                self.weights[i] -= self.lr * dW
                self.biases[i] -= self.lr * db

                if i > 0:
                    d_h = d_pre @ self.weights[i].T

            # Update output layer
            self.weights[-1] -= self.lr * dW_out
            self.biases[-1] -= self.lr * db_out

        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        temp_C: np.ndarray,
        rh_frac: np.ndarray,
        dt_h: np.ndarray,
    ) -> np.ndarray:
        """Return residual corrections for the given trajectory.

        Inputs are normalised internally before the forward pass.
        """
        X = self._normalize(
            np.asarray(temp_C, dtype=np.float64),
            np.asarray(rh_frac, dtype=np.float64),
            np.asarray(dt_h, dtype=np.float64),
        )
        return self.forward(X)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, list]:
        """Serialise network weights for reproducibility."""
        return {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    def load_state_dict(self, d: Dict[str, list]) -> None:
        """Load network weights from a serialised state dict."""
        self.weights = [np.array(w) for w in d["weights"]]
        self.biases = [np.array(b) for b in d["biases"]]
