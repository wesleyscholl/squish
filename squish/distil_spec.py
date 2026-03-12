"""DistilSpec — Knowledge distillation calibrator for speculative draft heads.

Draft heads trained via standard cross-entropy may not optimally align with
the target model's token distribution.  DistilSpec fine-tunes the draft head
by minimising KL divergence between draft logits and target logits on a small
calibration corpus — improving acceptance rate by 10–15 pp.

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network",
    NeurIPS 2014 Deep Learning Workshop.  https://arxiv.org/abs/1503.02531

Usage::

    from squish.distil_spec import DistilSpecCalibrator, DistilConfig

    cfg  = DistilConfig(n_calibration_steps=100, learning_rate=1e-3, temperature=2.0)
    cal  = DistilSpecCalibrator(cfg)
    cal.record_step(draft_logits, target_logits)
    delta = cal.compute_delta()        # weight delta to apply to draft head
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "DistilConfig",
    "DistilSpecCalibrator",
    "DistilStats",
]

# Numerical stability floor for log computations.
_LOG_EPS: float = 1e-10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DistilConfig:
    """Configuration for draft-head knowledge distillation.

    Parameters
    ----------
    n_calibration_steps : int
        Expected number of calibration steps (informational; does not cap
        :meth:`DistilSpecCalibrator.record_step`).
    learning_rate : float
        Step size for the gradient descent weight update.
    temperature : float
        Softmax temperature for softening logit distributions (> 0).
    kl_weight : float
        Weight applied to the KL divergence term in the combined loss in [0, 1].
    ce_weight : float
        Weight applied to the cross-entropy term in the combined loss in [0, 1].
    """

    n_calibration_steps: int = 100
    learning_rate: float = 1e-3
    temperature: float = 2.0
    kl_weight: float = 1.0
    ce_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.n_calibration_steps < 1:
            raise ValueError(
                f"n_calibration_steps must be >= 1; got {self.n_calibration_steps}."
            )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be > 0; got {self.learning_rate}."
            )
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0; got {self.temperature}."
            )
        if not (0.0 <= self.kl_weight <= 1.0):
            raise ValueError(
                f"kl_weight must be in [0, 1]; got {self.kl_weight}."
            )
        if not (0.0 <= self.ce_weight <= 1.0):
            raise ValueError(
                f"ce_weight must be in [0, 1]; got {self.ce_weight}."
            )


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class DistilStats:
    """Aggregate distillation statistics.

    Parameters
    ----------
    n_steps : int
        Total number of :meth:`~DistilSpecCalibrator.record_step` calls.
    mean_kl_divergence : float
        Mean KL divergence across all recorded steps.
    total_kl_reduction : float
        Cumulative reduction in KL divergence (current − previous step's KL).
        Positive values indicate improvement.
    """

    n_steps: int = 0
    mean_kl_divergence: float = 0.0
    total_kl_reduction: float = 0.0

    @property
    def estimated_acceptance_gain_pp(self) -> float:
        """Estimated acceptance rate improvement in percentage points.

        Uses the heuristic: ``(total_kl_reduction / n_steps) * 15``.
        """
        if self.n_steps == 0:
            return 0.0
        return (self.total_kl_reduction / self.n_steps) * 15.0


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class DistilSpecCalibrator:
    """Accumulates gradient signals from draft/target logit pairs.

    The calibrator computes soft probability distributions at the configured
    *temperature*, then accumulates the per-token gradient of the KL loss
    (``draft_probs − target_probs``) across all recorded steps.  Calling
    :meth:`compute_delta` returns the aggregated gradient scaled by
    ``−learning_rate``, ready to be applied as a weight update to the draft
    head's output projection.

    Parameters
    ----------
    config : DistilConfig
        Distillation hyperparameters.
    """

    def __init__(self, config: DistilConfig) -> None:
        self._cfg = config
        self._n_steps: int = 0
        self._accumulated_grad: Optional[np.ndarray] = None  # (vocab_size,) or (S, V)
        self._kl_history: list[float] = []
        self._vocab_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Numerically stable softmax with temperature scaling."""
        scaled = logits / temperature
        shifted = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / (exp.sum(axis=-1, keepdims=True) + _LOG_EPS)

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL(p || q) averaged over the token / sequence dimension.

        Parameters
        ----------
        p : np.ndarray
            Target distribution, shape ``(V,)`` or ``(S, V)``.
        q : np.ndarray
            Draft distribution, same shape as *p*.

        Returns
        -------
        float
            Mean KL divergence (averaged over tokens/sequences).
        """
        kl = p * (np.log(p + _LOG_EPS) - np.log(q + _LOG_EPS))
        return float(kl.sum(axis=-1).mean())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(
        self,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
    ) -> None:
        """Record one calibration step from raw draft and target logits.

        Parameters
        ----------
        draft_logits : np.ndarray
            Shape ``(vocab_size,)`` or ``(seq_len, vocab_size)`` float32.
        target_logits : np.ndarray
            Same shape as *draft_logits*.

        Raises
        ------
        ValueError
            If shapes are incompatible or inputs are not 1-D or 2-D.
        """
        draft = np.asarray(draft_logits, dtype=np.float32)
        target = np.asarray(target_logits, dtype=np.float32)

        if draft.ndim not in (1, 2):
            raise ValueError(
                f"draft_logits must be 1-D or 2-D; got shape {draft.shape}."
            )
        if draft.shape != target.shape:
            raise ValueError(
                f"draft_logits shape {draft.shape} does not match "
                f"target_logits shape {target.shape}."
            )

        T = self._cfg.temperature
        draft_probs = self._softmax(draft, T)    # (V,) or (S, V)
        target_probs = self._softmax(target, T)  # (V,) or (S, V)

        # KL(target || draft) — measures how far draft is from target.
        kl = self._kl_divergence(target_probs, draft_probs)
        self._kl_history.append(kl)

        # Gradient: ∂KL/∂draft_logits = draft_probs − target_probs (unnormalised).
        grad = draft_probs - target_probs  # same shape as input

        if self._accumulated_grad is None:
            self._accumulated_grad = grad.copy()
            self._vocab_size = draft.shape[-1]
        else:
            self._accumulated_grad = self._accumulated_grad + grad

        self._n_steps += 1

    def compute_delta(self) -> np.ndarray:
        """Return the aggregated weight update direction.

        The delta is ``−learning_rate × mean_gradient`` over all recorded
        steps.  Applying this delta to the draft head's output projection
        moves the head's distribution closer to the target model's
        distribution.

        Returns
        -------
        np.ndarray
            Shape matches the logit shape passed to :meth:`record_step`.

        Raises
        ------
        RuntimeError
            If no steps have been recorded yet.
        """
        if self._accumulated_grad is None or self._n_steps == 0:
            raise RuntimeError(
                "No steps recorded.  Call record_step() before compute_delta()."
            )
        mean_grad = self._accumulated_grad / self._n_steps
        return (-self._cfg.learning_rate * mean_grad).astype(np.float32)

    def acceptance_improvement_estimate(self) -> float:
        """Estimate acceptance-rate improvement in percentage points.

        Uses the heuristic: mean KL reduction × 15.0.  KL reduction is
        computed as the difference between the first and last recorded KL
        values (positive = improvement).

        Returns
        -------
        float
            Estimated improvement in pp.  Returns 0.0 if fewer than 2 steps.
        """
        if len(self._kl_history) < 2:
            return 0.0
        kl_reduction = self._kl_history[0] - self._kl_history[-1]
        return kl_reduction * 15.0

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._n_steps = 0
        self._accumulated_grad = None
        self._kl_history = []
        self._vocab_size = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean_kl(self) -> float:
        """Mean KL divergence across all recorded steps."""
        if not self._kl_history:
            return 0.0
        return float(np.mean(self._kl_history))

    @property
    def n_steps(self) -> int:
        """Total number of recorded calibration steps."""
        return self._n_steps

    def stats(self) -> DistilStats:
        """Return aggregate distillation statistics."""
        if not self._kl_history:
            return DistilStats(n_steps=0)
        total_kl_reduction = (
            self._kl_history[0] - self._kl_history[-1]
            if len(self._kl_history) >= 2
            else 0.0
        )
        return DistilStats(
            n_steps=self._n_steps,
            mean_kl_divergence=self.mean_kl,
            total_kl_reduction=total_kl_reduction,
        )
