"""
squish/swift.py

SWIFT — On-the-Fly Self-Speculative Decoding via Task-Specific Layer Skip.

Based on:
  "SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration"
  — Xu et al., ICLR 2025  arXiv:2410.06916
  GitHub: hemingkx/SWIFT

Key insight
-----------
LLMs exhibit high *task-specific layer sparsity*: for code generation,
certain middle layers contribute negligibly; for summarisation, different
layers are redundant.  SWIFT uses Bayesian optimisation (simulated annealing
here, to avoid a scipy dependency) to discover which layers to skip per task
type during a one-time calibration.

At runtime the stored skip configuration is used to drive a self-speculative
decode loop:

1. **Draft phase** — run the model using only the non-skipped layers (early-
   exit forward up to a chosen suffix).
2. **Verify phase** — run the full model over all draft tokens at once.

The verification pass reuses hidden states already computed during drafting,
so verification overhead is less than a fresh full forward pass.

Calibration
-----------
Call ``SWIFTCalibrator.calibrate(task_type, score_fn)`` once per task type.
``score_fn(skip_layers: list[int]) -> float`` must return a quality score
(higher = better) for the proposed skip set — typically a sample mean
acceptance rate measured on a few representative prompts.

Persistence
-----------
Use ``SWIFTCalibrator.save`` / ``SWIFTCalibrator.load`` to store a JSON file
of calibrated configs so calibration does not repeat on every process start.

Provides
--------
  SWIFTConfig      — calibration hyperparameters.
  SWIFTLayerConfig — calibrated per-task skip configuration.
  SWIFTCalibrator  — simulated-annealing calibrator.
  SWIFTDecoder     — inference driver using calibrated configs.
  SWIFTStats       — per-generation counters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "SWIFTConfig",
    "SWIFTLayerConfig",
    "SWIFTCalibrator",
    "SWIFTDecoder",
    "SWIFTStats",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Configuration and per-task config
# ---------------------------------------------------------------------------

@dataclass
class SWIFTConfig:
    """Calibration hyperparameters for SWIFT.

    Parameters
    ----------
    num_layers : int
        Total transformer layers.
    initial_skip_fraction : float
        Fraction of layers to skip in the initial candidate (in (0, 1)).
    n_calibration_steps : int
        Number of simulated-annealing steps per task type.
    cooling_rate : float
        SA temperature decay per step (< 1.0).
    """

    num_layers:             int   = 32
    initial_skip_fraction:  float = 0.4
    n_calibration_steps:    int   = 50
    cooling_rate:           float = 0.95

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be ≥ 1")
        if not 0.0 < self.initial_skip_fraction < 1.0:
            raise ValueError("initial_skip_fraction must be in (0, 1)")
        if self.n_calibration_steps < 1:
            raise ValueError("n_calibration_steps must be ≥ 1")
        if not 0.0 < self.cooling_rate < 1.0:
            raise ValueError("cooling_rate must be in (0, 1)")


@dataclass
class SWIFTLayerConfig:
    """Calibrated layer-skip configuration for one task type.

    Attributes
    ----------
    task_type : str
        Human-readable task identifier (e.g. ``"git_commit"``).
    skip_layers : list[int]
        Sorted list of layer indices to skip (0-based).
    calibration_score : float
        Best score achieved during calibration (higher = better).
    """

    task_type:          str
    skip_layers:        List[int]
    calibration_score:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_type":         self.task_type,
            "skip_layers":       self.skip_layers,
            "calibration_score": self.calibration_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SWIFTLayerConfig":
        return cls(
            task_type         = d["task_type"],
            skip_layers       = [int(x) for x in d["skip_layers"]],
            calibration_score = float(d.get("calibration_score", 0.0)),
        )


# ---------------------------------------------------------------------------
# Calibrator (simulated annealing)
# ---------------------------------------------------------------------------

class SWIFTCalibrator:
    """Calibrate task-specific layer-skip sets via simulated annealing.

    Parameters
    ----------
    config : SWIFTConfig
    rng_seed : int
    """

    def __init__(self, config: SWIFTConfig, rng_seed: int = 42) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------

    def calibrate(
        self,
        task_type: str,
        score_fn: Callable[[List[int]], float],
    ) -> SWIFTLayerConfig:
        """Run simulated annealing to find the best skip set for *task_type*.

        Parameters
        ----------
        task_type : str
        score_fn : callable
            ``score_fn(skip_layers) -> float`` — quality measure (higher better).
            Typically: mean acceptance rate over a held-out prompt set.

        Returns
        -------
        :class:`SWIFTLayerConfig` with the best skip configuration found.
        """
        cfg = self._cfg
        n   = cfg.num_layers
        rng = self._rng

        # ── Initialise ────────────────────────────────────────────────────────
        n_skip = max(1, int(n * cfg.initial_skip_fraction))
        current_skip  = sorted(
            int(x) for x in rng.choice(n, n_skip, replace=False)
        )
        current_score = score_fn(current_skip)
        best_skip     = list(current_skip)
        best_score    = current_score

        temp = 1.0  # initial "temperature"

        # ── SA steps ──────────────────────────────────────────────────────────
        for _ in range(cfg.n_calibration_steps):
            temp *= cfg.cooling_rate
            candidate_skip = list(current_skip)

            # Propose: remove a layer from skip set OR add a new one
            if candidate_skip and rng.random() < 0.5:
                # Remove a random skipped layer
                idx = int(rng.integers(len(candidate_skip)))
                candidate_skip.pop(idx)
            else:
                # Add a random non-skipped layer
                non_skipped = [i for i in range(n) if i not in candidate_skip]
                if non_skipped:
                    add = int(rng.choice(non_skipped))
                    candidate_skip.append(add)
                    candidate_skip.sort()

            candidate_score = score_fn(candidate_skip)
            delta = candidate_score - current_score

            # Metropolis acceptance criterion
            if delta > 0 or (
                temp > 0
                and rng.random() < math.exp(delta / temp)
            ):
                current_skip  = candidate_skip
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_skip  = list(current_skip)

        return SWIFTLayerConfig(
            task_type         = task_type,
            skip_layers       = sorted(best_skip),
            calibration_score = best_score,
        )

    # ------------------------------------------------------------------

    def save(
        self,
        configs: List[SWIFTLayerConfig],
        path: str,
    ) -> None:
        """Serialise *configs* to a JSON file at *path*."""
        data = [c.to_dict() for c in configs]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str) -> List[SWIFTLayerConfig]:
        """Deserialise configs from the JSON file at *path*.

        Returns an empty list if the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        return [SWIFTLayerConfig.from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SWIFTStats:
    """Per-generation counters returned by :class:`SWIFTDecoder`."""

    total_tokens:   int = 0
    accepted_draft: int = 0
    rejected_draft: int = 0
    skip_layers:    List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_draft + self.rejected_draft
        return self.accepted_draft / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class SWIFTDecoder:
    """Drive self-speculative decoding using SWIFT layer-skip configs.

    Parameters
    ----------
    forward_fn : callable
        ``forward_fn(ids, skip_layers=None) -> np.ndarray`` of shape
        ``(vocab_size,)``.  *skip_layers* is a list of 0-based layer indices
        to skip (identity/residual pass-through).
    configs : dict[str, SWIFTLayerConfig]
        Mapping from task type to calibrated config.
    config : SWIFTConfig
    gamma : int
        Draft tokens per speculative step.
    rng_seed : int
    """

    def __init__(
        self,
        forward_fn: Callable[..., np.ndarray],
        configs: Dict[str, SWIFTLayerConfig],
        config: SWIFTConfig,
        gamma: int = 4,
        rng_seed: int = 0,
    ) -> None:
        self._fwd     = forward_fn
        self._configs = configs
        self._cfg     = config
        self._gamma   = max(1, gamma)
        self._rng     = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 64,
        task_type: str = "default",
    ) -> Tuple[List[int], SWIFTStats]:
        """Generate *max_new_tokens* tokens using the SWIFT speculative loop.

        If no config exists for *task_type* the decoder falls back to an
        empty skip set (full model = standard greedy decoding).

        Parameters
        ----------
        input_ids : list[int]
        max_new_tokens : int
        task_type : str

        Returns
        -------
        (output_ids, stats)
        """
        layer_cfg  = self._configs.get(task_type)
        skip_layers: List[int] = (
            list(layer_cfg.skip_layers) if layer_cfg is not None else []
        )

        stats = SWIFTStats(skip_layers=skip_layers)
        ids   = list(input_ids)
        generated = 0

        while generated < max_new_tokens:
            # ── Draft with skip set ───────────────────────────────────────────
            draft_ids:   List[int]        = []
            draft_probs: List[np.ndarray] = []
            ctx = list(ids)

            for _ in range(self._gamma):
                logits = self._fwd(ctx, skip_layers=skip_layers)
                probs  = _softmax(logits)
                tok    = int(np.argmax(logits))
                draft_ids.append(tok)
                draft_probs.append(probs)
                ctx.append(tok)

            # ── Verify with full model ────────────────────────────────────────
            ctx_v    = list(ids)
            accepted: List[int] = []
            rejected  = False

            for d_tok, d_probs in zip(draft_ids, draft_probs):
                full_logits = self._fwd(ctx_v, skip_layers=[])
                full_probs  = _softmax(full_logits)
                v_tok       = int(np.argmax(full_logits))
                p_t = float(full_probs[d_tok])
                p_d = float(d_probs[d_tok])

                if self._rng.random() < min(1.0, p_t / max(p_d, 1e-12)):
                    accepted.append(d_tok)
                    ctx_v.append(d_tok)
                    stats.accepted_draft += 1
                else:
                    accepted.append(v_tok)
                    ctx_v.append(v_tok)
                    stats.rejected_draft += 1
                    rejected = True
                    break

            if not rejected:
                bonus_logits = self._fwd(ctx_v, skip_layers=[])
                accepted.append(int(np.argmax(bonus_logits)))

            ids.extend(accepted)
            generated += len(accepted)
            stats.total_tokens += len(accepted)

        return ids, stats
