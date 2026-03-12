"""squish/model_version_swap.py

ModelVersionManager — Zero-downtime hot model version swap with canary
promotion and one-step rollback.

Serving a new model version without user-visible downtime requires a traffic
ramp: a small fraction of requests are directed to the new version (canary),
its error rate is observed, and the version is promoted to full traffic if it
meets quality criteria.  If errors spike, the canary can be rolled back
instantly without touching the stable active version.

Lifecycle::

    register_version(vid)    — add a version to the registry
    stage(vid)               — designate vid as the canary
    route_request()          — return canary (with canary_fraction probability)
                               or active version
    record_result(vid, ok)   — accumulate success/failure counts
    commit()                 — promote canary → active once
                               min_canary_requests are served
    rollback()               — revert to the previous active version

Routing decision:
    A single :class:`numpy.random.Generator` draw is compared against
    ``canary_fraction``.  If the draw is less than ``canary_fraction`` **and**
    a canary is staged, the canary version is returned; otherwise the active
    version is returned.

Example usage::

    from squish.model_version_swap import SwapPolicy, ModelVersionManager

    policy  = SwapPolicy(canary_fraction=0.1, min_canary_requests=10)
    manager = ModelVersionManager(policy)

    manager.register_version("v1")
    manager.stage("v1")
    manager.commit()    # first version — promoted unconditionally

    manager.register_version("v2")
    manager.stage("v2")
    for _ in range(10):
        vid = manager.route_request()
        manager.record_result(vid, success=True)
    new_active = manager.commit()
    print(f"Active version: {new_active}")   # v2

    restored = manager.rollback()
    print(f"Rolled back to: {restored}")     # v1
"""

from __future__ import annotations

__all__ = ["SwapPolicy", "VersionInfo", "ModelVersionManager"]

import dataclasses
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SwapPolicy:
    """Policy parameters for canary-based version promotion.

    Attributes:
        canary_fraction:     Fraction of traffic routed to the canary version
                             during the ramp period.  Must be in ``(0, 1]``.
        min_canary_requests: Minimum number of requests the canary must serve
                             before :meth:`ModelVersionManager.commit` is
                             permitted (when an active version already exists).
                             Must be >= 1.
    """

    canary_fraction:     float = 0.1
    min_canary_requests: int   = 10

    def __post_init__(self) -> None:
        if not (0.0 < self.canary_fraction <= 1.0):
            raise ValueError(
                f"canary_fraction must be in (0, 1], "
                f"got {self.canary_fraction}"
            )
        if self.min_canary_requests < 1:
            raise ValueError(
                f"min_canary_requests must be >= 1, "
                f"got {self.min_canary_requests}"
            )


# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class VersionInfo:
    """Mutable state record for a single model version.

    Attributes:
        version_id:        Unique identifier string.
        n_requests_served: Total requests routed to this version.
        n_errors:          Requests recorded as failures via
                           :meth:`ModelVersionManager.record_result`.
        is_active:         Whether this is the current primary version.
        is_canary:         Whether this version is in canary promotion.
    """

    version_id:        str
    n_requests_served: int  = 0
    n_errors:          int  = 0
    is_active:         bool = False
    is_canary:         bool = False


# ---------------------------------------------------------------------------
# ModelVersionManager
# ---------------------------------------------------------------------------


class ModelVersionManager:
    """Zero-downtime model version manager with canary promotion and rollback.

    At most one version may be active and at most one may be in canary at any
    given time.  A seeded :class:`numpy.random.Generator` provides
    reproducible routing decisions.

    Args:
        policy: A :class:`SwapPolicy` controlling traffic-split and promotion
                thresholds.
    """

    def __init__(self, policy: SwapPolicy) -> None:
        self._policy:            SwapPolicy             = policy
        self._versions:          dict[str, VersionInfo] = {}
        self._active_version:    Optional[str]          = None
        self._canary_version:    Optional[str]          = None
        self._previous_active:   Optional[str]          = None
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Version lifecycle
    # ------------------------------------------------------------------

    def register_version(self, version_id: str) -> None:
        """Register a new model version in the registry.

        Args:
            version_id: Unique identifier string for the version.

        Raises:
            ValueError: If ``version_id`` is already registered.
        """
        if version_id in self._versions:
            raise ValueError(
                f"Version '{version_id}' is already registered."
            )
        self._versions[version_id] = VersionInfo(version_id=version_id)

    def stage(self, version_id: str) -> None:
        """Designate a registered version as the current canary.

        Only one canary may be active at a time.  A version that is already
        the active primary cannot be staged as canary.

        Args:
            version_id: Version to promote to canary status.

        Raises:
            KeyError:   If ``version_id`` is not registered.
            ValueError: If ``version_id`` is already the active version, or if
                        another canary is already staged.
        """
        info = self._get_version(version_id)

        if info.is_active:
            raise ValueError(
                f"Version '{version_id}' is already the active version."
            )
        if self._canary_version is not None:
            raise ValueError(
                f"Canary version '{self._canary_version}' is already staged; "
                "commit or rollback it before staging a new one."
            )

        info.is_canary       = True
        self._canary_version = version_id

    def route_request(self) -> str:
        """Select a version for the next incoming request.

        When a canary is staged, a uniform random draw is compared to
        ``canary_fraction``.  If the draw is less than ``canary_fraction``, the
        canary version is returned; otherwise the active version is returned.

        When no active version exists but a canary is staged, all requests are
        routed to the canary (bootstrapping the first version).

        Returns:
            Version ID to route the request to.

        Raises:
            RuntimeError: If neither an active nor a canary version is
                          available.
        """
        if self._canary_version is not None and self._active_version is None:
            # First version ever staged — route everything to it.
            return self._canary_version

        if self._active_version is None:
            raise RuntimeError(
                "No active or canary version is available. "
                "Register and stage a version first."
            )

        if self._canary_version is not None:
            draw = float(self._rng.uniform())
            if draw < self._policy.canary_fraction:
                return self._canary_version

        return self._active_version

    def record_result(self, version_id: str, success: bool) -> None:
        """Record the outcome of a completed request.

        Args:
            version_id: Version that served the request.
            success:    ``True`` if the request succeeded; ``False`` on error.

        Raises:
            KeyError: If ``version_id`` is not registered.
        """
        info = self._get_version(version_id)
        info.n_requests_served += 1
        if not success:
            info.n_errors += 1

    def commit(self) -> str:
        """Promote the staged canary to the active version.

        When no active version exists the canary is promoted unconditionally
        (bootstrapping the first version).  Otherwise the canary must have
        served at least ``min_canary_requests`` requests.

        After a successful commit the previous active version is stored for
        potential :meth:`rollback`.

        Returns:
            Version ID of the newly active version.

        Raises:
            RuntimeError: If no canary is staged, or if the canary has not yet
                          served ``min_canary_requests`` requests (and a prior
                          active version exists).
        """
        if self._canary_version is None:
            raise RuntimeError("No canary version is staged to commit.")

        canary_info = self._get_version(self._canary_version)

        if self._active_version is not None:
            served = canary_info.n_requests_served
            if served < self._policy.min_canary_requests:
                raise RuntimeError(
                    f"Canary '{self._canary_version}' has served {served} "
                    f"request(s) but min_canary_requests="
                    f"{self._policy.min_canary_requests}."
                )

        # Demote the current active version to standby.
        if self._active_version is not None:
            self._get_version(self._active_version).is_active = False
            self._previous_active = self._active_version

        # Promote the canary.
        canary_info.is_canary  = False
        canary_info.is_active  = True
        self._active_version   = self._canary_version
        self._canary_version   = None

        return self._active_version

    def rollback(self) -> str:
        """Revert to the version that was active before the last commit.

        The current active version is demoted to standby.  Any staged canary
        is cleared.  The previously active version is restored as active.

        Returns:
            Version ID of the restored (now-active) version.

        Raises:
            RuntimeError: If there is no previous active version available.
        """
        if self._previous_active is None:
            raise RuntimeError(
                "No previous active version available for rollback."
            )

        # Clear any live canary.
        if self._canary_version is not None:
            self._get_version(self._canary_version).is_canary = False
            self._canary_version = None

        # Demote the current active version.
        if self._active_version is not None:
            self._get_version(self._active_version).is_active = False

        # Restore the previous active version.
        prev = self._previous_active
        self._get_version(prev).is_active = True
        self._active_version  = prev
        self._previous_active = None

        return self._active_version

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_version(self) -> Optional[str]:
        """Version ID of the current primary version, or ``None``."""
        return self._active_version

    @property
    def canary_version(self) -> Optional[str]:
        """Version ID of the staged canary version, or ``None``."""
        return self._canary_version

    @property
    def versions(self) -> dict[str, VersionInfo]:
        """Snapshot of all registered :class:`VersionInfo` records."""
        return dict(self._versions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_version(self, version_id: str) -> VersionInfo:
        """Return :class:`VersionInfo` for ``version_id`` or raise :class:`KeyError`."""
        if version_id not in self._versions:
            raise KeyError(f"Version '{version_id}' is not registered.")
        return self._versions[version_id]
