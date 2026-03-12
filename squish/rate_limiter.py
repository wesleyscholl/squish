"""squish/rate_limiter.py

TokenBucketRateLimiter — Token-bucket per-tenant rate limiter with burst
support for LLM inference serving.

Controlling per-tenant inference request rates prevents individual tenants
from monopolising shared GPU capacity and protects the serving stack from
traffic spikes that would exceed memory or compute budgets.

The token-bucket algorithm provides both a sustained rate limit and a burst
allowance.  Each tenant maintains an independent bucket that replenishes at
``rate`` tokens per second up to a maximum of ``burst`` tokens.  Requests
consume tokens atomically; requests that would overdraft the bucket are
rejected immediately with an estimated wait time in milliseconds.

Replenishment is *lazy*: tokens are added on-demand when :meth:`consume` or
:meth:`refill` is called, based on the elapsed time since the last
interaction.  This design requires no background thread.

Unknown tenants are auto-registered with the default configuration on their
first :meth:`consume` call, enabling zero-configuration onboarding.

Refill formula::

    new_tokens = min(burst, current_tokens + rate * elapsed_seconds)

Example usage::

    import time
    from squish.rate_limiter import RateLimitConfig, TokenBucketRateLimiter

    cfg     = RateLimitConfig(rate=10.0, burst=20)
    limiter = TokenBucketRateLimiter(default_config=cfg)
    limiter.register_tenant("user-42")

    result = limiter.consume("user-42", n_tokens=5)
    print(result.allowed, result.tokens_remaining)
"""

from __future__ import annotations

__all__ = ["RateLimitConfig", "LimitResult", "TokenBucketRateLimiter"]

import time
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RateLimitConfig:
    """Configuration for a single token-bucket instance.

    Attributes:
        rate:  Token replenishment rate in tokens per second.  Must be > 0.
        burst: Bucket capacity — the maximum number of tokens that can
               accumulate.  Must be >= 1.
    """

    rate: float
    burst: int

    def __post_init__(self) -> None:
        if self.rate <= 0.0:
            raise ValueError(f"rate must be > 0, got {self.rate}")
        if self.burst < 1:
            raise ValueError(f"burst must be >= 1, got {self.burst}")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class LimitResult:
    """Result from :meth:`TokenBucketRateLimiter.consume`.

    Attributes:
        allowed:          ``True`` iff the requested tokens were consumed.
        tokens_consumed:  Number of tokens consumed (0 if not allowed).
        tokens_remaining: Bucket level after this operation.
        wait_ms:          Estimated milliseconds until *n_tokens* become
                          available.  0.0 when the request was allowed.
    """

    allowed: bool
    tokens_consumed: int
    tokens_remaining: float
    wait_ms: float


# ---------------------------------------------------------------------------
# Internal bucket state
# ---------------------------------------------------------------------------


class _Bucket:
    """Token-bucket state for a single tenant."""

    __slots__ = ("config", "tokens", "last_refill_time")

    def __init__(self, config: RateLimitConfig, now: float) -> None:
        self.config: RateLimitConfig = config
        # Start with a full bucket so the first burst is always allowed.
        self.tokens: float = float(config.burst)
        self.last_refill_time: float = now

    def refill(self, now: float) -> None:
        """Add accrued tokens since *last_refill_time*, capped at burst."""
        elapsed = max(0.0, now - self.last_refill_time)
        self.tokens = min(
            float(self.config.burst),
            self.tokens + self.config.rate * elapsed,
        )
        self.last_refill_time = now


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TokenBucketRateLimiter:
    """Per-tenant token-bucket rate limiter with lazy replenishment.

    Each tenant gets an independent :class:`_Bucket` instance.  Tokens
    replenish at ``rate`` per second up to ``burst``.  :meth:`consume` is
    the primary API for request admission control.

    Args:
        default_config: :class:`RateLimitConfig` applied to tenants that have
                        not been registered with a custom configuration.
    """

    def __init__(self, default_config: RateLimitConfig) -> None:
        self._default_config = default_config
        self._buckets: dict[str, _Bucket] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_tenant(
        self,
        tenant_id: str,
        config: Optional[RateLimitConfig] = None,
    ) -> None:
        """Register a tenant with an optional custom rate configuration.

        If *tenant_id* is already registered this call is a no-op; the
        existing bucket level is preserved.

        Args:
            tenant_id: Unique identifier for the tenant.
            config:    Optional per-tenant :class:`RateLimitConfig`.  Uses
                       the default config when ``None``.
        """
        self._ensure_tenant(tenant_id, config=config, now=None)

    def consume(
        self,
        tenant_id: str,
        n_tokens: int = 1,
        now: Optional[float] = None,
    ) -> LimitResult:
        """Attempt to consume *n_tokens* for *tenant_id*.

        Unknown tenants are auto-registered with the default configuration.
        Refill is applied before the bucket check so that tokens accrued
        since the last call are counted.

        Args:
            tenant_id: Tenant identifier.
            n_tokens:  Number of tokens to consume.  Must be >= 1.
            now:       Current timestamp in seconds.  Uses
                       ``time.monotonic()`` when ``None``.

        Returns:
            A :class:`LimitResult` describing whether the request is allowed
            and how many tokens remain.

        Raises:
            ValueError: If *n_tokens* < 1.
        """
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
        if now is None:
            now = time.monotonic()

        # Auto-register unknown tenants, passing the resolved timestamp to
        # avoid a second monotonic call.
        self._ensure_tenant(tenant_id, now=now)

        bucket = self._buckets[tenant_id]
        bucket.refill(now)

        if bucket.tokens >= n_tokens:
            bucket.tokens -= n_tokens
            return LimitResult(
                allowed=True,
                tokens_consumed=n_tokens,
                tokens_remaining=bucket.tokens,
                wait_ms=0.0,
            )

        # Compute estimated wait until the deficit is replenished.
        deficit = float(n_tokens) - bucket.tokens
        wait_s = deficit / bucket.config.rate
        return LimitResult(
            allowed=False,
            tokens_consumed=0,
            tokens_remaining=bucket.tokens,
            wait_ms=wait_s * 1000.0,
        )

    def refill(self, tenant_id: str, now: Optional[float] = None) -> float:
        """Trigger a manual refill for *tenant_id*.

        Args:
            tenant_id: Registered tenant identifier.
            now:       Current timestamp in seconds.  Uses
                       ``time.monotonic()`` when ``None``.

        Returns:
            Current token count after refill.

        Raises:
            KeyError: If *tenant_id* has never been registered.
        """
        if tenant_id not in self._buckets:
            raise KeyError(f"Unknown tenant: {tenant_id!r}")
        if now is None:
            now = time.monotonic()
        self._buckets[tenant_id].refill(now)
        return self._buckets[tenant_id].tokens

    @property
    def tenants(self) -> dict[str, float]:
        """Dict of tenant_id → current token count (snapshot, no refill)."""
        return {tid: bucket.tokens for tid, bucket in self._buckets.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_tenant(
        self,
        tenant_id: str,
        config: Optional[RateLimitConfig] = None,
        now: Optional[float] = None,
    ) -> None:
        """Register *tenant_id* if not already known.

        Accepts an optional *now* timestamp to avoid redundant
        ``time.monotonic()`` calls in hot paths like :meth:`consume`.
        """
        if tenant_id not in self._buckets:
            cfg = config if config is not None else self._default_config
            ts = now if now is not None else time.monotonic()
            self._buckets[tenant_id] = _Bucket(cfg, ts)
