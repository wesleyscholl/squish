"""ToolCache — Tool schema cache and fast function routing.

LLM tool/function calling involves parsing the tool schema JSON at every
request, which incurs repeated deserialization overhead.  ToolCache pre-parses
schemas and routes tool-call responses to the correct handler in O(1).

Reference:
    OpenAI Function Calling specification.
    https://platform.openai.com/docs/guides/function-calling

Usage::

    from squish.tool_cache import ToolSchemaCache, ToolRouter

    cache  = ToolSchemaCache()
    schema = {"name": "get_weather", "parameters": {"city": "string"}}
    hid    = cache.register(schema)
    router = ToolRouter(cache)
    result = router.route("get_weather", {"city": "Tokyo"}, handlers={"get_weather": fn})
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "ToolSchema",
    "ToolSchemaCache",
    "ToolRouter",
    "ToolCacheStats",
]

# ---------------------------------------------------------------------------
# ToolSchema
# ---------------------------------------------------------------------------


@dataclass
class ToolSchema:
    """Parsed representation of an LLM tool/function schema.

    Parameters
    ----------
    name : str
        Unique tool name (e.g. ``"get_weather"``).
    parameters : dict
        Mapping of parameter name to type description.
    description : str
        Human-readable description of the tool's purpose.
    handler_id : str
        Optional identifier linking this schema to a registered handler.
    schema_hash : str
        Hex digest uniquely identifying this schema's name and parameter keys.
    """

    name: str
    parameters: Dict[str, Any]
    description: str = ""
    handler_id: str = ""
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolSchema.name must not be empty.")
        if not isinstance(self.parameters, dict):
            raise ValueError(
                f"ToolSchema.parameters must be a dict; got {type(self.parameters)}."
            )

    @property
    def n_params(self) -> int:
        """Number of parameters defined in the schema."""
        return len(self.parameters)

    @property
    def required_params(self) -> List[str]:
        """List of parameter names (all parameters are treated as required)."""
        return list(self.parameters.keys())


# ---------------------------------------------------------------------------
# ToolCacheStats
# ---------------------------------------------------------------------------


@dataclass
class ToolCacheStats:
    """Aggregate statistics for :class:`ToolSchemaCache` and :class:`ToolRouter`.

    Parameters
    ----------
    n_registrations : int
        Number of schemas successfully registered.
    n_validations : int
        Total :meth:`~ToolSchemaCache.validate_call` invocations.
    n_failures : int
        Number of validation failures.
    n_routes : int
        Total :meth:`~ToolRouter.route` invocations.
    """

    n_registrations: int = 0
    n_validations: int = 0
    n_failures: int = 0
    n_routes: int = 0

    @property
    def success_rate(self) -> float:
        """Fraction of validations that passed (no missing required params)."""
        if self.n_validations == 0:
            return 0.0
        return (self.n_validations - self.n_failures) / self.n_validations


# ---------------------------------------------------------------------------
# ToolSchemaCache
# ---------------------------------------------------------------------------


class ToolSchemaCache:
    """Pre-parsed tool schema registry with O(1) lookup by name or hash.

    Parameters
    ----------
    max_entries : int
        Maximum number of schemas that can be registered.  Attempting to
        exceed this limit raises a ``RuntimeError``.
    """

    def __init__(self, max_entries: int = 512) -> None:
        if max_entries < 1:
            raise ValueError(
                f"max_entries must be >= 1; got {max_entries}."
            )
        self._max_entries = max_entries
        self._by_name: Dict[str, ToolSchema] = {}
        self._by_hash: Dict[str, ToolSchema] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._n_registrations: int = 0
        self._n_validations: int = 0
        self._n_validation_failures: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hash(name: str, param_keys: List[str]) -> str:
        """Compute a deterministic hex hash from the tool name and sorted param keys."""
        raw = name + "|" + ",".join(sorted(param_keys))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def register(self, schema_dict: Dict[str, Any]) -> str:
        """Parse *schema_dict* into a :class:`ToolSchema` and cache it.

        Parameters
        ----------
        schema_dict : dict
            Must contain at least ``"name"`` and ``"parameters"`` keys.

        Returns
        -------
        str
            The hex schema hash (16 characters).

        Raises
        ------
        ValueError
            If required keys are missing or the schema is malformed.
        RuntimeError
            If the cache is full (``max_entries`` reached).
        """
        if "name" not in schema_dict:
            raise ValueError("schema_dict must contain a 'name' key.")
        if "parameters" not in schema_dict:
            raise ValueError("schema_dict must contain a 'parameters' key.")

        name: str = str(schema_dict["name"])
        parameters: Dict[str, Any] = dict(schema_dict["parameters"])
        description: str = str(schema_dict.get("description", ""))
        handler_id: str = str(schema_dict.get("handler_id", ""))

        if not name:
            raise ValueError("Tool name must not be empty.")

        schema_hash = self._make_hash(name, list(parameters.keys()))

        if schema_hash in self._by_hash:
            # Already registered — return existing hash idempotently.
            return schema_hash

        if len(self._by_name) >= self._max_entries:
            raise RuntimeError(
                f"ToolSchemaCache is full ({self._max_entries} entries).  "
                "Increase max_entries or evict old schemas."
            )

        schema = ToolSchema(
            name=name,
            parameters=parameters,
            description=description,
            handler_id=handler_id,
            schema_hash=schema_hash,
        )
        self._by_name[name] = schema
        self._by_hash[schema_hash] = schema
        self._n_registrations += 1
        return schema_hash

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ToolSchema]:
        """Look up a schema by tool name.

        Parameters
        ----------
        name : str
            Tool name to search for.

        Returns
        -------
        ToolSchema or None
            The registered schema, or ``None`` if not found.
        """
        schema = self._by_name.get(name)
        if schema is not None:
            self._hits += 1
        else:
            self._misses += 1
        return schema

    def get_by_hash(self, schema_hash: str) -> Optional[ToolSchema]:
        """Look up a schema by its hash.

        Parameters
        ----------
        schema_hash : str
            16-character hex hash as returned by :meth:`register`.

        Returns
        -------
        ToolSchema or None
        """
        schema = self._by_hash.get(schema_hash)
        if schema is not None:
            self._hits += 1
        else:
            self._misses += 1
        return schema

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check that *arguments* satisfies the schema for *tool_name*.

        All keys in the schema's ``parameters`` dict are treated as required.
        Extra keys in *arguments* are permitted (they are simply ignored).

        Parameters
        ----------
        tool_name : str
            Name of the tool being called.
        arguments : dict
            Arguments supplied by the model.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` on success, or ``(False, error_message)`` on
            failure.
        """
        self._n_validations += 1
        schema = self._by_name.get(tool_name)
        if schema is None:
            self._n_validation_failures += 1
            return False, f"Unknown tool '{tool_name}': not registered in cache."

        missing = [p for p in schema.required_params if p not in arguments]
        if missing:
            self._n_validation_failures += 1
            return (
                False,
                f"Tool '{tool_name}' is missing required parameters: "
                + ", ".join(missing),
            )
        return True, ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cached(self) -> int:
        """Number of schemas currently registered in the cache."""
        return len(self._by_name)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of lookups (get / get_by_hash) served from cache."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> ToolCacheStats:
        """Return aggregate cache statistics."""
        return ToolCacheStats(
            n_registrations=self._n_registrations,
            n_validations=self._n_validations,
            n_failures=self._n_validation_failures,
            n_routes=0,  # populated by ToolRouter
        )


# ---------------------------------------------------------------------------
# ToolRouter
# ---------------------------------------------------------------------------


class ToolRouter:
    """Routes validated tool calls to the appropriate Python handler.

    Parameters
    ----------
    cache : ToolSchemaCache
        The schema cache used to validate calls before routing.
    """

    def __init__(self, cache: ToolSchemaCache) -> None:
        self._cache = cache
        self._n_routes: int = 0
        self._n_validation_failures: int = 0

    def route(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        handlers: Dict[str, Callable[..., Any]],
    ) -> Any:
        """Validate and dispatch a tool call.

        Parameters
        ----------
        tool_name : str
            Name of the tool to invoke.
        arguments : dict
            Arguments to pass to the handler.
        handlers : dict
            Mapping of tool names to callables.  Each callable must accept
            a single ``dict`` argument.

        Returns
        -------
        any
            The return value of the handler.

        Raises
        ------
        KeyError
            If *tool_name* is not present in *handlers*.
        ValueError
            If the call fails schema validation.
        """
        valid, error_msg = self._cache.validate_call(tool_name, arguments)
        if not valid:
            self._n_validation_failures += 1
            raise ValueError(
                f"Tool call validation failed for '{tool_name}': {error_msg}"
            )

        if tool_name not in handlers:
            raise KeyError(
                f"No handler registered for tool '{tool_name}'.  "
                f"Available handlers: {sorted(handlers.keys())}"
            )

        self._n_routes += 1
        return handlers[tool_name](arguments)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_routes(self) -> int:
        """Total number of successful :meth:`route` calls."""
        return self._n_routes

    @property
    def n_validation_failures(self) -> int:
        """Total number of validation failures encountered during routing."""
        return self._n_validation_failures

    def stats(self) -> ToolCacheStats:
        """Return aggregate routing statistics merged with cache stats."""
        cache_s = self._cache.stats()
        return ToolCacheStats(
            n_registrations=cache_s.n_registrations,
            n_validations=cache_s.n_validations,
            n_failures=cache_s.n_failures,
            n_routes=self._n_routes,
        )
