# squish/benchmarks/base.py
"""Base classes and shared utilities for the Squish benchmark suite."""
from __future__ import annotations

__all__ = [
    "BenchmarkRunner",
    "EngineClient",
    "EngineConfig",
    "ResultRecord",
    "SQUISH_ENGINE",
    "OLLAMA_ENGINE",
    "LMSTUDIO_ENGINE",
    "MLXLM_ENGINE",
    "LLAMACPP_ENGINE",
    "ENGINE_REGISTRY",
    "parse_engines",
]

import abc
import datetime
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Connection parameters for an OpenAI-compatible inference engine."""
    name: str
    base_url: str
    api_key: str = "squish"
    timeout: float = 120.0

    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    def health_url(self) -> str:
        return self.base_url.rstrip("/") + "/health"

    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/models"


SQUISH_ENGINE   = EngineConfig("squish",   "http://localhost:11434")
OLLAMA_ENGINE   = EngineConfig("ollama",   "http://localhost:11434")
LMSTUDIO_ENGINE = EngineConfig("lmstudio", "http://localhost:1234")
MLXLM_ENGINE    = EngineConfig("mlxlm",    "http://localhost:8080")
LLAMACPP_ENGINE = EngineConfig("llamacpp", "http://localhost:8080")

ENGINE_REGISTRY: Dict[str, EngineConfig] = {
    "squish":   SQUISH_ENGINE,
    "ollama":   OLLAMA_ENGINE,
    "lmstudio": LMSTUDIO_ENGINE,
    "mlxlm":    MLXLM_ENGINE,
    "llamacpp": LLAMACPP_ENGINE,
}


def parse_engines(spec: str) -> List[EngineConfig]:
    """Parse a comma-separated engine spec like 'squish,ollama' into EngineConfig list."""
    names = [n.strip() for n in spec.split(",") if n.strip()]
    result = []
    for name in names:
        if name in ENGINE_REGISTRY:
            result.append(ENGINE_REGISTRY[name])
        else:
            # Custom URL in name=url form
            if "=" in name:
                n, url = name.split("=", 1)
                result.append(EngineConfig(n.strip(), url.strip()))
            else:
                raise ValueError(f"Unknown engine {name!r}. Known: {list(ENGINE_REGISTRY)}")
    return result


# ---------------------------------------------------------------------------
# ResultRecord
# ---------------------------------------------------------------------------

@dataclass
class ResultRecord:
    """Structured benchmark result for one (track, engine, model) combination."""
    track: str
    engine: str
    model: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track":     self.track,
            "engine":    self.engine,
            "model":     self.model,
            "timestamp": self.timestamp,
            "metrics":   self.metrics,
            "metadata":  self.metadata,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ResultRecord":
        d = json.loads(Path(path).read_text())
        return cls(
            track=d["track"],
            engine=d["engine"],
            model=d["model"],
            timestamp=d.get("timestamp", ""),
            metrics=d.get("metrics", {}),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# EngineClient
# ---------------------------------------------------------------------------

class EngineClient:
    """Thin, stdlib-only OpenAI-compatible HTTP client.

    Uses only ``urllib`` so the benchmark suite has no additional dependencies.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def is_alive(self) -> bool:
        """Return True if the engine is reachable."""
        for url in (self.config.health_url(), self.config.models_url()):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )
                with urllib.request.urlopen(req, timeout=5):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        tools: Optional[list] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a chat completion request; return the parsed JSON response."""
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.config.chat_url(),
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                raw = resp.read()
                ttft = time.perf_counter() - t0
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Engine '{self.config.name}' at {self.config.chat_url()} unreachable: {exc}"
            ) from exc

        result = json.loads(raw)
        result["_ttft_s"] = ttft
        return result

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        """Yield (chunk_text, ttft_s, total_s) from an SSE streaming response."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.config.chat_url(),
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        first = True
        ttft = 0.0
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        if first:
                            ttft = time.perf_counter() - t0
                            first = False
                        yield delta, ttft, time.perf_counter() - t0
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Engine '{self.config.name}' at {self.config.chat_url()} unreachable: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# BenchmarkRunner ABC
# ---------------------------------------------------------------------------

class BenchmarkRunner(abc.ABC):
    """Abstract base class for all Squish benchmark tracks.

    Subclasses must implement :meth:`run` and :meth:`track_name`.
    """

    @property
    @abc.abstractmethod
    def track_name(self) -> str:
        """Short track identifier, e.g. ``"quality"``."""

    @abc.abstractmethod
    def run(
        self,
        engine: EngineConfig,
        model: str,
        *,
        limit: Optional[int] = None,
    ) -> ResultRecord:
        """Execute the benchmark and return a ResultRecord."""

    def output_path(self, engine: str, model: str, base_dir: str = "eval_output") -> Path:
        """Return the canonical output file path for this (engine, model) combination."""
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace("/", "_").replace(":", "_")
        fname = f"{self.track_name}_{safe_model}_{engine}_{ts}.json"
        return Path(base_dir) / fname
