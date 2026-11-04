"""Benchmark contracts, without a universal agent, action format or model client."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, NotRequired, Protocol, TypedDict


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    dataset_source: str
    split: str


class ArtifactReference(TypedDict):
    """A stored observation/output file, not a container image or base64 payload."""

    path: str
    media_type: str
    metadata: NotRequired[dict[str, Any]]  # e.g. width, height, timestamp, sha256


class Observation(TypedDict):
    """Optional record convention; the benchmark owns actual runtime serialization."""

    kind: Literal["text", "image", "state", "tool_result"]
    text: NotRequired[str]
    artifact: NotRequired[ArtifactReference]
    data: NotRequired[Any]  # public structured state or tool result only


class Verification(TypedDict):
    score: float | None
    training_reward: float | None
    details: dict[str, Any]  # trusted evaluation record; never an agent observation


class EpisodeResult(TypedDict):
    """Common recording fields; adapters may retain benchmark-specific fields."""

    task_id: str
    dataset_source: str
    split: str
    policy_checkpoint: str
    harness_revision: str
    sampling_seed: int
    sampling_params: dict[str, Any]
    trajectory: str
    termination_reason: str | None
    error_type: str | None
    score: float | None
    training_reward: float | None
    reward: dict[str, Any]
    tokens: int | None
    tool_calls: int | None
    latency_seconds: float | None
    simulated: bool
    benchmark: NotRequired[str]
    artifacts: NotRequired[list[ArtifactReference]]


class BenchmarkAdapter(Protocol):
    """Trusted host code. Opaque task records may contain private grader data.

    Only public_task() output may be passed to the agent. The adapter owns its
    runtime/environment, cleanup, failure records and verifier; raw tasks and
    Verification.details must never be serialized into agent-visible traces.
    Existing SWE task dictionaries and public projections remain unchanged.
    """

    name: str

    def load_tasks(self, config: dict, role: str, limit: int) -> list[Any]: ...

    def task_info(self, task: Any) -> TaskInfo: ...

    def public_task(self, task: Any) -> Mapping[str, Any]: ...

    def verify(self, task: Any, episode: Mapping, runtime_config: dict, output: Path) -> Verification: ...

    def run_episode(
        self, client: Any, task: Any, checkpoint: str, harness: str, config: dict, output: Path, seed: int
    ) -> EpisodeResult:
        """Run using public inputs, verify separately, return a normalized result.

        Persist runtime-specific trajectory/failure records under output. On
        infrastructure failure raise rather than inventing a negative reward.
        execution.rollout() persists the successful result, adding benchmark ID.
        """
        ...
