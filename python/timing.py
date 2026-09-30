"""Opt-in nested wall-clock and CUDA-event timing for pydisort experiments.

Use :func:`timed` on functions whose contribution should be visible, and place
an experiment in a :class:`TimingCollector` context.  Without an active
collector the decorator calls the function directly.  CUDA timings use events
on the tensor's device and synchronize only when records or summaries are
requested, so instrumentation does not serialize each profiled call.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar, overload

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class TimingRecord:
    """One inclusive profiled call, in seconds."""

    name: str
    wall_seconds: float
    cuda_seconds: float | None


@dataclass
class _PendingRecord:
    name: str
    wall_seconds: float
    device: torch.device | None
    start: Any | None
    end: Any | None


def _cuda_device(value: object) -> torch.device | None:
    """Find the first CUDA tensor device in nested call arguments."""
    if isinstance(value, torch.Tensor):
        return value.device if value.is_cuda else None
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            device = _cuda_device(getattr(value, field.name))
            if device is not None:
                return device
    if isinstance(value, Mapping):
        for item in value.values():
            device = _cuda_device(item)
            if device is not None:
                return device
    elif isinstance(value, (tuple, list)):
        for item in value:
            device = _cuda_device(item)
            if device is not None:
                return device
    return None


_active_collector: ContextVar[TimingCollector | None] = ContextVar(
    "pydisort_active_timing_collector", default=None
)


class TimingCollector(AbstractContextManager["TimingCollector"]):
    """Collect inclusive timings from :func:`timed` calls.

    Args:
        cuda_device: Default CUDA device for calls with no CUDA tensor argument.
            Set this for functions that allocate their CUDA tensors internally.
    """

    def __init__(self, cuda_device: torch.device | str | None = None) -> None:
        self._default_cuda_device = (
            torch.device(cuda_device) if cuda_device is not None else None
        )
        if (
            self._default_cuda_device is not None
            and self._default_cuda_device.type != "cuda"
        ):
            raise ValueError("cuda_device must name a CUDA device")
        self._pending: list[_PendingRecord] = []
        self._token: Token[TimingCollector | None] | None = None

    def __enter__(self) -> TimingCollector:  # noqa: PYI034
        if self._token is not None:
            raise RuntimeError("a TimingCollector cannot be entered twice")
        self._token = _active_collector.set(self)
        return self

    def __exit__(self, *args: object) -> None:
        if self._token is None:
            return
        _active_collector.reset(self._token)
        self._token = None

    def measure(
        self, name: str, function: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """Run ``function`` and append its inclusive timing record."""
        device = _cuda_device((args, kwargs)) or self._default_cuda_device
        start_event = end_event = None
        if device is not None:
            with torch.cuda.device(device):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            wall_seconds = perf_counter() - started
            if end_event is not None:
                with torch.cuda.device(device):
                    end_event.record()
            self._pending.append(
                _PendingRecord(
                    name, wall_seconds, device, start_event, end_event
                )
            )

    def records(self) -> tuple[TimingRecord, ...]:
        """Synchronize recorded CUDA work once per device and return all calls."""
        devices = {
            record.device
            for record in self._pending
            if record.device is not None
        }
        for device in devices:
            torch.cuda.synchronize(device)
        return tuple(
            TimingRecord(
                name=record.name,
                wall_seconds=record.wall_seconds,
                cuda_seconds=(
                    record.start.elapsed_time(record.end) / 1.0e3
                    if record.start is not None and record.end is not None
                    else None
                ),
            )
            for record in self._pending
        )

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        """Return per-function inclusive counts and accumulated timings."""
        grouped: dict[str, list[TimingRecord]] = defaultdict(list)
        for record in self.records():
            grouped[record.name].append(record)
        result: dict[str, dict[str, float | int | None]] = {}
        for name, records in grouped.items():
            cuda_values = [record.cuda_seconds for record in records]
            result[name] = {
                "calls": len(records),
                "wall_seconds": sum(record.wall_seconds for record in records),
                "cuda_seconds": (
                    sum(value for value in cuda_values if value is not None)
                    if any(value is not None for value in cuda_values)
                    else None
                ),
            }
        return result

    def append_text_report(
        self, path: Path, *, case: str, backend: str
    ) -> None:
        """Append this collector's summary to a stable human-readable log."""
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write(
                    "# Incremental tensor-backend timing log\n"
                    "# timestamp_utc | case | backend | function | calls | "
                    "wall_seconds | cuda_seconds\n"
                )
            timestamp = datetime.now(UTC).isoformat()
            for name, values in self.summary().items():
                cuda_seconds = values["cuda_seconds"]
                cuda_text = (
                    "-" if cuda_seconds is None else f"{cuda_seconds:.9f}"
                )
                handle.write(
                    f"{timestamp} | {case} | {backend} | {name} | "
                    f"{values['calls']} | {values['wall_seconds']:.9f} | "
                    f"{cuda_text}\n"
                )


@overload
def timed(
    function: Callable[..., T], *, name: str | None = None
) -> Callable[..., T]:
    ...


@overload
def timed(
    function: None = None, *, name: str | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    ...


def timed(
    function: Callable[..., Any] | None = None, *, name: str | None = None
) -> Any:
    """Time a function when called inside a :class:`TimingCollector` context.

    It supports both ``@timed`` and ``@timed(name="layer_setup")``.
    """

    def decorate(target: Callable[..., T]) -> Callable[..., T]:
        timing_name = name or target.__qualname__

        @wraps(target)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            collector = _active_collector.get()
            if collector is None:
                return target(*args, **kwargs)
            return collector.measure(timing_name, target, *args, **kwargs)

        return wrapped

    if function is None:
        return decorate
    return decorate(function)
