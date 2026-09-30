"""Tests for the opt-in Python/CUDA reconstruction timing utility."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
import timing as TIMING


@TIMING.timed(name="inner")
def inner(value):
    return value + 1


@TIMING.timed
def outer(value):
    return inner(value) * 2


def test_decorator_is_transparent_without_a_collector():
    assert outer(2) == 6


def test_collector_records_nested_cpu_calls():
    with TIMING.TimingCollector() as collector:
        assert outer(2) == 6

    summary = collector.summary()
    assert summary["inner"]["calls"] == 1
    assert summary["outer"]["calls"] == 1
    assert summary["inner"]["wall_seconds"] >= 0.0
    assert summary["outer"]["wall_seconds"] >= 0.0
    assert summary["inner"]["cuda_seconds"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_collector_records_cuda_event_time_without_per_call_sync():
    @TIMING.timed(name="cuda_add")
    def cuda_add(value):
        return value + 1.0

    with TIMING.TimingCollector() as collector:
        result = cuda_add(torch.ones(128, device="cuda", dtype=torch.float64))

    assert result.is_cuda
    record = collector.records()[0]
    assert record.name == "cuda_add"
    assert record.cuda_seconds is not None
    assert record.cuda_seconds >= 0.0


def test_collector_appends_plain_text_report(tmp_path):
    with TIMING.TimingCollector() as collector:
        inner(1)

    path = tmp_path / "tensor_timing.txt"
    collector.append_text_report(path, case="tp9", backend="cpu")
    content = path.read_text()
    assert "timestamp_utc | case | backend" in content
    assert "| tp9 | cpu | inner | 1 |" in content
