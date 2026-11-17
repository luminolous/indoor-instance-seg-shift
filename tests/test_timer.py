import csv
from pathlib import Path

import pytest

from iiss.utils.timer import LOG_COLUMNS, RunTimer, logged_seconds


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _timer(log: Path, clock: FakeClock, cap_hours: float | None = 1.0, **kw) -> RunTimer:
    return RunTimer("yolo_s640_seed42", cap_hours, log, clock=clock, track_cuda=False, **kw)


def test_completed_run_logs_one_row(tmp_path: Path):
    log, clock = tmp_path / "gpu_time_log.csv", FakeClock()
    with _timer(log, clock) as timer:
        clock.advance(120)
        assert not timer.should_stop()
        assert timer.remaining() == pytest.approx(3600 - 120)
    rows = _rows(log)
    assert list(rows[0].keys()) == list(LOG_COLUMNS)
    assert len(rows) == 1
    assert rows[0]["wall_clock_s"] == "120.0"
    assert rows[0]["cap_hit"] == "False"
    assert rows[0]["status"] == "completed"


def test_cap_hit_runs_callbacks_once(tmp_path: Path):
    log, clock = tmp_path / "log.csv", FakeClock()
    calls: list[str] = []
    with _timer(log, clock, cap_hours=0.5) as timer:
        timer.add_cap_callback(lambda: calls.append("save"))
        clock.advance(1799)
        assert not timer.should_stop()
        clock.advance(1)
        assert timer.should_stop()
        assert timer.should_stop()
    assert calls == ["save"]
    row = _rows(log)[0]
    assert row["cap_hit"] == "True"
    assert row["status"] == "cap_hit"


def test_resume_subtracts_logged_time(tmp_path: Path):
    log, clock = tmp_path / "log.csv", FakeClock()
    with _timer(log, clock):
        clock.advance(3000)
    with _timer(log, clock) as timer:
        assert timer.prior_seconds == pytest.approx(3000)
        clock.advance(599)
        assert not timer.should_stop()
        clock.advance(1)
        assert timer.should_stop()
    rows = _rows(log)
    assert len(rows) == 2  # header written only once
    assert logged_seconds(log, "yolo_s640_seed42") == pytest.approx(3600)
    assert logged_seconds(log, "other_run") == 0


def test_resume_false_ignores_previous_sessions(tmp_path: Path):
    log, clock = tmp_path / "log.csv", FakeClock()
    with _timer(log, clock):
        clock.advance(3500)
    with _timer(log, clock, resume=False) as timer:
        assert timer.remaining() == pytest.approx(3600)


def test_no_cap_never_stops(tmp_path: Path):
    log, clock = tmp_path / "log.csv", FakeClock()
    with _timer(log, clock, cap_hours=None) as timer:
        clock.advance(10**6)
        assert not timer.should_stop()
        assert timer.remaining() == float("inf")


@pytest.mark.parametrize(
    ("exc", "status"), [(RuntimeError("boom"), "error"), (KeyboardInterrupt(), "interrupted")]
)
def test_exception_still_logs_row(tmp_path: Path, exc: BaseException, status: str):
    log, clock = tmp_path / "log.csv", FakeClock()
    with pytest.raises(type(exc)), _timer(log, clock):
        clock.advance(42)
        raise exc
    row = _rows(log)[0]
    assert row["status"] == status
    assert row["wall_clock_s"] == "42.0"
