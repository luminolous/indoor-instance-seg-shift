import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from iiss.utils.timer import LOG_COLUMNS, RunTimer, heartbeat_path, logged_seconds


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


def test_clean_exit_removes_heartbeat(tmp_path: Path):
    log = tmp_path / "gpu_time_log.csv"
    clock = FakeClock()
    with _timer(log, clock) as timer:
        hb = heartbeat_path(log, timer.run_id)
        assert hb.is_file()
        clock.t += 40
        timer.beat()
        assert json.loads(hb.read_text())["wall_clock_s"] == 40.0
    assert not hb.exists()
    assert logged_seconds(log, "yolo_s640_seed42") == 40.0


def test_leftover_heartbeat_is_logged_as_killed_and_counts(tmp_path: Path):
    log = tmp_path / "gpu_time_log.csv"
    clock = FakeClock()
    timer = _timer(log, clock)
    timer.__enter__()  # simulate a hard kill: __exit__ never runs
    clock.t += 1200
    timer.beat()
    timer._hb_stop.set()

    resumed = _timer(log, FakeClock(), resume=True)
    assert resumed.recovered["status"] == "killed"
    assert resumed.prior_seconds == 1200.0
    assert not heartbeat_path(log, "yolo_s640_seed42").exists()
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert [r["status"] for r in rows] == ["killed"]
    assert _timer(log, FakeClock()).recovered is None  # recovered only once


KILL_SCRIPT = """
import sys, time
from iiss.utils.timer import RunTimer
with RunTimer("r", 1.0, sys.argv[1], track_cuda=False, heartbeat_s=0.2):
    print("started", flush=True)
    time.sleep(60)
"""


def test_hard_kill_of_a_real_process_is_recovered(tmp_path: Path):
    log = tmp_path / "gpu_time_log.csv"
    proc = subprocess.Popen(
        [sys.executable, "-c", KILL_SCRIPT, str(log)], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout.readline().strip() == "started"
    time.sleep(1.5)
    proc.kill()  # TerminateProcess on Windows, SIGKILL elsewhere: no cleanup runs
    proc.wait(timeout=10)
    assert not log.exists()  # nothing was logged by the killed process itself
    seconds = RunTimer("r", 1.0, log, track_cuda=False, heartbeat_s=None).prior_seconds
    assert 1.0 <= seconds <= 2.5
