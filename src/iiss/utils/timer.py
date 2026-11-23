"""Training wall-clock timer with a per-run time cap and an append-only GPU time log.

Usage inside a model wrapper::

    with RunTimer(run_id, cap_hours=cfg["time_cap_hours"], log_path=gpu_time_log) as timer:
        timer.add_cap_callback(lambda: save_checkpoint("last"))
        for epoch in ...:
            for batch in ...:
                ...
                if timer.should_stop():
                    break  # stop gracefully; the cap callbacks have already run

The cap is **cumulative per run id**: when a run is resumed (crash, laptop sleep), the time
already logged for that run id in ``gpu_time_log.csv`` is subtracted from the budget.

Each ``with`` block appends exactly one row to the log, also when it ends in an exception::

    run_id,start,end,wall_clock_s,gpu_name,peak_vram_mb,cap_hit,status

``peak_vram_mb`` is ``torch.cuda.max_memory_reserved()`` (closest to what ``nvidia-smi``
shows). ``status`` is ``completed``, ``cap_hit``, ``interrupted``, ``error`` or ``killed``.

Hard kills (power loss, Task Manager, crash of the Python process) skip ``__exit__``, so no row
would be written and that time would silently not count. To prevent this, a background thread
rewrites a heartbeat file ``<log dir>/<log stem>.open/<run_id>.json`` every ``heartbeat_s``
seconds while the block runs; a clean exit deletes it. When a new ``RunTimer`` for the same run
id finds a leftover heartbeat, it first appends a ``killed`` row with the session time recorded
in it (at most ``heartbeat_s`` seconds are lost per kill), so the cap stays cumulative.

Note: elapsed time uses ``time.monotonic()``. Keep the laptop awake during training: time
spent suspended may count towards the cap, depending on the OS clock.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)

LOG_COLUMNS = (
    "run_id",
    "start",
    "end",
    "wall_clock_s",
    "gpu_name",
    "peak_vram_mb",
    "cap_hit",
    "status",
)


def logged_seconds(log_path: str | Path, run_id: str) -> float:
    """Sum ``wall_clock_s`` of all rows for ``run_id`` in the log (0 if the file is missing)."""
    log_path = Path(log_path)
    if not log_path.is_file():
        return 0.0
    total = 0.0
    with log_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_id") == run_id:
                total += float(row["wall_clock_s"])
    return total


def heartbeat_path(log_path: str | Path, run_id: str) -> Path:
    """Heartbeat file of an open session of ``run_id`` (see the module docstring)."""
    log_path = Path(log_path)
    return log_path.parent / f"{log_path.stem}.open" / f"{run_id}.json"


def _append_row(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.is_file() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def recover_killed_session(log_path: str | Path, run_id: str) -> dict[str, Any] | None:
    """Log a leftover heartbeat of ``run_id`` as a ``killed`` row and delete it.

    Returns the row written, or None if there was no leftover heartbeat.
    """
    hb = heartbeat_path(log_path, run_id)
    if not hb.is_file():
        return None
    try:
        beat = json.loads(hb.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable heartbeat %s (%s); time of that session is lost", hb, exc)
        hb.unlink(missing_ok=True)
        return None
    row = {k: beat.get(k) for k in LOG_COLUMNS}
    row.update(run_id=run_id, status="killed")
    _append_row(Path(log_path), row)
    hb.unlink(missing_ok=True)
    logger.warning(
        "Previous session of %s (pid %s, started %s) ended without logging; "
        "recorded %s s as status=killed",
        run_id, beat.get("pid"), row["start"], row["wall_clock_s"],
    )  # fmt: skip
    return row


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _cuda_device_name() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return "cpu"


class RunTimer:
    """Context manager that enforces a wall-clock cap and logs GPU time for one run session.

    Args:
        run_id: Run identifier, e.g. ``yolo_s640_seed42``.
        cap_hours: Total training budget for this run id. ``None`` disables the cap.
        log_path: CSV file to append to (created with a header if missing).
        resume: If True, subtract time already logged for ``run_id`` from the budget.
        clock: Monotonic clock in seconds. Injectable for tests.
        track_cuda: If True, reset and read CUDA peak-memory statistics.
        heartbeat_s: Interval of the heartbeat file written during the block (``None`` or 0
            writes it only on entry).
    """

    def __init__(
        self,
        run_id: str,
        cap_hours: float | None,
        log_path: str | Path,
        *,
        resume: bool = True,
        clock: Callable[[], float] = time.monotonic,
        track_cuda: bool = True,
        heartbeat_s: float | None = 10.0,
    ) -> None:
        self.run_id = run_id
        self.cap_seconds = None if cap_hours is None else float(cap_hours) * 3600.0
        self.log_path = Path(log_path)
        self.recovered = recover_killed_session(self.log_path, run_id)
        self.prior_seconds = logged_seconds(self.log_path, run_id) if resume else 0.0
        self._clock = clock
        self._track_cuda = track_cuda
        self._callbacks: list[Callable[[], Any]] = []
        self._t0: float | None = None
        self._start_iso: str | None = None
        self.cap_hit = False
        self._heartbeat_s = heartbeat_s
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

    # -- context manager ------------------------------------------------------------------

    def __enter__(self) -> RunTimer:
        self._reset_cuda_peak()
        self._start_iso = _now_iso()
        self._t0 = self._clock()
        self.beat()
        if self._heartbeat_s:
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, name=f"heartbeat-{self.run_id}", daemon=True
            )
            self._hb_thread.start()
        if self.cap_seconds is not None:
            logger.info(
                "Timer started for %s: budget %.0f s, already used %.0f s, remaining %.0f s",
                self.run_id,
                self.cap_seconds,
                self.prior_seconds,
                self.remaining(),
            )
            if self.remaining() <= 0:
                logger.warning("Run %s has no time budget left before starting", self.run_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            status = "cap_hit" if self.cap_hit else "completed"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "error"
        self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join()
        row = self._write_row(status)
        heartbeat_path(self.log_path, self.run_id).unlink(missing_ok=True)
        logger.info(
            "Timer stopped for %s: %s s this session, status=%s, peak VRAM %s MB",
            self.run_id,
            row["wall_clock_s"],
            status,
            row["peak_vram_mb"],
        )

    # -- cap interface --------------------------------------------------------------------

    def add_cap_callback(self, fn: Callable[[], Any]) -> None:
        """Register a function called once, the first time the cap is reached."""
        self._callbacks.append(fn)

    def session_seconds(self) -> float:
        """Seconds elapsed in the current ``with`` block."""
        if self._t0 is None:
            return 0.0
        return self._clock() - self._t0

    def elapsed(self) -> float:
        """Total seconds used by this run id, including previous sessions."""
        return self.prior_seconds + self.session_seconds()

    def remaining(self) -> float:
        """Seconds left in the budget (``inf`` if there is no cap)."""
        if self.cap_seconds is None:
            return float("inf")
        return self.cap_seconds - self.elapsed()

    def should_stop(self) -> bool:
        """Return True once the cap is reached. Runs the cap callbacks on the first True."""
        if self.cap_hit:
            return True
        if self.cap_seconds is None or self.remaining() > 0:
            return False
        self.cap_hit = True
        logger.warning(
            "Time cap reached for %s after %.0f s total; stopping gracefully",
            self.run_id,
            self.elapsed(),
        )
        for fn in self._callbacks:
            fn()
        return True

    def beat(self) -> None:
        """Rewrite the heartbeat file with the current session time (atomic replace)."""
        hb = heartbeat_path(self.log_path, self.run_id)
        hb.parent.mkdir(parents=True, exist_ok=True)
        data = {**self._row("killed"), "pid": os.getpid()}
        tmp = hb.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(hb)

    # -- internals ------------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(self._heartbeat_s):
            try:
                self.beat()
            except Exception:  # never let the heartbeat kill training
                logger.exception("Heartbeat write failed for %s", self.run_id)

    def _reset_cuda_peak(self) -> None:
        if not self._track_cuda:
            return
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _peak_vram_mb(self) -> float | None:
        if not self._track_cuda:
            return None
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_reserved() / 2**20, 1)

    def _row(self, status: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start": self._start_iso,
            "end": _now_iso(),
            "wall_clock_s": round(self.session_seconds(), 1),
            "gpu_name": _cuda_device_name() if self._track_cuda else "n/a",
            "peak_vram_mb": self._peak_vram_mb(),
            "cap_hit": self.cap_hit,
            "status": status,
        }

    def _write_row(self, status: str) -> dict[str, Any]:
        row = self._row(status)
        _append_row(self.log_path, row)
        return row
