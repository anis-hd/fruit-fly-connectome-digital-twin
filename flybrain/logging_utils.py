"""stdout/stderr tee to sim.log. Extracted from server.py lines 16-63."""
import os
import sys
from datetime import datetime


def setup_file_logging(log_name: str = "sim.log", log_dir: str | None = None):
    """Mirror stdout+stderr to sim.log, line-buffered.

    Defaults to the project root (parent of the ``flybrain`` package,
    i.e. where server.py lives) — matching the original
    ``os.path.dirname(os.path.abspath(__file__))`` behaviour even when
    launched via ``uvicorn server:app`` (where sys.argv[0] is uvicorn).
    Returns the log path. Never raises — falls back to console-only.
    """
    if log_dir is None:
        log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(log_dir, log_name)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def writelines(self, lines):
            for line in lines:
                self.write(line)

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

        def isatty(self):
            try:
                return self.streams[0].isatty()
            except Exception:
                return False

        def fileno(self):
            return self.streams[0].fileno()

        @property
        def encoding(self):
            return getattr(self.streams[0], "encoding", "utf-8")

    try:
        log_file = open(log_path, "a", buffering=1)
        sys.stdout = _Tee(sys.stdout, log_file)
        sys.stderr = _Tee(sys.stderr, log_file)
        print(
            f"\n{'=' * 70}\nSESSION {datetime.now().isoformat(timespec='seconds')} "
            f"| logging to {log_path}\n{'=' * 70}",
            flush=True,
        )
    except Exception as e:  # pragma: no cover - logging must never crash boot
        print(f"[log] file logging disabled ({e})")
    return log_path
