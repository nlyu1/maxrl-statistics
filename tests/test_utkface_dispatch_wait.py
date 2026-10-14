import subprocess
import time

from src.experiments.utkface.scheduler import wait_for_progress


def test_owned_process_exit_wakes_dispatcher_before_poll_timeout() -> None:
    with subprocess.Popen(["/bin/sleep", "0.05"]) as child:
        started = time.monotonic()
        wait_for_progress(active=[{"attempts": [{"pid": child.pid}]}], timeout=5)
        assert time.monotonic() - started < 2
        child.wait()
