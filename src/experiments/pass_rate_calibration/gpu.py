import fcntl
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from src.experiments.pass_rate_calibration.config import STATE_ROOT
from src.experiments.utkface.scheduler import gpu_inventory, gpu_is_free

CLAIMS = STATE_ROOT / "gpu-claims"


@contextmanager
def claimed_gpus(*, indices: tuple[int, ...]) -> Iterator[list[dict]]:
    inventory = {gpu["index"]: gpu for gpu in gpu_inventory()}
    CLAIMS.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        claimed = []
        for index in indices:
            gpu = inventory[index]
            if not gpu_is_free(gpu=gpu):
                raise RuntimeError(f"GPU {index} is occupied: {gpu}")
            handle = stack.enter_context((CLAIMS / f"{gpu['uuid']}.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claimed.append(gpu)
        yield claimed


def claimable(*, gpu: dict) -> bool:
    if not gpu_is_free(gpu=gpu):
        return False
    CLAIMS.mkdir(parents=True, exist_ok=True)
    with (CLAIMS / f"{gpu['uuid']}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
    return True


def wait_for_gpus(*, count: int, poll_seconds: float) -> tuple[int, ...]:
    while True:
        free = [gpu["index"] for gpu in gpu_inventory() if claimable(gpu=gpu)]
        if len(free) >= count:
            return tuple(free[:count])
        time.sleep(poll_seconds)


def visible_devices(*, gpus: list[dict]) -> str:
    return ",".join(gpu["uuid"] for gpu in gpus)
