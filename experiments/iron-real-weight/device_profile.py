"""Opt-in wall-clock accounting around IRON host API calls."""

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


_active = None


class DeviceProfile:
    def __init__(self):
        self.samples = defaultdict(lambda: [0, 0.0])

    def record(self, name, duration):
        entry = self.samples[name]
        entry[0] += 1
        entry[1] += duration

    def report(self):
        accounted = sum(seconds for _, seconds in self.samples.values())
        print("Instrumented IRON host-call wall time (not device-only time):")
        for name, (count, seconds) in sorted(self.samples.items(), key=lambda item: -item[1][1]):
            print(f"  {name}: {count} calls, {seconds:.3f}s total, {1000 * seconds / count:.2f}ms/call")
        print(f"  accounted calls: {accounted:.3f}s; excludes buffer allocation and Python/model work")


@contextmanager
def profiling(enabled):
    global _active
    if not enabled:
        yield None
        return
    if _active is not None:
        raise RuntimeError("An IRON profile is already active")
    profile = DeviceProfile()
    _active = profile
    try:
        yield profile
    finally:
        _active = None


def call(name, function, *args, **kwargs):
    if _active is None:
        return function(*args, **kwargs)
    started = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        _active.record(name, perf_counter() - started)


def readback(tensor):
    return call("tensor readback", tensor.numpy)
