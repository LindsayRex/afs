#!/usr/bin/env python3
"""
Minimal multi-core sanity test for Windows: spins up N processes, each doing
CPU work for D seconds, and reports speedup vs. serial time.

Usage (PowerShell):
  "J:/Google Drive/Software/msFlow/msflow_env/Scripts/python.exe" "j:/Google Drive/Software/msFlow/scripts/mp_sanity.py" --workers ([Environment]::ProcessorCount) --duration 3

You should see ~`workers` python.exe processes and wall-clock time close to
`duration` seconds (not workers*duration) if parallelism is working.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def cpu_burn(seconds: float) -> float:
    """Busy-loop floating point ops for approximately `seconds` seconds."""
    t0 = time.perf_counter()
    x = 0.0
    while (time.perf_counter() - t0) < seconds:
        # A couple of cheap transcendentals to keep the core busy
        x = math.sin(x + 0.1234567) * math.cos(x + 0.7654321)
    return time.perf_counter() - t0


def _worker(seconds: float) -> tuple[int, float]:
    elapsed = cpu_burn(seconds)
    return (os.getpid(), elapsed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Number of processes to launch")
    ap.add_argument("--duration", type=float, default=3.0, help="Per-process CPU burn duration (seconds)")
    args = ap.parse_args()

    workers = max(1, int(args.workers))
    duration = float(args.duration)

    print(f"Starting {workers} workers; per-process duration ~{duration:.2f}s")
    t0 = time.perf_counter()
    results: list[tuple[int, float]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, duration) for _ in range(workers)]
        for fut in as_completed(futures):
            pid, elapsed = fut.result()
            results.append((pid, elapsed))
            print(f"worker pid={pid} ran {elapsed:.3f}s")

    wall = time.perf_counter() - t0
    serial = workers * duration
    speedup = serial / wall if wall > 0 else float("inf")
    uniq_pids = sorted({pid for pid, _ in results})

    print("\nSummary")
    print(f"  logical processors (os.cpu_count): {os.cpu_count()}")
    print(f"  workers launched: {workers}")
    print(f"  distinct worker PIDs: {len(uniq_pids)} -> {uniq_pids}")
    print(f"  wall time: {wall:.3f}s  |  serial time: {serial:.3f}s  |  speedup: {speedup:.2f}x")
    if speedup < 1.5 and workers > 1:
        print("  note: low speedup suggests multiprocessing isn’t engaging; check __main__ guard or antivirus/defender interference.")


if __name__ == "__main__":
    main()
