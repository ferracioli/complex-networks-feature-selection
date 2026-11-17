#!/usr/bin/env python3
"""Initialize any empty seed values, then launch main.py with hash seed set.

Copy this file to the repository root and run:
    python run_experiment.py

The launcher itself does not import the project. It persists any newly
initialized seeds to input/config.json, then starts a fresh Python process
whose PYTHONHASHSEED is set before that process initializes its interpreter.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "input" / "config.json"
CHECKER_CANDIDATES = (
    ROOT / "scripts" / "verify_reproducibility.py",
    ROOT / "verify_reproducibility.py",
)

# Keep these keys aligned with the top-level "seeds" object in input/config.json
# and REQUIRED_SEED_KEYS in verify_reproducibility.py.
SEED_DEFAULTS = {
    # These values reproduce the seeds already hard-coded in the repository
    # and the PYTHONHASHSEED recovered from the BraTS snapshot.
    "python_hash_seed": 1788688503,
    "numpy_global": 42,
    "outer_cv": 42,
    "inner_cv": 42,
    "fold_seed_base": 42,
    "inner_rf": 42,
    "random_state": 42
}


def write_json_atomically(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if not CONFIG_PATH.is_file():
        print(f"ERROR: missing config file: {CONFIG_PATH}")
        return 2

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seeds = config.setdefault("seeds", {})
    changed = False

    # Include future seed keys already listed in the config.
    all_seed_keys = sorted(set(SEED_DEFAULTS) | set(seeds))
    for key in all_seed_keys:
        known_default = SEED_DEFAULTS.get(key)
        value = seeds.get(key)
        if value is None or str(value).strip() == "":
            # Restore known historical values. For a future seed with no known
            # historical value, use a cryptographically random value once and
            # persist it so every later run reuses the same integer.
            if key in SEED_DEFAULTS:
                seeds[key] = known_default
            else:
                seeds[key] = secrets.randbelow(4_294_967_295) + 1
            print(f"Initialized missing seed '{key}' = {seeds[key]}")
            changed = True
        else:
            try:
                number = int(value)
            except (TypeError, ValueError):
                print(f"ERROR: seed '{key}' must be an integer, got {value!r}")
                return 2
            if number < 0 or number > 4_294_967_295:
                print(f"ERROR: seed '{key}' is outside the 32-bit range: {number}")
                return 2
            seeds[key] = number

    if changed:
        write_json_atomically(CONFIG_PATH, config)
        print(f"Persisted generated seed values in {CONFIG_PATH}")

    hash_seed = str(seeds["python_hash_seed"])
    import os
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed

    checker = next((path for path in CHECKER_CANDIDATES if path.is_file()), None)
    if checker is None:
        print("ERROR: verify_reproducibility.py not found in scripts/ or repository root.")
        return 2

    # BraTS is the first and only preflight with a supplied input-file SHA-256.
    print("Running BraTS Africa preflight before starting the benchmark...")
    preflight = subprocess.run(
        [sys.executable, str(checker), "--preflight", "brats_africa"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if preflight.returncode != 0:
        print("Preflight failed. main.py was not started.")
        return preflight.returncode

    print(f"Launching main.py with PYTHONHASHSEED={hash_seed} ...")
    run = subprocess.run(
        [sys.executable, str(ROOT / "main.py")],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
