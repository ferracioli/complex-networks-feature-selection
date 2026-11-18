#!/usr/bin/env python3
"""Compare local experiment CSVs with the frozen GitHub baseline.

Recommended location in the repository: scripts/verify_reproducibility.py

Examples (run from the repository root):
    python scripts/verify_reproducibility.py --preflight brats_africa
    python scripts/verify_reproducibility.py --dataset brats_africa
    python scripts/verify_reproducibility.py --dataset four_class_nsclc
    python scripts/verify_reproducibility.py --dataset radiomics_lgg

The comparison is exact for scientific/result values. Wall-clock runtime fields
are deliberately excluded because they cannot be expected to reproduce exactly
across runs or machines. The DyGraFS inner-grid CSV is compared separately, so
its balanced-accuracy and feature-count values are still checked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

BASELINE_COMMIT = "a950faaa3edabd10a1c21550de66a2a7fa1bf16b"
BASELINE_ROOT = (
    "https://raw.githubusercontent.com/ferracioli/"
    f"complex-networks-feature-selection/{BASELINE_COMMIT}/"
)

DATASET_ORDER = ("brats_africa", "four_class_nsclc", "radiomics_lgg")

# Input fingerprint supplied with the BraTS Africa experiment metadata.
EXPECTED_INPUT_SHA256 = {
    "brats_africa": "20d07d581a87ef0428be53ed179e660020435edadb8b331cf44812d02d9efec5",
}

# Versions recorded in the supplied experiment metadata. Preflight fails on
# mismatch because package/version drift can change exact numerical outputs.
EXPECTED_PYTHON = "3.11.9"
EXPECTED_PACKAGES = {
    "numpy": "1.26.4",
    "pandas": "3.0.1",
    "matplotlib": "3.10.8",
    "networkx": "3.6.1",
    "scikit-learn": "1.8.0",
}

REQUIRED_SEED_KEYS = (
    "python_hash_seed",
    "numpy_global",
    "outer_cv",
    "inner_cv",
    "random_state",
)

# The outer benchmark CSV stores the inner-grid DataFrame as a formatted object
# cell. That object contains runtime values; the same grid is checked, in a
# machine-readable CSV, below. The separately stored grid's runtime column is
# also omitted because it is wall-clock measurement, not algorithmic output.
CSV_ARTIFACTS = (
    {
        "label": "benchmark results",
        "relative_path": "outputs/{dataset}/{dataset}_benchmark_results.csv",
        "ignored_columns": {"runtime_mean", "inner_grid_summary"},
    },
    {
        "label": "DyGraFS inner grid",
        "relative_path": "outputs/{dataset}/{dataset}_dygrafs_inner_grid.csv",
        "ignored_columns": {"runtime_mean"},
    },
)


def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = (here, here.parent)
    for candidate in candidates:
        if (candidate / "input" / "config.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot locate repository root. Place this file in the repository root "
        "or in its scripts/ directory (input/config.json must exist)."
    )


def read_config(root: Path) -> dict:
    path = root / "input" / "config.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_preflight(root: Path, dataset: str) -> bool:
    """Check seeds, PYTHONHASHSEED, recorded software versions, and input hash."""
    ok = True
    print(f"Preflight for {dataset} | baseline commit {BASELINE_COMMIT}")

    try:
        config = read_config(root)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read input/config.json: {exc}")
        return False

    seeds = config.get("seeds")
    if not isinstance(seeds, dict):
        print("FAIL: input/config.json has no top-level 'seeds' object.")
        return False

    missing = [
        key for key in REQUIRED_SEED_KEYS
        if key not in seeds or seeds[key] is None or str(seeds[key]).strip() == ""
    ]
    if missing:
        print("FAIL: missing/empty seed keys: " + ", ".join(missing))
        ok = False
    else:
        try:
            for key in REQUIRED_SEED_KEYS:
                value = int(seeds[key])
                if value < 0 or value > 4_294_967_295:
                    raise ValueError(f"{key}={value} is outside the supported 32-bit range")
            print("PASS: all required seed keys are present and valid integers.")
        except (TypeError, ValueError) as exc:
            print(f"FAIL: invalid seed value: {exc}")
            ok = False

    expected_hash_seed = str(seeds.get("python_hash_seed", ""))
    actual_hash_seed = os.environ.get("PYTHONHASHSEED")
    if actual_hash_seed != expected_hash_seed:
        print(
            "FAIL: PYTHONHASHSEED in this process is "
            f"{actual_hash_seed!r}; config requires {expected_hash_seed!r}. "
            "Launch through run_experiment.py so the variable is set before "
            "the benchmark interpreter starts."
        )
        ok = False
    else:
        print(f"PASS: PYTHONHASHSEED={actual_hash_seed}.")

    actual_python = ".".join(map(str, sys.version_info[:3]))
    if actual_python != EXPECTED_PYTHON:
        print(f"FAIL: Python is {actual_python}; metadata recorded {EXPECTED_PYTHON}.")
        ok = False
    else:
        print(f"PASS: Python {actual_python}.")

    for package, expected_version in EXPECTED_PACKAGES.items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual_version = "not installed"
        if actual_version != expected_version:
            print(
                f"FAIL: {package} is {actual_version}; "
                f"metadata recorded {expected_version}."
            )
            ok = False
        else:
            print(f"PASS: {package} {actual_version}.")

    expected_hash = EXPECTED_INPUT_SHA256.get(dataset)
    if expected_hash:
        input_path = root / "outputs" / dataset / f"{dataset}_radiomic_features.csv"
        if not input_path.is_file():
            print(f"FAIL: expected input CSV does not exist: {input_path}")
            ok = False
        else:
            actual_hash = sha256_file(input_path)
            if actual_hash != expected_hash:
                print(
                    f"FAIL: SHA-256 mismatch for {input_path}\n"
                    f"      expected: {expected_hash}\n"
                    f"      actual:   {actual_hash}"
                )
                ok = False
            else:
                print(f"PASS: BraTS Africa radiomic input SHA-256 matches metadata.")
    else:
        print(f"INFO: no input SHA-256 was supplied for {dataset}; skipped input hash check.")

    if ok:
        print("\nPREFLIGHT PASSED. It is safe to start this dataset's benchmark.")
    else:
        print("\nPREFLIGHT FAILED. Do not start the 48-hour experiment yet.")
    return ok


def download_text(relative_path: str) -> str:
    url = BASELINE_ROOT + relative_path
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "complex-networks-feature-selection-repro-check/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not retrieve {url}: {exc}") from exc

    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Remote CSV is not UTF-8: {url}") from exc


def parse_csv(text: str, source_name: str) -> tuple[list[str], list[list[str]]]:
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise RuntimeError(f"Could not parse CSV {source_name}: {exc}") from exc
    if not rows:
        raise RuntimeError(f"CSV is empty: {source_name}")
    header = rows[0]
    if len(set(header)) != len(header):
        raise RuntimeError(f"CSV has duplicate column names: {source_name}")
    data = rows[1:]
    for number, row in enumerate(data, start=2):
        if len(row) != len(header):
            raise RuntimeError(
                f"Malformed CSV {source_name}, row {number}: "
                f"expected {len(header)} cells, found {len(row)}"
            )
    return header, data


def cells_equal(left: str, right: str) -> bool:
    """Compare numeric cells by exact Decimal value, text cells byte-for-byte."""
    if left == right:
        return True
    try:
        left_number = Decimal(left.strip())
        right_number = Decimal(right.strip())
    except (InvalidOperation, ValueError):
        return False
    if left_number.is_nan() or right_number.is_nan():
        return left_number.is_nan() and right_number.is_nan()
    return left_number == right_number


def compare_csv(
    local_path: Path,
    remote_text: str,
    ignored_columns: set[str],
    label: str,
) -> bool:
    if not local_path.is_file():
        print(f"FAIL: local {label} file is missing: {local_path}")
        return False

    try:
        local_header, local_rows = parse_csv(
            local_path.read_text(encoding="utf-8-sig"), str(local_path)
        )
        remote_header, remote_rows = parse_csv(remote_text, f"GitHub:{label}")
    except (OSError, RuntimeError) as exc:
        print(f"FAIL: {exc}")
        return False

    local_keep = [i for i, name in enumerate(local_header) if name not in ignored_columns]
    remote_keep = [i for i, name in enumerate(remote_header) if name not in ignored_columns]
    kept_local_header = [local_header[i] for i in local_keep]
    kept_remote_header = [remote_header[i] for i in remote_keep]

    if kept_local_header != kept_remote_header:
        print(f"FAIL: {label}: column names/order do not match after exclusions.")
        print(f"      local:  {kept_local_header}")
        print(f"      remote: {kept_remote_header}")
        return False

    if len(local_rows) != len(remote_rows):
        print(
            f"FAIL: {label}: row count differs; local={len(local_rows)}, "
            f"remote={len(remote_rows)}."
        )
        return False

    mismatches: list[str] = []
    mismatch_count = 0
    for row_index, (local_row, remote_row) in enumerate(zip(local_rows, remote_rows), start=2):
        for local_col, remote_col, column in zip(local_keep, remote_keep, kept_local_header):
            left = local_row[local_col]
            right = remote_row[remote_col]
            if not cells_equal(left, right):
                mismatch_count += 1
                if len(mismatches) < 20:
                    mismatches.append(
                        f"row {row_index}, column '{column}': "
                        f"local={left[:180]!r} | remote={right[:180]!r}"
                    )

    if mismatch_count:
        print(f"FAIL: {label}: {mismatch_count} value mismatch(es). First differences:")
        for mismatch in mismatches:
            print("      " + mismatch)
        if mismatch_count > len(mismatches):
            print(f"      ... plus {mismatch_count - len(mismatches)} additional mismatch(es).")
        return False

    skipped = ", ".join(sorted(ignored_columns))
    print(
        f"PASS: {label}: {len(local_rows)} rows and {len(kept_local_header)} "
        f"compared columns match exactly (numeric values compared exactly)."
    )
    print(f"      Excluded non-reproducible/duplicated fields: {skipped}.")
    return True


def compare_dataset(root: Path, dataset: str) -> bool:
    print(f"\nComparing dataset: {dataset}")
    print(f"Frozen GitHub baseline: {BASELINE_COMMIT}")
    all_ok = True

    for artifact in CSV_ARTIFACTS:
        relative_path = artifact["relative_path"].format(dataset=dataset)
        local_path = root / relative_path
        try:
            remote_text = download_text(relative_path)
        except RuntimeError as exc:
            print(f"FAIL: {exc}")
            all_ok = False
            continue

        ok = compare_csv(
            local_path=local_path,
            remote_text=remote_text,
            ignored_columns=artifact["ignored_columns"],
            label=artifact["label"],
        )
        all_ok = all_ok and ok

    if all_ok:
        print(f"\nDATASET PASSED: {dataset}")
    else:
        print(f"\nDATASET FAILED: {dataset}. The next dataset must not be started.")
    return all_ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local experiment CSV results with the frozen GitHub baseline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset", choices=DATASET_ORDER,
        help="Compare completed benchmark CSVs for one dataset.",
    )
    group.add_argument(
        "--preflight", choices=DATASET_ORDER,
        help="Check seed config, PYTHONHASHSEED, software versions, and known input hash before running.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        root = find_repo_root()
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}")
        return 2

    if args.preflight:
        return 0 if check_preflight(root, args.preflight) else 1
    return 0 if compare_dataset(root, args.dataset) else 1


if __name__ == "__main__":
    raise SystemExit(main())
