"""Behavioral tests for the audit/cleaning pipeline.

Unlike tests/splitguard.test.mjs (which only greps source files for magic
strings), these tests build tiny synthetic datasets, run the real
audit_dataset.py and prepare_impact_datasets.py scripts against them, and
assert on the resulting files. Run with:

    python -m unittest backend.test_pipeline -v
"""
from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "backend" / "audit_dataset.py"
PREPARE = ROOT / "backend" / "prepare_impact_datasets.py"

_seeds = itertools.count(1)


def make_image(path: Path) -> None:
    # Random noise, not a solid color: dHash encodes local gradients, so a
    # uniform-color image always hashes to all-zero bits and every "distinct"
    # solid image would collide as a near-duplicate of every other one.
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(next(_seeds))
    arr = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True,
    )


class SplitAwareCleaningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def audit(self, dataset: Path) -> Path:
        audit_dir = self.tmp / "audit"
        result = run(AUDIT, str(dataset), "--output", str(audit_dir))
        self.assertEqual(result.returncode, 0, result.stderr)
        return audit_dir

    def test_train_image_matching_validation_is_removed_and_copies_are_independent(self):
        dataset = self.tmp / "dataset"
        make_image(dataset / "train/cat/a1.png")
        dup_train = dataset / "train/cat/dup_with_val.png"
        make_image(dup_train)
        make_image(dataset / "train/dog/d1.png")
        (dataset / "validation/cat").mkdir(parents=True)
        shutil.copy2(dup_train, dataset / "validation/cat/v1.png")  # byte-identical to train image
        make_image(dataset / "validation/dog/v2.png")
        val_test_dup = dataset / "validation/dog/v3.png"
        make_image(val_test_dup)
        make_image(dataset / "test/cat/t1.png")
        test_dup = dataset / "test/dog/t2.png"
        test_dup.parent.mkdir(parents=True)
        shutil.copy2(val_test_dup, test_dup)  # byte-identical: validation duplicates a test image

        audit_dir = self.audit(dataset)
        output = self.tmp / "prepared"
        result = run(PREPARE, "--audit", str(audit_dir), "--dataset", str(dataset), "--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)

        summary = json.loads((output / "preparation_summary.json").read_text())
        removed = json.loads((output / "removed_training_images.json").read_text())
        removed_paths = {r["path"] for r in removed}

        # Train image duplicating a held-out validation image must be removed.
        self.assertIn("train/cat/dup_with_val.png", removed_paths)
        # Validation image duplicating a test image must be removed from validation,
        # never from test (test set is immutable).
        self.assertIn("validation/dog/v3.png", removed_paths)
        self.assertEqual(summary["baseline"]["test"], 2)
        self.assertEqual(summary["cleaned"]["test"], 2)
        self.assertEqual(summary["baseline"]["val"], 2)  # v1, v2 kept; v3 dropped for duplicating a test image
        self.assertEqual(summary["baseline"]["val"], summary["cleaned"]["val"])  # same validation set

        # Prepared "copies" must be real copies, not hardlinks: mutating the raw
        # source file must not affect the already-copied prepared file.
        prepared_dog = list((output / "baseline_original/train/dog").glob("*d1.png"))
        self.assertEqual(len(prepared_dog), 1)
        before = prepared_dog[0].read_bytes()
        (dataset / "train/dog/d1.png").write_bytes(b"mutated")
        after = prepared_dog[0].read_bytes()
        self.assertEqual(before, after)
        self.assertNotEqual(after, b"mutated")

    def test_exact_conflict_confined_to_test_set_blocks_the_run(self):
        dataset = self.tmp / "dataset"
        make_image(dataset / "train/cat/a1.png")
        make_image(dataset / "train/dog/d1.png")
        make_image(dataset / "validation/cat/v1.png")
        make_image(dataset / "validation/dog/v2.png")
        cat_test = dataset / "test/cat/tc.png"
        make_image(cat_test)
        dog_test = dataset / "test/dog/td.png"
        dog_test.parent.mkdir(parents=True)
        shutil.copy2(cat_test, dog_test)  # byte-identical image, contradictory labels, both in test

        audit_dir = self.audit(dataset)
        output = self.tmp / "prepared"
        result = run(PREPARE, "--audit", str(audit_dir), "--dataset", str(dataset), "--output", str(output))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conflicting labels", result.stderr)
        self.assertFalse(output.exists())

    def test_unsplit_dataset_drops_and_reports_undersized_classes(self):
        dataset = self.tmp / "dataset"
        make_image(dataset / "catsonly/1.png")
        make_image(dataset / "catsonly/2.png")  # only 2: below the 3-example floor
        for i in range(6):
            make_image(dataset / "dogsonly" / f"{i}.png")
        for i in range(6):
            make_image(dataset / "birdsonly" / f"{i}.png")

        audit_dir = self.audit(dataset)
        output = self.tmp / "prepared"
        result = run(PREPARE, "--audit", str(audit_dir), "--dataset", str(dataset), "--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)

        summary = json.loads((output / "preparation_summary.json").read_text())
        dropped = {d["label"] for d in summary["dropped_classes_insufficient_examples"]}
        self.assertIn("catsonly", dropped)
        self.assertEqual(summary["eligible_classes"], 2)


if __name__ == "__main__":
    unittest.main()
