# SplitGuard

SplitGuard audits image-classification datasets for evaluation leakage and measures how a controlled benchmark changes after leaking or conflicting training images are removed.

## What it checks

- SHA-256 byte-identical images
- Indexed 64-bit dHash near-duplicate candidates
- Same-label and cross-label relationships
- Cross-split train/validation/test leakage
- Class and split composition
- Corrupt images in the worker audit
- A reversible before/after benchmark using one unchanged held-out test set

Near-duplicate findings are candidates for human review, not guaranteed duplicates. The review signal is a heuristic screening aid, not a validated dataset-quality score.

## Accuracy interpretation

The impact result is the mean of three real MobileNetV2 linear-probe runs. It includes a 95% confidence interval and macro F1. This is a **SplitGuard benchmark**, not the accuracy of a user's own model.

For supplied train/validation/test folders, SplitGuard preserves the test set. If validation is absent, it derives validation data from training. The cleaned run removes training images that exactly or nearly match held-out images, plus cross-label training conflicts. Baseline and cleaned runs use the same validation and test images.

For an unsplit class-folder dataset, SplitGuard creates one deterministic stratified split, then removes training members connected to the held-out set. The source folder is never modified. Classes with fewer than 3 examples cannot be split and are dropped; dropped classes are listed in `preparation_summary.json`.

Validation images that exactly or nearly match a test image are also removed from validation, so the test set can never leak into model selection. The test set itself is never modified or filtered. If two byte-identical images inside the test set carry conflicting labels, SplitGuard refuses to run the benchmark rather than silently scoring against contradictory ground truth — fix the labels in the source dataset and re-run. Near-duplicate or same-split conflicts confined to validation or to test alone are reported as review items in `preparation_summary.json` but are not auto-removed.

## Local setup

Requirements: Node.js 22+, Python 3.11+, and the packages in `backend/requirements.txt`.

```bash
npm install
pip install -r backend/requirements.txt
npm run dev
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:5173`.

The browser audit remains on the device. Dataset transfer begins only after the user explicitly starts the optional benchmark. By default, the worker is also local. Configure a hosted worker with `VITE_SPLITGUARD_API_URL` and matching `SPLITGUARD_CORS_ORIGINS` values; see `.env.example`.

The client-side audit is a preliminary, in-browser heuristic scan. Once training starts, the worker re-derives leakage from the uploaded files itself (`backend/audit_dataset.py`) and decides what to remove from that independent, authoritative pass — the browser's findings and score are a review aid, not the source of truth for cleaning.

## Dataset layouts

Split dataset:

```text
dataset/
  train/class_name/images...
  validation/class_name/images...
  test/class_name/images...
```

Unsplit dataset:

```text
dataset/
  class_name/images...
```

Classes need enough examples to produce non-empty training and test data. SplitGuard rejects experiments that leave fewer than two trainable classes.

## Worker behavior

- Jobs use a bounded worker queue instead of unlimited training threads.
- Interrupted jobs are safely re-queued after restart.
- Active jobs can be cancelled; cancelled subprocesses are terminated and reaped.
- Completed, failed and cancelled jobs expire after 24 hours by default, swept by a background thread every hour (not just at process start, and never blocking server startup or new requests).
- Upload size, file count, retention, concurrency and allowed origins are configurable.
- Uploaded files with a non-image extension are dropped server-side, regardless of what the client claims.
- `DELETE /api/experiments/{job}` immediately removes a stored experiment.

## Security and access control

- Each created experiment returns a per-job `owner_token`. `GET /api/experiments/{job}`, its `/cancel`, and its `DELETE` all require the matching `X-SplitGuard-Owner-Token` header once a job has one, so knowing a job id alone isn't enough to read, cancel, or delete someone else's experiment. The frontend stores and sends this automatically.
- To survive a page reload, the frontend keeps the audit report plus the job id and owner token in the tab's `sessionStorage` (cleared when the tab closes). After a reload it reconnects to a running or finished experiment instead of returning to the home screen. Re-running an experiment still needs the dataset folder to be chosen again, because browsers cannot persist file handles.
- Set `SPLITGUARD_API_KEY` on the worker to require a matching `X-SplitGuard-Api-Key` header on `POST /api/experiments`, gating who may submit jobs at all. Set `VITE_SPLITGUARD_API_KEY` to the same value so the frontend can authenticate. Leave both unset for local, single-user development.
- CORS origins (`SPLITGUARD_CORS_ORIGINS`) are trimmed of whitespace and trailing slashes before matching, and only `GET`/`POST`/`DELETE`/`OPTIONS` with the headers SplitGuard actually uses are allowed — no wildcard methods or headers, and credentials are never sent.
- Raw uploaded datasets are stored unencrypted on the worker's disk for up to `SPLITGUARD_RETENTION_HOURS`. Do not point untrusted or multi-tenant deployments at a worker without disk-level encryption and network isolation if the datasets are sensitive.

For public deployment, run the Python worker on persistent compute with adequate CPU/GPU and disk/object storage. Do not point a deployed frontend at `localhost:8000`.

## Validation

```bash
npm test
python -m unittest backend.test_pipeline -v
```

`npm test` builds the React/Vite frontend and checks that dataset-specific accuracy fallbacks are absent, experiment consent is present, split-aware cleaning is configured, multi-seed uncertainty is emitted, and worker safeguards remain enabled.

`backend.test_pipeline` builds tiny synthetic datasets and runs the real `audit_dataset.py` / `prepare_impact_datasets.py` scripts against them, then asserts on the output: a train image matching a held-out image is removed, a validation image matching a test image is removed from validation only, an exact label conflict confined to the test set aborts the run instead of silently corrupting evaluation, undersized classes are dropped and reported, and prepared "copies" are real independent files rather than hardlinks to the raw upload.
