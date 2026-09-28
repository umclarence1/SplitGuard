# SplitGuard

**Live demo:** [split-guard-omega.vercel.app](https://split-guard-omega.vercel.app) — the backend runs on a free instance, so the first request after a period of inactivity can take up to a minute to wake up.

Reported model accuracy is only meaningful if the test set is actually held out. In practice, datasets assembled or scraped quickly often end up with the same image, or a near-identical crop, resize, or recompression of it, sitting on both sides of the train/test split. When that happens, a model can partly just recognize an image it already saw, and the accuracy number stops meaning what everyone assumes it means.

SplitGuard checks for exactly that. It audits an image-classification dataset for exact and near-duplicate images crossing train/validation/test splits, and for images that carry conflicting labels. If you want to go further than a duplicate count, it can also train a real benchmark model on the dataset as-is and on a cleaned version with the leaking images removed, and report the measured accuracy difference with a confidence interval, not a single number you have to take on faith.

## How it works

The audit runs first, entirely in the browser: SHA-256 hashing catches byte-identical duplicates, and a perceptual hash (dHash) catches near-duplicates like crops or recompressions, using locality-sensitive banding so it stays fast even on large datasets. Nothing leaves your device for this part.

If you choose to run the benchmark, the dataset uploads to a Python worker, which re-runs its own, more precise version of the same audit (SHA-256 plus a BK-tree indexed dHash search), builds two copies of the dataset — the original, and a cleaned one with leaking training images removed — and trains the same MobileNetV2 linear probe on both, three times each with different random seeds. Training is stochastic, so a single run's accuracy is noisy; the three-seed mean and its 95% confidence interval are what make the before/after comparison trustworthy rather than anecdotal.

The client-side scan is a fast, preliminary heuristic. The server-side audit that actually decides what gets cleaned is a separate, independent pass over the uploaded files — the two are expected to broadly agree, but the worker's decision is the one that matters.

## What it checks

- SHA-256 byte-identical images
- Indexed 64-bit dHash near-duplicate candidates
- Same-label and cross-label relationships
- Cross-split train/validation/test leakage
- Class and split composition
- Corrupt images
- A reversible before/after benchmark using one unchanged, held-out test set

Near-duplicate findings are candidates for human review, not guaranteed duplicates. The review score is a heuristic screening aid, not a validated dataset-quality metric.

## Cleaning policy

For supplied train/validation/test folders, SplitGuard preserves the test set exactly as given. If validation is absent, it's derived from training. The cleaned run removes training images that exactly or nearly match a held-out image, plus any cross-label training conflicts. Validation images that match a test image are also removed from validation, so the test set can't leak into model selection either — but the test set itself is never touched.

If two byte-identical images inside the test set carry conflicting labels, SplitGuard refuses to run the benchmark rather than silently scoring against contradictory ground truth; fix the labels in the source data and re-run. Near-duplicate conflicts confined entirely to validation or entirely to test are reported for review in `preparation_summary.json` but aren't auto-removed, since resolving them would mean deciding which label is correct on the tool's behalf.

For an unsplit, class-folders-only dataset, SplitGuard creates one deterministic stratified split first. Classes with fewer than three examples can't be split meaningfully and are dropped; which classes were dropped is recorded in `preparation_summary.json`. The source folder is never modified either way.

```text
dataset/                       dataset/
  train/class_name/images...     class_name/images...
  validation/class_name/...
  test/class_name/images...
```

## Running it locally

Requirements: Node.js 22+, Python 3.11+, and the packages in `backend/requirements.txt`.

```bash
npm install
pip install -r backend/requirements.txt
npm run dev        # frontend, http://localhost:5173
npm run backend    # worker, http://localhost:8000, in a second terminal
```

Both need to be running — the frontend never starts the worker for you. Copy `.env.example` to `.env.local` if you want to override any defaults.

## Deploying

The frontend and the worker are independent deployments; they don't have to live on the same platform. This repo is set up for a specific free-tier pairing:

- **Frontend → Vercel.** It's a static Vite build with zero required configuration — Vercel detects it automatically. Set `VITE_SPLITGUARD_API_URL` (and `VITE_SPLITGUARD_API_KEY`, if the worker requires one) as project environment variables.
- **Worker → Render**, using the included `render.yaml` Blueprint. Deploy via Render's **New → Blueprint** flow; it reads the file and provisions the service, including generating a random `SPLITGUARD_API_KEY`.

Whichever host you pick for the worker, it needs to be a real, persistent process, not a serverless function: a single dataset audit and training run can take minutes, relies on a background job queue that lives in memory for the process's lifetime, and writes intermediate files to local disk across that whole run. Don't point a deployed frontend at `localhost:8000`.

One practical limit worth knowing if you're also running on a free tier: Render's free instance has 512MB of RAM, which PyTorch and a training batch can get close to. `SPLITGUARD_TRAIN_BATCH_SIZE` and `SPLITGUARD_TRAIN_IMAGE_SIZE` (see below) trade off memory against how demanding a dataset the worker can handle — the values in `render.yaml` are tuned down from the script's own defaults for exactly this reason. A free instance can also be restarted by the host at any time, which will interrupt whatever job is running — treat a free-tier deployment as fine for a demo, not as somewhere to trust a long unattended run.

After the worker is deployed, update its `SPLITGUARD_CORS_ORIGINS` to include the frontend's real URL — without this the browser will block every request to it.

## Configuration

All variables are optional and have working defaults for local, single-user use.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_SPLITGUARD_API_URL` | `http://localhost:8000` | Where the frontend sends experiment requests |
| `VITE_SPLITGUARD_API_KEY` | — | Sent as `X-SplitGuard-Api-Key` if the worker requires one |
| `SPLITGUARD_API_KEY` | — | If set, gates `POST /api/experiments` |
| `SPLITGUARD_CORS_ORIGINS` | `localhost:5173` (both schemes) | Comma-separated allowlist of origins permitted to call the worker |
| `SPLITGUARD_WORKERS` | `1` | Concurrent training jobs |
| `SPLITGUARD_RETENTION_HOURS` | `24` | How long finished/failed/cancelled jobs are kept before being swept |
| `SPLITGUARD_MAX_FILES` | `25000` | Upload file-count ceiling |
| `SPLITGUARD_MAX_UPLOAD_BYTES` | `21474836480` (20GB) | Upload size ceiling |
| `SPLITGUARD_TRAIN_BATCH_SIZE` | `32` | Training batch size — lower to reduce peak memory |
| `SPLITGUARD_TRAIN_IMAGE_SIZE` | `160` | Training image resolution — lower to reduce peak memory |

## Worker behavior

- A bounded job queue, not unlimited concurrent training threads.
- Interrupted jobs are safely re-queued if the worker restarts.
- Active jobs can be cancelled; cancelled subprocesses are terminated and reaped, not left running.
- Finished jobs expire on a schedule (`SPLITGUARD_RETENTION_HOURS`), swept by a background thread every hour — this runs continuously, not just once at startup, and never blocks the server from handling requests.
- Uploaded files with a non-image extension are dropped server-side regardless of what the client claims.

## Security

- Every created experiment gets a random `owner_token`. Reading, cancelling, or deleting it requires that token — knowing a job's id alone isn't enough. The frontend handles this automatically.
- `SPLITGUARD_API_KEY` optionally gates who can submit a job at all, separately from per-job ownership.
- CORS origins are matched exactly (trimmed of whitespace and trailing slashes) against an explicit allowlist, with only the methods and headers the app actually uses — no wildcards, and credentials are never sent.
- Uploaded datasets sit unencrypted on the worker's disk until they expire. Don't point an untrusted or multi-tenant deployment at sensitive data without disk-level encryption and network isolation.
- The audit report and the job's id/token are kept in the browser tab's `sessionStorage`, so a reload reconnects to a running or finished experiment instead of losing it. Re-running an experiment still needs the dataset folder reselected, since browsers don't let a page hold onto file handles across a reload.

## Testing

```bash
npm test
python -m unittest backend.test_pipeline -v
```

`npm test` builds the frontend and checks that dataset-specific accuracy fallbacks are absent, experiment consent is present, split-aware cleaning is configured, multi-seed uncertainty is emitted, and worker safeguards remain enabled.

`backend.test_pipeline` builds small synthetic datasets and runs the real audit and cleaning scripts against them, then checks the actual output: a train image matching a held-out image gets removed, a validation image matching a test image is removed from validation only, a label conflict confined to the test set aborts the run instead of corrupting the benchmark, undersized classes are dropped and reported, and the "cleaned" copies are genuinely independent files, not hardlinks back to the raw upload.

## Known limitations

- Near-duplicate detection is a heuristic; it will occasionally flag distinct images as similar or miss a genuine duplicate, especially under heavy crops or color shifts.
- The training benchmark measures one small transfer-learning model (a frozen MobileNetV2 backbone), not the accuracy of whatever model you actually plan to ship.
- On a memory-constrained free host, very large or high-class-count datasets can still exceed available RAM even at the reduced batch/image settings.
- A free-tier worker deployment has no persistent disk guarantee across host-initiated restarts; an interrupted job's data may not survive one.
