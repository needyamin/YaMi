# Training on Kaggle (free GPU) and downloading the result

Your local machine has 16 GB RAM and no GPU — Kaggle notebooks give you a free
T4 x2 / P100 GPU (30 GPU-hours/week, 12 h max per session) and persist
everything written to `/kaggle/working`. This guide trains Fontaine there and
brings the checkpoints home.

Everything needed lives in the repo:

| File | Purpose |
|---|---|
| `kaggle/train_fontaine_kaggle.ipynb` | The training notebook — run it on Kaggle |
| `kaggle/dataset-metadata.json` | Template for pushing your data via the Kaggle CLI |
| `configs/training/kaggle.yaml` | GPU-tuned training config used by the notebook |

## 1. One-time Kaggle setup

1. Sign in at [kaggle.com](https://www.kaggle.com) and **verify your phone**
   (Settings → Phone verification) — required for GPUs and internet access.
2. Get your API token: Settings → **Create New Token** → saves `kaggle.json`.

## 2. Upload your data as a Kaggle Dataset

Your `datasets/` folder is git-ignored, so data travels separately. Either:

**Option A — web UI:** Create → New Dataset → upload the files from
`datasets/raw/` (e.g. `codealpaca_20k.jsonl`). Private is fine.

**Option B — CLI (repeatable):**

```bash
pip install kaggle
mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/   # Windows: %USERPROFILE%\.kaggle
cp kaggle/dataset-metadata.json datasets/raw/               # edit id: <username>/fontaine-data
kaggle datasets create -p datasets/raw
# later updates: kaggle datasets version -p datasets/raw -m "update"
```

7 MB (CodeAlpaca-20k) uploads in seconds; the limit is ~100 GB per dataset.

## 3. Run the notebook

1. Create → New Notebook → **File → Import Notebook** → upload
   `kaggle/train_fontaine_kaggle.ipynb`.
2. **Add Input → Your Work/Datasets → fontaine-data** (your dataset from step 2).
3. Notebook Settings → **Accelerator: GPU T4 x2**, **Internet: On**.
4. Edit the constants in the first code cell if you want a different model
   (`tiny` / `small` / `medium`), run name, or step count.
5. **Run All.** The notebook clones the repo, installs it, trains the
   tokenizer, prepares shards, trains the model, runs a generation sanity
   check, and zips `experiments/<run>/` + `datasets/tokenizer/` into
   `/kaggle/working/fontaine_<run>.zip`.

Keep the browser tab alive (or use **Save Version → Save & Run All** to run it
in the background — the zip then appears under the version's **Output** tab).

## 4. Download the trained model

- Interactive session: right panel → **Output** → download
  `fontaine_<run>.zip`. A tiny checkpoint is tens of megabytes (fp32 weights
  plus Adam state), well under the ~20 GB output limit.
- Saved version: notebook page → **Output** tab → download.
- Big checkpoints can also be saved as a new Kaggle Dataset ("New Dataset"
  from output) and pulled locally with
  `kaggle datasets download <username>/<slug>`.

Then on your machine, inside your `yami` clone:

```bash
unzip fontaine_<run>.zip -d kaggle_output
fontaine generate \
  --checkpoint kaggle_output/experiments/<run>/checkpoints \
  --tokenizer-dir kaggle_output/datasets/tokenizer --interactive
```

The checkpoints are plain PyTorch files with hash-sealed manifests — the same
`fontaine generate / evaluate / serve` commands work locally exactly as they
do for CPU-trained runs.

## Limits & tips

- **Quota:** 30 GPU-hours/week (resets Saturday midnight UTC); a session caps
  at 12 h. Fontaine tiny finishes in minutes, small in ~1–2 h on a T4.
- **Precision:** `training.precision: auto` in `configs/training/kaggle.yaml`
  picks bf16 on bf16-capable GPUs and fp16 + GradScaler on T4/P100 — no edits
  needed.
- **Resuming:** files already written to `/kaggle/working` survive a saved
  session, but the notebook starts each run from step 0. The CLI can continue
  a run with `fontaine train --resume <checkpoints dir>` (that directory's
  `latest.json`). Prefer sizing `max_steps` to finish within one session.
- **Medium model:** set `batch_size: 32` and `gradient_checkpointing: true`
  in `configs/training/kaggle.yaml` if you hit VRAM limits.
- **No internet?** If you must keep Internet off, upload the repo zip as a
  Kaggle Dataset too and unzip it in the notebook instead of `git clone`.
