# Training on Kaggle (free GPU) and downloading the result

Your local machine has 16 GB RAM and no GPU. Kaggle notebooks give you a free
T4 (16 GB, often offered as T4 x2), 30 GPU-hours per week, and a 12 hour cap
per session. This guide trains Fontaine there and brings the checkpoint home.

The notebook trains on **GPU 0 only**. A second T4 is left idle until a
single-GPU resume run has succeeded. Do not train `yami_base`, `yami_large`,
or any `configs/yami_super.yaml` profile above `tiny` on a T4. The 1b AdamW
estimate is already about 16 GB before activations.

| File | Purpose |
|---|---|
| `kaggle/train_fontaine_kaggle.ipynb` | The training notebook. Run it on Kaggle |
| `kaggle/dataset-metadata.json` | Dataset id `needyamin/fontaine-data` |
| `configs/training/kaggle.yaml` | Shared GPU baseline (precision, seed, checkpoint interval, early stop) |

## 1. One-time Kaggle setup

1. Sign in at [kaggle.com](https://www.kaggle.com) and verify your phone
   (Settings → Phone verification). GPUs and internet access require it.
2. Settings → Create New Token. That saves `kaggle.json`.

## 2. Upload the corpus

Data stays out of git. Put files in `datasets/raw`, then publish the dataset
named in `kaggle/dataset-metadata.json`.

```bash
pip install kaggle
mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/   # Windows: %USERPROFILE%\.kaggle
copy kaggle\dataset-metadata.json datasets\raw\
kaggle datasets create -p datasets/raw
```

If `needyamin/fontaine-data` already exists, publish a new version instead:

```bash
copy kaggle\dataset-metadata.json datasets\raw\
kaggle datasets version -p datasets\raw -m "update"
```

Do not commit `datasets/raw`. The metadata file in `kaggle/` is the template.
The notebook skips `dataset-metadata.json` when it copies the input.

## 3. Run the notebook

Push the repo to `main` first. The notebook clones that branch, prints the
commit, and will not see unpushed work. If Internet must stay off, upload a
zip of the repo as a second dataset and unzip it to `/kaggle/working/yami`
instead of the clone cell.

1. Create → New Notebook → File → Import Notebook →
   `kaggle/train_fontaine_kaggle.ipynb`.
2. Add Input → Your Work → Datasets → `fontaine-data`.
3. Settings → Accelerator: **GPU T4 x2**, Internet: **On**.
4. In the first code cell, set `MODEL` to one of `tiny`, `small`, `medium`,
   `coding_low`, `yami_nano`, `yami_small`. Each preset sets sequence length,
   batch size, and gradient checkpointing for a 16 GB T4. `MAX_STEPS` defaults
   to 5000. `EARLY_STOP` empty keeps the yaml value of 5 evals.
5. Run All.

The notebook copies each attached dataset to
`datasets/raw/<dataset-name>/<relative-path>` (same filenames in two datasets
do not collide), checks feasibility, trains a byte-level BPE tokenizer,
prepares shards, trains, generates a short sample, and writes
`/kaggle/working/fontaine_<run>.zip`. The zip contains the run, the tokenizer,
and `HOW_TO_RUN.txt`.

After prepare it prints the train-window count. The first log line (step 25)
includes `tokens_per_second`. A 12 hour session holds about
`tokens_per_second * 43200 / sequence_length` steps. Stop and Run All again
if you need more than that.

Keep the tab alive, or use Save Version → Save & Run All. The zip is under
the version's Output tab.

## 4. Resume a saved session

`/kaggle/working` survives a saved session. Run All again with the same
`RUN_NAME` and `START_FRESH = False`. If `experiments/*<RUN_NAME>*/checkpoints/latest.json`
exists together with `datasets/tokenizer` and `datasets/prepared/manifest.json`,
the notebook skips tokenizer training and shard preparation and calls
`train --resume` on that checkpoints directory.

`START_FRESH = True` ignores the checkpoint and rebuilds the tokenizer and shards.

## 5. Download and run locally

Interactive session: Output panel → `fontaine_<run>.zip`.
Saved version: notebook page → Output.

Unzip beside your clone, not over it. The commands are also in `HOW_TO_RUN.txt`
inside the zip:

```bash
unzip fontaine_<run>.zip -d kaggle_output
python -m fontaine.cli.main generate \
  --checkpoint kaggle_output/experiments/<run>/checkpoints \
  --tokenizer-dir kaggle_output/datasets/tokenizer --interactive
python -m fontaine.cli.main serve \
  --checkpoint kaggle_output/experiments/<run>/checkpoints \
  --tokenizer-dir kaggle_output/datasets/tokenizer
```

Checkpoints are the same hashed PyTorch files a local run writes. A tiny
checkpoint is tens of megabytes. `/kaggle/working` holds about 20 GB, and
`keep_last_n_checkpoints` is 2. A checkpoint that is too large to download
can be saved as a new Kaggle dataset and pulled with
`kaggle datasets download <username>/<slug>`.

## Limits

- Quota: 30 GPU-hours per week (resets Saturday midnight UTC). A session
  ends at 12 hours.
- Precision: `training.precision: auto` uses fp16 and GradScaler on a T4,
  and bf16 on an A100.
- GPU 0 only. Do not launch `torchrun` on both T4s until one single-GPU
  resume has finished.
- `early_stopping_patience: 5` stops a small corpus from using the whole
  week once validation loss stalls. Set `EARLY_STOP = "0"` in the notebook
  to turn that off.
- Models that feasibility marks `training: not feasible` are refused before
  tokenizer training.
