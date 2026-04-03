# unbranch

Instantly split a branched HuggingFace repo of quantized models (e.g. `2.10bpw`, `3.00bpw`, `6.00bpw`) into individual single-BPW repos while retaining likes and downloads.

> [!CAUTION]
> **This script performs destructive, irreversible operations.** It overwrites the parent repo's main branch, renames the parent repo, and deletes branches. There is no undo. Always do a `--dry-run` first (or use the Jupyter notebook with `DRY_RUN = True`) to preview what will happen before committing to a real run.

> [!WARNING]
> **HuggingFace rate limits apply.** `duplicate_repo` is limited to **15 duplications per day**. The general API limit is **1000 requests per 5-minute window**. A repo with 5 BPWs uses 4 duplications (the largest BPW reuses the parent), so you can split ~3-4 repos per day. Plan accordingly.

## Why

HuggingFace repos that use branches for different quantization bitrates are harder to browse and download. This tool converts them into the cleaner single-repo-per-bitrate convention:

```
BEFORE (branched):
  UnstableLlama/Qwen3.5-4B-exl3
    ├── branch: 2.10bpw
    ├── branch: 3.00bpw
    ├── branch: 4.00bpw
    ├── branch: 5.00bpw
    └── branch: 6.00bpw

AFTER (unbranched):
  UnstableLlama/Qwen3.5-4B-exl3-2.10bpw  (new repo)
  UnstableLlama/Qwen3.5-4B-exl3-3.00bpw  (new repo)
  UnstableLlama/Qwen3.5-4B-exl3-4.00bpw  (new repo)
  UnstableLlama/Qwen3.5-4B-exl3-5.00bpw  (new repo)
  UnstableLlama/Qwen3.5-4B-exl3-6.00bpw  (renamed parent)
```

The largest BPW gets the renamed parent repo, preserving download counts and stars. The original main branch content is saved as a `main_original` branch on that repo, so nothing is lost.

## How it works

Pure HuggingFace API — no git, no model file downloads. The only file that touches your disk is the README.

1. Downloads the README and rewrites branch links to point at the new single-BPW repos.
2. For each BPW except the largest:
   - `CommitOperationCopy`: server-side copy of branch files → parent's main
   - `duplicate_repo`: snapshot parent's main → new single-BPW repo
   - Restore parent's main from a backup branch
3. For the largest BPW: copy branch → parent's main, rename the parent repo.
4. Verifies all repos have files.
5. Deletes the old BPW branches from the renamed parent.

The original parent repo's main branch content is preserved as a branch called `main_original` on the renamed (largest BPW) repo, with its likes and downloads.

## Requirements

```bash
pip install huggingface_hub
```

## Usage

### CLI

```bash
export HF_TOKEN=hf_...

python unbranch.py \
  --author UnstableLlama \
  --repo-name Qwen3.5-4B-exl3 \
  --bpws 2.10 3.00 4.00 5.00 6.00 \
  # --private \
  --dry-run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--author` | *(required)* | HF username or org |
| `--repo-name` | *(required)* | Source repo name (without author) |
| `--bpws` | *(required)* | BPW values present as branches |
| `--dry-run` | off | Preview actions without making changes |
| `--private` | off | Create new repos as private |

Remove `--dry-run` when you're ready to execute for real.

### Jupyter Notebook (recommended)

Open `unbranch.ipynb` and fill in the config cell. Same workflow, interactive output. The notebook runs each step in its own cell so you can inspect results before proceeding, and defaults to `DRY_RUN = True`.

## Naming

The BPW is appended to the repo name: `Qwen3.5-4B-exl3` → `Qwen3.5-4B-exl3-4.00bpw`.
