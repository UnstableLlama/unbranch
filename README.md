# unbranch

Split a HuggingFace repo that stores quantized models on separate branches (e.g. `2.10bpw`, `3.00bpw`, `6.00bpw`) into individual single-BPW repos.

> [!CAUTION]
> **This script performs destructive, irreversible operations.** It renames your parent repo, force-pushes branches to main, and deletes branches. There is no undo. Always do a `--dry-run` first (or use the Jupyter notebook with `DRY_RUN = True`) to preview what will happen before committing to a real run.

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
  UnstableLlama/Qwen3.5-4B-2.10bpw-exl3  (new repo)
  UnstableLlama/Qwen3.5-4B-3.00bpw-exl3  (new repo)
  UnstableLlama/Qwen3.5-4B-4.00bpw-exl3  (new repo)
  UnstableLlama/Qwen3.5-4B-5.00bpw-exl3  (new repo)
  UnstableLlama/Qwen3.5-4B-6.00bpw-exl3  (renamed parent)
```

The largest BPW gets the renamed parent repo, preserving download counts and stars.

## How it works

1. Downloads the README and rewrites branch links to point at the new single-BPW repos.
2. For each BPW except the largest: creates a new repo and pushes the branch content as `main`.
3. For the largest BPW: force-pushes that branch to `main` on the parent repo, then renames it.
4. Verifies all repos have files.
5. Deletes the old BPW branches from the renamed parent.

No large files are downloaded. Branches are cloned with `GIT_LFS_SKIP_SMUDGE=1` (only LFS pointers touch disk). HuggingFace's server resolves the pointers since the LFS objects already exist in its storage.

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

## Supported quant formats

The BPW is inserted before the quant suffix in the repo name. Recognized suffixes:

| Suffix | Format |
|--------|--------|
| `-exl3` | ExLlamaV3 |
| `-exl2` | ExLlamaV2 |
| `-gguf` | GGUF |
| `-gptq` | GPTQ |
| `-awq` | AWQ |

For unrecognized suffixes, the BPW is appended at the end.
