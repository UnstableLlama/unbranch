#!/usr/bin/env python3
"""
unbranch.py — Split a HuggingFace repo with branched quantizations into
separate single-BPW repos.

WARNING: This script performs destructive, irreversible operations. It renames
repos, force-pushes branches to main, and deletes branches. Always use
--dry-run first to preview what will happen.

Usage:
    export HF_TOKEN=hf_...
    python unbranch.py \
        --author UnstableLlama \
        --repo-name Qwen3.5-4B-exl3 \
        --bpws 2.10 3.00 4.00 5.00 6.00

What it does:
    1. Downloads the README from the parent repo and rewrites branch links
       to point at the new single-BPW repos.
    2. For every BPW *except* the largest: duplicates the parent repo
       server-side via HF API, then pushes the correct branch as main
       and deletes the other branches. No model files are downloaded.
    3. For the largest BPW: pushes that branch as main on the parent repo,
       then renames it.
    4. Verifies every new repo has files.
    5. Deletes the old BPW branches from the (now-renamed) parent repo.

    The only file downloaded locally is the README. All model weight
    transfers happen server-side via duplicate_repo().

Requires: huggingface_hub  (pip install huggingface_hub)
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

from huggingface_hub import HfApi, hf_hub_download


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_bpw(bpw: float) -> str:
    """Format a BPW value to two decimal places."""
    return f"{bpw:.2f}"


def run_git(args: list[str], cwd: str | None = None, env: dict | None = None):
    """Run a git command, printing it and raising on failure."""
    merged_env = {**os.environ, **(env or {})}
    display = " ".join(args)
    display = re.sub(r"(https?://)[^@]+@", r"\1***@", display)
    print(f"  $ {display}")
    result = subprocess.run(
        args, cwd=cwd, env=merged_env, capture_output=True, text=True
    )
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[:10]:
            print(f"    {line}")
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr.strip()}")
        result.check_returncode()
    return result


# ── README Rewriting ─────────────────────────────────────────────────────────

def make_target_name(repo_name: str, bpw: float) -> str:
    """Build the single-BPW repo name by inserting the BPW before the quant suffix.

    Example: Qwen3.5-4B-exl3  +  4.00  →  Qwen3.5-4B-4.00bpw-exl3
    """
    # Known quantization suffixes (add more as needed)
    for suffix in ("-exl3", "-exl2", "-gguf", "-gptq", "-awq"):
        if repo_name.endswith(suffix):
            base = repo_name[: -len(suffix)]
            return f"{base}-{fmt_bpw(bpw)}bpw{suffix}"

    # Fallback: append at end
    return f"{repo_name}-{fmt_bpw(bpw)}bpw"


def rewrite_readme(
    readme: str,
    author: str,
    repo_name: str,
    bpws: list[float],
) -> str:
    """Replace branch-style URLs/commands with single-repo equivalents."""

    parent = f"{author}/{repo_name}"

    for bpw in bpws:
        b = fmt_bpw(bpw)
        branch = f"{b}bpw"
        target = f"{author}/{make_target_name(repo_name, bpw)}"

        # Table / inline links:  …/Author/Repo/tree/X.XXbpw  →  …/Author/Repo-X.XXbpw-exl3
        readme = readme.replace(
            f"{parent}/tree/{branch}",
            f"{target}",
        )

        # CLI download with quoted revision:
        #   hf download Author/Repo --revision "X.XXbpw" --local-dir …
        #   →  hf download Author/Repo-X.XXbpw-exl3 --local-dir …
        readme = readme.replace(
            f'{parent} --revision "{branch}"',
            f"{target}",
        )

        # CLI download with unquoted revision:
        readme = readme.replace(
            f"{parent} --revision {branch}",
            f"{target}",
        )

    # Also update --local-dir references to match new naming
    # Old: --local-dir ./Repo-X.XXbpw  →  --local-dir ./NewRepoName
    for bpw in bpws:
        b = fmt_bpw(bpw)
        old_dir = f"{repo_name}-{b}bpw"
        new_dir = make_target_name(repo_name, bpw)
        if old_dir != new_dir:
            readme = readme.replace(
                f"--local-dir ./{old_dir}",
                f"--local-dir ./{new_dir}",
            )

    return readme


# ── Core Logic ───────────────────────────────────────────────────────────────

def _clone_branch_and_update_readme(*, hf_url, repo_id, branch, readme_text, tmpdir):
    """Clone a single branch (LFS skip), update README, prepare for push."""
    lfs_env = {"GIT_LFS_SKIP_SMUDGE": "1"}

    run_git(
        ["git", "clone", "--single-branch", "--branch", branch,
         hf_url(repo_id), tmpdir],
        env=lfs_env,
    )

    run_git(["git", "config", "user.email", "unbranch@local"], cwd=tmpdir)
    run_git(["git", "config", "user.name", "unbranch"], cwd=tmpdir)

    # Write the updated README
    with open(os.path.join(tmpdir, "README.md"), "w") as f:
        f.write(readme_text)

    run_git(["git", "add", "README.md"], cwd=tmpdir)

    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=tmpdir, capture_output=True,
    )
    if diff.returncode != 0:
        run_git(
            ["git", "commit", "-m",
             "Update README: branch links -> single-repo links"],
            cwd=tmpdir,
        )

    run_git(["git", "branch", "-M", "main"], cwd=tmpdir)


def wait_for_branch(api, repo_id: str, branch: str, timeout: int = 120):
    """Poll until a branch exists on a HuggingFace repo.

    duplicate_repo() is async on HF's side — branches may not be available
    immediately after the call returns.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            refs = api.list_repo_refs(repo_id, repo_type="model")
            branch_names = [b.name for b in refs.branches]
            if branch in branch_names:
                return
        except Exception:
            pass
        print(f"    Waiting for branch {branch} on {repo_id}...")
        time.sleep(2)
    raise TimeoutError(
        f"Branch {branch} did not appear on {repo_id} within {timeout}s"
    )


def push_branch_to_main(*, repo_id, branch, readme_text, token):
    """Push a branch as main within the SAME repo (no large file downloads).

    Clones with GIT_LFS_SKIP_SMUDGE=1 so only LFS pointers touch disk.
    Since the push target is the same repo, the LFS objects already exist
    server-side and HuggingFace accepts the pointers.
    """
    lfs_env = {"GIT_LFS_SKIP_SMUDGE": "1"}

    def hf_url(rid):
        return f"https://user:{token}@huggingface.co/{rid}"

    with tempfile.TemporaryDirectory() as tmpdir:
        _clone_branch_and_update_readme(
            hf_url=hf_url, repo_id=repo_id, branch=branch,
            readme_text=readme_text, tmpdir=tmpdir,
        )
        run_git(
            ["git", "push", "-u", "origin", "main", "--force"],
            cwd=tmpdir, env=lfs_env,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Split a branched HF quant repo into single-BPW repos.",
    )
    parser.add_argument("--author", required=True, help="HF username or org")
    parser.add_argument("--repo-name", required=True, help="Source repo name (without author)")
    parser.add_argument(
        "--bpws", nargs="+", required=True, type=float,
        help="BPW values present as branches (e.g. 2.10 3.00 4.00 5.00 6.00)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create new repos as private (does not affect the renamed parent repo)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Set the HF_TOKEN environment variable first.")
        sys.exit(1)

    api = HfApi(token=token)

    bpws = sorted(args.bpws)
    if len(bpws) < 2:
        print("Error: Need at least 2 BPW values (one becomes the renamed parent).")
        sys.exit(1)

    largest_bpw = bpws[-1]
    smaller_bpws = bpws[:-1]

    parent_repo = f"{args.author}/{args.repo_name}"

    def target_name(bpw):
        return make_target_name(args.repo_name, bpw)

    def target_id(bpw):
        return f"{args.author}/{target_name(bpw)}"

    # ── 1. Download & rewrite README ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 1: Download & rewrite README from {parent_repo}")
    print(f"{'=' * 60}\n")

    readme_file = hf_hub_download(parent_repo, "README.md", token=token)
    with open(readme_file) as f:
        original_readme = f.read()

    readme_text = rewrite_readme(original_readme, args.author, args.repo_name, bpws)

    # Show a quick diff summary
    changed = original_readme != readme_text
    print(f"  README modified: {changed}")
    if changed:
        old_lines = set(original_readme.splitlines())
        new_lines = set(readme_text.splitlines())
        for line in sorted(new_lines - old_lines):
            if "huggingface.co" in line or "hf download" in line:
                print(f"    + {line.strip()[:120]}")

    largest_branch = f"{fmt_bpw(largest_bpw)}bpw"
    largest_repo = target_id(largest_bpw)

    # ── 2. Create repos for smaller BPWs ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 2: Create repos for smaller BPWs")
    print(f"{'=' * 60}")

    if args.dry_run:
        for bpw in smaller_bpws:
            print(f"  [DRY RUN] Would create {target_id(bpw)} ← branch {fmt_bpw(bpw)}bpw")
    else:
        for bpw in smaller_bpws:
            branch = f"{fmt_bpw(bpw)}bpw"
            repo_id = target_id(bpw)
            print(f"\n  ── {repo_id} (from branch {branch}) ──")

            # Duplicate parent repo server-side (all branches + LFS, no local downloads)
            print(f"  Duplicating {parent_repo} → {repo_id}")
            api.duplicate_repo(
                from_id=parent_repo,
                to_id=repo_id,
                private=args.private,
                exist_ok=True,
                repo_type="model",
            )
            time.sleep(0.5)

            # Wait for async duplication to finish
            wait_for_branch(api, repo_id, branch)

            # Push the correct branch as main (same repo, no LFS download)
            push_branch_to_main(
                repo_id=repo_id,
                branch=branch,
                readme_text=readme_text,
                token=token,
            )
            time.sleep(0.5)

            # Delete all BPW branches from the duplicate
            for other_bpw in bpws:
                other_branch = f"{fmt_bpw(other_bpw)}bpw"
                try:
                    api.delete_branch(repo_id, branch=other_branch, repo_type="model")
                    time.sleep(0.5)
                except Exception:
                    pass

            print(f"  ✓ {repo_id}")

    # ── 3. Handle parent repo → largest BPW ──────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 3: Convert parent repo to largest BPW ({fmt_bpw(largest_bpw)})")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print(f"  [DRY RUN] Would push {largest_branch} → main on {parent_repo}")
        print(f"  [DRY RUN] Would rename {parent_repo} → {largest_repo}")
    else:
        print(f"  Pushing {largest_branch} → main on {parent_repo}")
        push_branch_to_main(
            repo_id=parent_repo,
            branch=largest_branch,
            readme_text=readme_text,
            token=token,
        )
        time.sleep(0.5)

        print(f"\n  Renaming {parent_repo} → {largest_repo}")
        api.move_repo(from_id=parent_repo, to_id=largest_repo, repo_type="model")
        print(f"  ✓ Renamed")
        time.sleep(0.5)

    # ── 4. Verify ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 4: Verification")
    print(f"{'=' * 60}\n")

    all_ok = True
    if args.dry_run:
        for bpw in bpws:
            print(f"  [DRY RUN] Would verify {target_id(bpw)}")
    else:
        for bpw in bpws:
            repo_id = target_id(bpw)
            try:
                files = api.list_repo_files(repo_id, repo_type="model")
                count = len(files)
                has_safetensors = any(f.endswith(".safetensors") for f in files)
                status = "OK" if count >= 2 and has_safetensors else "⚠ CHECK"
                print(f"  {repo_id}: {count} files [{status}]")
                if count < 2 or not has_safetensors:
                    all_ok = False
            except Exception as e:
                print(f"  {repo_id}: ERROR — {e}")
                all_ok = False

        if not all_ok:
            print("\n  ⚠ Some repos may have issues. Skipping branch cleanup.")
            print("  Please verify manually, then delete branches with:")
            for bpw in bpws:
                branch = f"{fmt_bpw(bpw)}bpw"
                print(f"    huggingface_hub.HfApi().delete_branch('{largest_repo}', branch='{branch}')")
            sys.exit(1)

    # ── 5. Delete old branches ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 5: Delete old branches from {largest_repo}")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        for bpw in bpws:
            print(f"  [DRY RUN] Would delete branch {fmt_bpw(bpw)}bpw")
    else:
        for bpw in bpws:
            branch = f"{fmt_bpw(bpw)}bpw"
            try:
                api.delete_branch(largest_repo, branch=branch, repo_type="model")
                print(f"  Deleted: {branch}")
                time.sleep(0.5)
            except Exception as e:
                print(f"  Could not delete {branch}: {e}")

    # ── Done ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("Dry run complete. Remove --dry-run to execute for real.")
    else:
        print(f"Done! Created {len(bpws)} single-BPW repos:")
    for bpw in bpws:
        print(f"  https://huggingface.co/{target_id(bpw)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
