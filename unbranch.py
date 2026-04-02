#!/usr/bin/env python3
"""
unbranch.py — Split a HuggingFace repo with branched quantizations into
separate single-BPW repos.

WARNING: This script performs destructive, irreversible operations. It renames
repos, overwrites main branches, and deletes branches. Always use --dry-run
first to preview what will happen.

Usage:
    export HF_TOKEN=hf_...
    python unbranch.py \
        --author UnstableLlama \
        --repo-name Qwen3.5-4B-exl3 \
        --bpws 2.10 3.00 4.00 5.00 6.00

How it works (no git, no model file downloads — pure HF API):
    1. Downloads ONLY the README and rewrites branch links.
    2. For each smaller BPW:
       a. CommitOperationCopy: copy branch files → parent's main
       b. duplicate_repo: snapshot parent's main → new single-BPW repo
       c. Restore parent's main to its original state
    3. For the largest BPW: copy branch → parent's main, rename parent.
    4. Verify all repos have files.
    5. Delete old branches from the renamed parent.

Requires: huggingface_hub  (pip install huggingface_hub)
"""

import argparse
import os
import sys
import time

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_bpw(bpw: float) -> str:
    """Format a BPW value to two decimal places."""
    return f"{bpw:.2f}"


def make_target_name(repo_name: str, bpw: float) -> str:
    """Build the single-BPW repo name by inserting the BPW before the quant suffix.

    Example: Qwen3.5-4B-exl3  +  4.00  →  Qwen3.5-4B-4.00bpw-exl3
    """
    for suffix in ("-exl3", "-exl2", "-gguf", "-gptq", "-awq"):
        if repo_name.endswith(suffix):
            base = repo_name[: -len(suffix)]
            return f"{base}-{fmt_bpw(bpw)}bpw{suffix}"
    return f"{repo_name}-{fmt_bpw(bpw)}bpw"


def rewrite_readme(readme: str, author: str, repo_name: str, bpws: list[float]) -> str:
    """Replace branch-style URLs/commands with single-repo equivalents."""
    parent = f"{author}/{repo_name}"

    for bpw in bpws:
        b = fmt_bpw(bpw)
        branch = f"{b}bpw"
        target = f"{author}/{make_target_name(repo_name, bpw)}"

        readme = readme.replace(f"{parent}/tree/{branch}", target)
        readme = readme.replace(f'{parent} --revision "{branch}"', target)
        readme = readme.replace(f"{parent} --revision {branch}", target)

    for bpw in bpws:
        b = fmt_bpw(bpw)
        old_dir = f"{repo_name}-{b}bpw"
        new_dir = make_target_name(repo_name, bpw)
        if old_dir != new_dir:
            readme = readme.replace(f"--local-dir ./{old_dir}", f"--local-dir ./{new_dir}")

    return readme


# ── Core API Operations ─────────────────────────────────────────────────────

def list_branch_files(api, repo_id: str, branch: str) -> list:
    """List all files on a branch."""
    items = list(api.list_repo_tree(
        repo_id=repo_id, revision=branch, recursive=True, repo_type="model",
    ))
    return [item for item in items if hasattr(item, "rfilename")]


def list_main_files(api, repo_id: str) -> list:
    """List all files on main."""
    return list_branch_files(api, repo_id, "main")


def copy_branch_to_main(api, repo_id: str, branch: str, readme_text: str):
    """Server-side copy: overwrite main with branch content + updated README."""
    print(f"    Listing files on branch {branch}...")
    branch_files = list_branch_files(api, repo_id, branch)
    print(f"    Found {len(branch_files)} file(s)")

    operations = []

    # Copy every file from the branch (except README which we'll add fresh)
    for f in branch_files:
        if f.rfilename == "README.md":
            continue
        operations.append(CommitOperationCopy(
            src_path_in_repo=f.rfilename,
            path_in_repo=f.rfilename,
            src_revision=branch,
        ))

    # Add the rewritten README
    operations.append(CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj=readme_text.encode(),
    ))

    # Delete any files on main that aren't on the branch
    main_files = list_main_files(api, repo_id)
    branch_filenames = {f.rfilename for f in branch_files} | {"README.md"}
    for f in main_files:
        if f.rfilename not in branch_filenames:
            operations.append(CommitOperationDelete(path_in_repo=f.rfilename))

    print(f"    Committing {len(operations)} operations to main...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Copy {branch} to main and update README",
        revision="main",
    )


def restore_main(api, repo_id: str, original_main_files: list, original_readme: str):
    """Restore parent's main branch to its original state."""
    print(f"    Restoring parent main...")

    current_files = list_main_files(api, repo_id)
    original_filenames = {f.rfilename for f in original_main_files}

    operations = []

    # Copy back original files from... wait, we need the original commit.
    # Simpler: we saved the original main file list. We can use
    # CommitOperationCopy with src_revision pointing to the commit before
    # our changes. But we don't have that easily.
    #
    # Better approach: just delete everything and re-copy from the original
    # main state. But we need a reference to it.
    #
    # Simplest: before we modify main, we create a backup branch.
    # Then restore from the backup branch.
    raise NotImplementedError("Use save/restore pattern instead")


def save_main_as_backup(api, repo_id: str):
    """Create a backup branch of main before we start modifying it."""
    backup_branch = "_unbranch_backup_main"
    try:
        api.create_branch(repo_id, branch=backup_branch, repo_type="model", revision="main")
        print(f"    Created backup branch: {backup_branch}")
    except Exception:
        # Branch might already exist from a previous run — delete and recreate
        try:
            api.delete_branch(repo_id, branch=backup_branch, repo_type="model")
        except Exception:
            pass
        api.create_branch(repo_id, branch=backup_branch, repo_type="model", revision="main")
        print(f"    Recreated backup branch: {backup_branch}")
    return backup_branch


def restore_main_from_backup(api, repo_id: str, backup_branch: str):
    """Restore main from the backup branch."""
    print(f"    Restoring main from {backup_branch}...")

    backup_files = list_branch_files(api, repo_id, backup_branch)
    current_files = list_main_files(api, repo_id)

    operations = []

    # Copy all files from backup
    for f in backup_files:
        operations.append(CommitOperationCopy(
            src_path_in_repo=f.rfilename,
            path_in_repo=f.rfilename,
            src_revision=backup_branch,
        ))

    # Delete files that aren't in the backup
    backup_filenames = {f.rfilename for f in backup_files}
    for f in current_files:
        if f.rfilename not in backup_filenames:
            operations.append(CommitOperationDelete(path_in_repo=f.rfilename))

    if operations:
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message="Restore main from backup",
            revision="main",
        )
    print(f"    Main restored.")


# ── Main ─────────────────────────────────────────────────────────────────────

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

    largest_branch = f"{fmt_bpw(largest_bpw)}bpw"
    largest_repo = target_id(largest_bpw)

    # ── 1. Download & rewrite README ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 1: Download & rewrite README from {parent_repo}")
    print(f"{'=' * 60}\n")

    readme_file = hf_hub_download(parent_repo, "README.md", token=token)
    with open(readme_file) as f:
        original_readme = f.read()

    readme_text = rewrite_readme(original_readme, args.author, args.repo_name, bpws)

    changed = original_readme != readme_text
    print(f"  README modified: {changed}")
    if changed:
        old_lines = set(original_readme.splitlines())
        new_lines = set(readme_text.splitlines())
        for line in sorted(new_lines - old_lines):
            if "huggingface.co" in line or "hf download" in line:
                print(f"    + {line.strip()[:120]}")

    # ── 2. Create repos for smaller BPWs ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 2: Create repos for smaller BPWs")
    print(f"{'=' * 60}")

    if args.dry_run:
        for bpw in smaller_bpws:
            print(f"  [DRY RUN] Would create {target_id(bpw)} ← branch {fmt_bpw(bpw)}bpw")
    else:
        # Back up parent's main so we can restore it after each cycle
        print(f"\n  Backing up parent main branch...")
        backup_branch = save_main_as_backup(api, parent_repo)
        time.sleep(0.5)

        for bpw in smaller_bpws:
            branch = f"{fmt_bpw(bpw)}bpw"
            repo_id = target_id(bpw)
            print(f"\n  ── {repo_id} (from branch {branch}) ──")

            # a. Copy branch files → parent's main
            print(f"  Copying {branch} → main on parent...")
            copy_branch_to_main(api, parent_repo, branch, readme_text)
            time.sleep(0.5)

            # b. Delete target if it exists from a previous failed run
            try:
                api.repo_info(repo_id, repo_type="model")
                print(f"  Deleting stale repo {repo_id} from previous run...")
                api.delete_repo(repo_id, repo_type="model")
                time.sleep(0.5)
            except Exception:
                pass

            # c. Duplicate parent (main now has this BPW's files) → new repo
            print(f"  Duplicating parent → {repo_id}")
            api.duplicate_repo(
                from_id=parent_repo,
                to_id=repo_id,
                private=args.private,
                repo_type="model",
            )
            time.sleep(0.5)

            # d. Restore parent's main from backup
            restore_main_from_backup(api, parent_repo, backup_branch)
            time.sleep(0.5)

            # e. Delete BPW branches from the new repo (it inherited them)
            for other_bpw in bpws:
                other_branch = f"{fmt_bpw(other_bpw)}bpw"
                try:
                    api.delete_branch(repo_id, branch=other_branch, repo_type="model")
                    time.sleep(0.5)
                except Exception:
                    pass
            # Also delete the backup branch from the duplicate
            try:
                api.delete_branch(repo_id, branch=backup_branch, repo_type="model")
            except Exception:
                pass

            print(f"  ✓ {repo_id}")

        # Clean up backup branch from parent
        try:
            api.delete_branch(parent_repo, branch=backup_branch, repo_type="model")
            print(f"\n  Deleted backup branch from parent.")
        except Exception:
            pass

    # ── 3. Handle parent repo → largest BPW ──────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Step 3: Convert parent repo to largest BPW ({fmt_bpw(largest_bpw)})")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print(f"  [DRY RUN] Would copy {largest_branch} → main on {parent_repo}")
        print(f"  [DRY RUN] Would rename {parent_repo} → {largest_repo}")
    else:
        print(f"  Copying {largest_branch} → main on {parent_repo} (via API)")
        copy_branch_to_main(api, parent_repo, largest_branch, readme_text)
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
                print(f"    api.delete_branch('{largest_repo}', branch='{branch}')")
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
