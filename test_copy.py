#!/usr/bin/env python3
"""
Test two approaches for copying files from a branch to main
within the same HuggingFace repo using only the HuggingFace API.
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationAdd, CommitOperationDelete
from huggingface_hub.utils import HfHubHTTPError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test HF API approaches for copying files from a branch to main"
    )
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g. 'username/my-model')")
    parser.add_argument("branch", help="Source branch name to copy files from")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type (default: model)",
    )
    return parser.parse_args()


def approach1_commit_operation_copy(api: HfApi, repo_id: str, branch: str, repo_type: str):
    """
    Approach 1: CommitOperationCopy (same repo, branch to main)

    Uses api.list_repo_tree() to enumerate files on the branch, then
    creates CommitOperationCopy operations with src_revision=branch so
    the server resolves each source blob from the branch while writing
    the copies to main.
    """
    print("\n" + "=" * 60)
    print("APPROACH 1: CommitOperationCopy (same repo, branch -> main)")
    print("=" * 60)

    # Step 1: List all files on the branch
    print(f"\n[1] Listing files on branch '{branch}' of '{repo_id}'...")
    try:
        tree_items = list(
            api.list_repo_tree(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=branch,
                recursive=True,
            )
        )
    except HfHubHTTPError as e:
        print(f"  FAILED to list repo tree: {e}")
        return False

    files = [item for item in tree_items if item.rfilename]
    if not files:
        print("  No files found on branch.")
        return False

    print(f"  Found {len(files)} file(s):")
    for f in files:
        print(f"    - {f.rfilename}")

    # Step 2: Build CommitOperationCopy for each file
    print(f"\n[2] Building CommitOperationCopy operations (src_revision='{branch}')...")
    operations = []
    for f in files:
        op = CommitOperationCopy(
            src_path_in_repo=f.rfilename,
            path_in_repo=f.rfilename,
            src_revision=branch,  # resolve source blob from the branch
        )
        operations.append(op)
        print(f"    Copy: {f.rfilename}  (src_revision={branch})")

    # Step 3: Commit to main
    print(f"\n[3] Creating commit on main with {len(operations)} copy operation(s)...")
    try:
        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operations,
            commit_message=f"[test_copy.py] Copy files from branch '{branch}' to main via CommitOperationCopy",
            revision="main",
        )
        print(f"  SUCCESS! Commit URL: {commit_info.commit_url}")
        return True
    except HfHubHTTPError as e:
        print(f"  FAILED to create commit: {e}")
        return False
    except Exception as e:
        print(f"  UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def get_lfs_pointer_text(oid: str, size: int) -> str:
    """Return a well-formed Git LFS pointer file as a string."""
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{oid}\n"
        f"size {size}\n"
    )


def approach2_cross_repo_lfs_oid_reuse(api: HfApi, repo_id: str, branch: str, repo_type: str):
    """
    Approach 2: Cross-repo LFS OID reuse

    For each file on the branch, retrieve its LFS metadata (oid + size).
    Create a *new* scratch repo, then attempt to create a commit that
    contains CommitOperationAdd operations whose content is a raw LFS
    pointer pointing at those OIDs.  If the HF server accepts the
    pointer without requiring a fresh upload it means OIDs are shared
    globally; if it rejects them (404 / validation error) they are not.
    """
    print("\n" + "=" * 60)
    print("APPROACH 2: Cross-repo LFS OID reuse (pointer smuggling)")
    print("=" * 60)

    # Step 1: Collect LFS metadata for files on the branch
    print(f"\n[1] Fetching LFS metadata for files on branch '{branch}' of '{repo_id}'...")
    try:
        tree_items = list(
            api.list_repo_tree(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=branch,
                recursive=True,
            )
        )
    except HfHubHTTPError as e:
        print(f"  FAILED to list repo tree: {e}")
        return False

    lfs_files = []
    for item in tree_items:
        if item.lfs is not None:
            lfs_files.append(item)
            print(f"    LFS file: {item.rfilename}  oid={item.lfs.sha256}  size={item.lfs.size}")

    if not lfs_files:
        print("  No LFS files found on branch — nothing to test.")
        return False

    # Step 2: Create a new temporary scratch repo
    scratch_repo_id = f"{repo_id.split('/')[0]}/test-copy-scratch-{os.getpid()}"
    print(f"\n[2] Creating scratch repo '{scratch_repo_id}'...")
    try:
        api.create_repo(
            repo_id=scratch_repo_id,
            repo_type=repo_type,
            private=True,
            exist_ok=False,
        )
        print(f"  Created: {scratch_repo_id}")
    except HfHubHTTPError as e:
        print(f"  FAILED to create scratch repo: {e}")
        return False

    try:
        # Step 3: Build CommitOperationAdd with raw LFS pointer content
        print(f"\n[3] Building CommitOperationAdd with raw LFS pointer content...")
        operations = []
        for item in lfs_files:
            pointer_text = get_lfs_pointer_text(item.lfs.sha256, item.lfs.size)
            print(f"    Pointer for {item.rfilename}:")
            for line in pointer_text.strip().splitlines():
                print(f"      {line}")
            op = CommitOperationAdd(
                path_in_repo=item.rfilename,
                # Pass the pointer as raw bytes — we want the server to
                # interpret this as an LFS pointer rather than uploading
                # it as a plain text file.
                path_or_fileobj=pointer_text.encode(),
            )
            operations.append(op)

        # Step 4: Attempt the commit on the scratch repo
        print(f"\n[4] Attempting commit on scratch repo '{scratch_repo_id}'...")
        try:
            commit_info = api.create_commit(
                repo_id=scratch_repo_id,
                repo_type=repo_type,
                operations=operations,
                commit_message="[test_copy.py] Cross-repo LFS OID reuse test",
                revision="main",
            )
            print(f"  SUCCESS! Commit URL: {commit_info.commit_url}")
            print(
                "  INTERPRETATION: HF accepted the pointer — LFS OIDs may be globally accessible."
            )
            return True
        except HfHubHTTPError as e:
            print(f"  FAILED (HTTP error): {e}")
            print(
                "  INTERPRETATION: HF rejected the pointer — OIDs are likely scoped per-repo."
            )
            return False
        except Exception as e:
            print(f"  UNEXPECTED ERROR: {type(e).__name__}: {e}")
            return False

    finally:
        # Clean up scratch repo regardless of outcome
        print(f"\n[5] Cleaning up scratch repo '{scratch_repo_id}'...")
        try:
            api.delete_repo(repo_id=scratch_repo_id, repo_type=repo_type)
            print("  Deleted scratch repo.")
        except HfHubHTTPError as e:
            print(f"  WARNING: Could not delete scratch repo: {e}")


def main():
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Repo       : {args.repo_id}")
    print(f"Branch     : {args.branch}")
    print(f"Repo type  : {args.repo_type}")

    # --- Approach 1 ---
    result1 = approach1_commit_operation_copy(api, args.repo_id, args.branch, args.repo_type)

    # --- Approach 2 ---
    result2 = approach2_cross_repo_lfs_oid_reuse(api, args.repo_id, args.branch, args.repo_type)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Approach 1 (CommitOperationCopy, same repo): {'PASSED' if result1 else 'FAILED'}")
    print(f"  Approach 2 (cross-repo LFS OID reuse):       {'PASSED' if result2 else 'FAILED'}")
    print()


if __name__ == "__main__":
    main()
