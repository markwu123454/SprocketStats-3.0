"""Push and pull every module's data/ folder to and from the SprocketStats/modelfiles
Hugging Face repo. Git is not involved: commit and push code with git as usual.

Files already on Hugging Face are treated as immutable. `push` only ever adds files HF
doesn't have yet; replacing a file whose contents differ, or deleting one, takes an
explicit --override. The same goes the other way: `pull` only adds files you don't
have, and never overwrites a local file that differs from HF's copy unless you pass
--override. Overridden versions are not lost: every change is a commit on the HF repo,
so old versions stay reachable through its history.

This is a guard rail in this script, not a lock on the server: anyone with write access
can still change files through the web UI or the hf CLI. The repo is read once per run,
and every write passes that commit as parent_commit, so if someone else pushed in the
meantime the write is refused instead of silently replacing their file. Rerun to see
the new state.

HF paths mirror repo paths (hrnet/data/checkpoints/best.pt is stored at exactly that
path), so files land where the code reads them.

Usage, from the repo root with the project venv:
    python hf_sync.py status                        what push and pull would do; changes nothing
    python hf_sync.py push [--override]             upload new files (--override: also replace changed ones)
    python hf_sync.py pull [--override]             download missing files (--override: also replace changed ones)
    python hf_sync.py delete PATH... --override     delete files from HF (local copies are kept)
Log in once per machine (a write token is needed for push/delete):
    .venv\\Scripts\\hf auth login
"""

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ID = "SprocketStats/modelfiles"
REPO_TYPE = "model"
ROOT = Path(__file__).resolve().parent
MODULES = ("homography", "broadcast", "hrnet", "yolo", "ocr", "mtmct")
# Caches that can always be fetched again from their real source, and are far too large
# to mirror: the local copy of the R2 training-image bucket (up to ~150 GB) and
# downloaded YouTube matches.
SKIP_PREFIXES = ("hrnet/data/images/", "hrnet/data/youtube_cache/")
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
SKIP_DIRS = {"__pycache__", ".cache"}


def in_scope(rel: str) -> bool:
    """Is this repo-relative posix path one of the files this script syncs?"""
    parts = rel.split("/")
    return (len(parts) >= 3 and parts[0] in MODULES and parts[1] == "data"
            and not rel.startswith(SKIP_PREFIXES) and parts[-1] not in SKIP_NAMES
            and not SKIP_DIRS & set(parts[:-1]))


def local_files(root: Path) -> dict[str, Path]:
    out = {}
    for m in MODULES:
        for p in (root / m / "data").rglob("*"):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                if in_scope(rel):
                    out[rel] = p
    return out


def remote_state(api):
    """(head commit sha, {path: RepoFile}) for the in-scope files on HF."""
    if not api.repo_exists(REPO_ID, repo_type=REPO_TYPE):
        sys.exit(f"{REPO_ID} isn't visible to this token. Either it doesn't exist (create it "
                 f"once: hf repo create {REPO_ID}), or it's private and the token can't see "
                 f"the org: a fine-grained token needs {REPO_ID.split('/')[0]} added under its "
                 f"org or repository permissions, since it doesn't inherit org membership.")
    head = api.repo_info(REPO_ID, repo_type=REPO_TYPE).sha
    files = {}
    for entry in api.list_repo_tree(REPO_ID, recursive=True, repo_type=REPO_TYPE, revision=head):
        if hasattr(entry, "blob_id") and in_scope(entry.path):  # files, not folders
            files[entry.path] = entry
    return head, files


def _hash(path: Path, algo: str, prefix: bytes = b"") -> str:
    h = hashlib.new(algo, prefix)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_content(path: Path, remote) -> bool:
    """Does the local file match HF's copy? Large files are compared by their LFS sha256,
    small ones by their git blob sha1, so nothing has to be downloaded. A file HF gives
    neither hash for counts as different, so it can't slip past the override."""
    size = path.stat().st_size
    if size != remote.size:
        return False
    if remote.lfs is not None:
        return _hash(path, "sha256") == remote.lfs.sha256
    return _hash(path, "sha1", b"blob %d\0" % size) == remote.blob_id


def compare(root: Path, api):
    head, remote = remote_state(api)
    local = local_files(root)
    new = sorted(set(local) - set(remote))
    missing = sorted(set(remote) - set(local))
    differ = sorted(p for p in set(local) & set(remote) if not same_content(local[p], remote[p]))
    same = len(set(local) & set(remote)) - len(differ)
    return head, local, new, missing, differ, same


def _list(title, paths, limit=40):
    if paths:
        print(f"{title} ({len(paths)}):")
        for p in paths[:limit]:
            print(f"  {p}")
        if len(paths) > limit:
            print(f"  ... and {len(paths) - limit} more")


def cmd_status(api, root=ROOT):
    _, _, new, missing, differ, same = compare(root, api)
    print(f"{REPO_ID}: {same} files identical locally and on HF")
    _list("only local -- `push` uploads", new)
    _list("only on HF -- `pull` downloads", missing)
    _list("different on each side -- needs --override (push: replace HF's; pull: replace yours)", differ)
    return 0


def cmd_push(api, override=False, root=ROOT):
    from huggingface_hub import CommitOperationAdd
    head, local, new, _, differ, _ = compare(root, api)
    replace = differ if override else []
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(local[p])) for p in new + replace]
    if ops:
        msg = f"hf_sync push: {len(new)} added" + (f", {len(replace)} replaced (--override)" if replace else "")
        desc = "Replaced with --override:\n" + "\n".join(replace) if replace else None
        api.create_commit(REPO_ID, ops, commit_message=msg, commit_description=desc,
                          repo_type=REPO_TYPE, parent_commit=head)
        print(msg)
    else:
        print("nothing new to push")
    if differ and not override:
        _list("NOT pushed: already on HF with different contents (rerun with --override to replace)", differ)
        return 1
    return 0


def cmd_pull(api, override=False, root=ROOT):
    from huggingface_hub import hf_hub_download
    head, _, _, missing, differ, _ = compare(root, api)
    for p in missing + (differ if override else []):
        hf_hub_download(REPO_ID, p, repo_type=REPO_TYPE, revision=head, local_dir=root,
                        force_download=p in differ)
    print(f"pulled {len(missing)} new" + (f", replaced {len(differ)} local (--override)" if override and differ else ""))
    if differ and not override:
        _list("NOT pulled: your copy differs from HF's (rerun with --override to replace yours)", differ)
        return 1
    return 0


def cmd_delete(api, paths, override=False, root=ROOT):
    from huggingface_hub import CommitOperationDelete
    paths = [Path(p).resolve().relative_to(root).as_posix() if Path(p).is_absolute() else Path(p).as_posix()
             for p in paths]
    outside = [p for p in paths if not in_scope(p)]
    if outside:
        sys.exit("not synced data paths: " + ", ".join(outside))
    head, remote = remote_state(api)
    absent = [p for p in paths if p not in remote]
    if absent:
        sys.exit("not on HF: " + ", ".join(absent))
    if not override:
        _list("files on HF are immutable; rerun with --override to delete", paths)
        return 1
    ops = [CommitOperationDelete(path_in_repo=p) for p in paths]
    api.create_commit(REPO_ID, ops, commit_message=f"hf_sync delete: {len(paths)} files (--override)",
                      commit_description="\n".join(paths), repo_type=REPO_TYPE, parent_commit=head)
    print(f"deleted {len(paths)} from {REPO_ID} (local copies kept)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="See the module docstring for the immutability rules.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for name in ("push", "pull"):
        sub.add_parser(name).add_argument("--override", action="store_true")
    d = sub.add_parser("delete")
    d.add_argument("paths", nargs="+")
    d.add_argument("--override", action="store_true")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi
    api = HfApi()
    if args.cmd == "status":
        return cmd_status(api)
    if args.cmd == "push":
        return cmd_push(api, args.override)
    if args.cmd == "pull":
        return cmd_pull(api, args.override)
    return cmd_delete(api, args.paths, args.override)


if __name__ == "__main__":
    sys.exit(main())
