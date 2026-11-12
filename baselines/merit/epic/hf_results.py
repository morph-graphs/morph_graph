#!/usr/bin/env python3
"""MERIT results on the Hugging Face dataset (private, morph-graphs/epic-morph-results by default).

One video's results folder (memory, answers, runs.jsonl) lives at merit/<arm>/<video>/ in the dataset.
The memory (<video>_30sec.json, <video>_30sec_4_key.json) depends only on the video and the model, so it is
built once and shared by every bank, probe set and variant answered from it.
- `upload` adds a local folder's files there (runs.jsonl is merged with the dataset's, not replaced). If the dataset already holds a different memory for the video
  (a rebuild), the folder is mirrored instead: remote files it lacks are removed, so no answers made from the
  old memory remain. Run it after the video's stages have finished.
- `fetch-memory` copies the dataset's memory into a local folder (exit 3 when there is none), so `build` does
  not redo it on another machine.
- `download` fetches a whole folder (e.g. the batch10 reference) to <dest>/merit/<arm>/<video>/, so
  MERIT_OUT=<dest>/merit lets run_merit.sh use it. The token comes from HF_TOKEN or the cached `hf auth login`; it needs write access
for upload. MERIT_HF_REPO picks another dataset.

    python epic/hf_results.py upload       --arm m8 --video P07-20240531-201703 --dir <local folder>
    python epic/hf_results.py fetch-memory --arm m8 --video P07-20240531-201703 --dir <local folder>
    python epic/hf_results.py download     --arm m8 --video P07-20240531-201703 --dest <local root>
"""
import argparse
import os
import random
import shutil
import sys
import tempfile
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

REPO = os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results")


def memory_files(video):
    return [f"{video}_30sec.json", f"{video}_30sec_4_key.json"]


def remote_memory(path, video, dest):
    """Download the dataset's memory files for the video into dest; False when it has none. Any other failure
    (network, token, offline mode) raises: taking "unreachable" for "absent" would rebuild a memory for nothing,
    or upload a rebuilt one without removing the answers made from the old one."""
    try:
        for f in memory_files(video):
            shutil.copy(hf_hub_download(REPO, f"{path}/{f}", repo_type="dataset"), os.path.join(dest, f))
        return True
    except EntryNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["upload", "fetch-memory", "download"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--dir", help="upload / fetch-memory: the local results folder of this video and arm")
    ap.add_argument("--dest", help="download: local root; files land in <dest>/merit/<arm>/<video>/")
    a = ap.parse_args()
    path = f"merit/{a.arm}/{a.video}"
    if a.action == "fetch-memory":
        os.makedirs(a.dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                found = remote_memory(path, a.video, tmp)
            except Exception as e:
                print(f"cannot read {REPO}/{path} ({type(e).__name__}: {str(e)[:160]})", file=sys.stderr)
                sys.exit(4)
            if not found:
                print(f"no memory at {REPO}/{path}")
                sys.exit(3)
            for f in memory_files(a.video):
                shutil.copy(os.path.join(tmp, f), os.path.join(a.dir, f))
        print(f"fetched the memory from {REPO}/{path} into {a.dir}")
    elif a.action == "upload":
        files = sorted(f for f in os.listdir(a.dir) if not f.startswith(".") and not f.endswith(".partial"))
        if not all(f in files for f in memory_files(a.video)):   # answers mean nothing without their memory
            raise SystemExit(f"{a.dir} lacks the memory ({', '.join(memory_files(a.video))}): not uploading")
        with tempfile.TemporaryDirectory() as tmp:   # rebuilt memory? then mirror, else add
            rebuilt = remote_memory(path, a.video, tmp) and any(
                open(os.path.join(tmp, f), "rb").read() != open(os.path.join(a.dir, f), "rb").read()
                for f in memory_files(a.video))
            # runs.jsonl is the folder's provenance: keep the dataset's lines, then add this machine's new ones
            try:
                old = open(hf_hub_download(REPO, f"{path}/runs.jsonl", repo_type="dataset")).read().splitlines()
            except EntryNotFoundError:
                old = []
            mine = os.path.join(a.dir, "runs.jsonl")
            new = open(mine).read().splitlines() if os.path.exists(mine) else []
            merged = old + [x for x in new if x not in old]
            if merged:
                open(mine, "w").write("\n".join(merged) + "\n")
        if rebuilt:
            print("the dataset holds a different memory for this video: mirroring, so answers made from it are removed")
        for attempt in range(6):   # parallel jobs uploading other videos can move main under us (HTTP 412)
            try:
                HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=a.dir, path_in_repo=path,
                                      ignore_patterns=[".*", "*.partial"], delete_patterns="*" if rebuilt else None,
                                      commit_message=f"MERIT {a.arm} {a.video}" + (" (rebuilt memory)" if rebuilt else ""))
                break
            except Exception as e:
                if attempt == 5 or "401" in str(e) or "403" in str(e) or "404" in str(e):
                    raise
                wait = 10 * 2 ** attempt * random.uniform(0.5, 1.5)
                print(f"upload failed ({str(e)[:120]}); retrying in {wait:.0f} s", flush=True)
                time.sleep(wait)
        print(f"uploaded {len(files)} files to {REPO}/{path}: {', '.join(files)}")
    else:
        snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=[f"{path}/*"], local_dir=a.dest)
        got = os.listdir(os.path.join(a.dest, path)) if os.path.isdir(os.path.join(a.dest, path)) else []
        print(f"downloaded {len(got)} files from {REPO}/{path} to {os.path.join(a.dest, path)}")
        if not got:
            raise SystemExit(f"nothing at {REPO}/{path}")


if __name__ == "__main__":
    main()
