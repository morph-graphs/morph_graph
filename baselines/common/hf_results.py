#!/usr/bin/env python3
"""Baseline results on the Hugging Face dataset (private, morph-graphs/epic-morph-results by default).

Shared by every baseline under baselines/. One video's results folder (memory, answers, runs.jsonl) lives at
<baseline>/<arm>/<video>/ in the dataset. The memory files (--memory, patterns with {video}; MERIT's by
default) depend only on the video and the model, so a memory is built once and shared by every bank, probe
set and variant answered from it.
- `upload` adds a local folder's files there (runs.jsonl is merged with the dataset's, not replaced). If the dataset already holds a different memory for the video
  (a rebuild), the folder is mirrored instead: remote files it lacks are removed, so no answers made from the
  old memory remain. Run it after the video's stages have finished.
- `fetch-memory` copies the dataset's memory into a local folder (exit 3 when there is none), so `build` does
  not redo it on another machine.
- With MERIT_OUTBOX=<name> set, `upload` touches nothing remote: it copies the folder to <checkout>/outbox/<name>/
  with an `.upload.json` (written last: the folder is complete), for machines whose jobs cannot reach the dataset
  (Zaratan); `relay --dir <that folder>` uploads it later from a machine that can, exactly as `upload` would.
- `earlier-answers` prints the answers files the video and arm already have under any bank (epic-morph BANKS, the
  same variant --suffix): those in the local folder, then the dataset's when its memory is the local one (compared by
  hash). qa reuses their rows for unchanged questions, so a question is not answered again under another bank.
- `download` fetches a whole folder (e.g. the batch10 reference) to <dest>/merit/<arm>/<video>/, so
  MERIT_OUT=<dest>/merit lets run_merit.sh use it. The token comes from HF_TOKEN or the cached `hf auth login`; it needs write access
for upload. MERIT_HF_REPO picks another dataset.

    python baselines/common/hf_results.py upload       --baseline merit --arm m8 --video <V> --dir <local folder>
    python baselines/common/hf_results.py fetch-memory --baseline merit --arm m8 --video <V> --dir <local folder>
    python baselines/common/hf_results.py download     --baseline merit --arm m8 --video <V> --dest <local root>
    python baselines/common/hf_results.py relay        --dir <checkout>/outbox/<name>
    python baselines/common/hf_results.py earlier-answers --baseline merit --arm m8 --video <V> --dir <local folder> \
        --epic <epic-morph checkout> --suffix=<_tag or empty>
    --memory '{video}_graph.pkl' '{video}_captions.json'   (another baseline's memory files)
    --memory none                                         (a baseline without a memory: baselines/api)
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

REPO = os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results")
OUTBOX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outbox")


MEMORY = ["{video}_30sec.json", "{video}_30sec_4_key.json"]   # MERIT's; other baselines pass --memory


def memory_files(video):
    return [m.format(video=video) for m in MEMORY]


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


def file_hash(p, git=False):
    """sha256 of a file, or with git=True its git blob id (what the Hub reports for a file not stored in LFS)."""
    h = hashlib.sha1(b"blob %d\0" % os.path.getsize(p)) if git else hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def same_memory(path, video, local):
    """True when the dataset's memory for the video is the one in the folder `local` (by hash, nothing downloaded)."""
    info = {os.path.basename(i.path): i for i in HfApi().get_paths_info(REPO, [f"{path}/{f}" for f in memory_files(video)],
                                                                          repo_type="dataset")}
    for f in memory_files(video):
        p, i = os.path.join(local, f), info.get(f)
        if not (i and os.path.isfile(p)):
            return False
        if (file_hash(p) != i.lfs.sha256) if i.lfs else (file_hash(p, git=True) != i.blob_id):
            return False
    return bool(MEMORY)


def earlier_answers(path, a):
    """Paths of this video and arm's answers files under every bank: local ones, then the dataset's."""
    sys.path.insert(0, os.path.join(a.epic, "v3"))
    from export_bank import BANKS
    names = [f"answers_{b}{a.suffix}.json.gz" for b in BANKS]
    out = [os.path.join(a.dir, n) for n in names if os.path.isfile(os.path.join(a.dir, n))]
    try:   # optional: without the dataset, only the questions it alone answered are asked again
        there = {os.path.basename(i.path) for i in HfApi().list_repo_tree(REPO, path_in_repo=path, repo_type="dataset")}
        remote = [n for n in names if n in there]
        if remote and same_memory(path, a.video, a.dir):
            out += [hf_hub_download(REPO, f"{path}/{n}", repo_type="dataset") for n in remote]
        elif remote:
            print(f"{REPO}/{path} holds another memory: its answers are not reused", file=sys.stderr)
    except Exception as e:
        print(f"earlier answers in {REPO}/{path} not read ({type(e).__name__}: {str(e)[:160]})", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["upload", "fetch-memory", "download", "relay", "earlier-answers"])
    ap.add_argument("--baseline", default="merit", help="top-level folder in the dataset (merit, worldmm, ...)")
    ap.add_argument("--memory", nargs="+", help="the memory's file names, with {video}; default MERIT's; `none` for a baseline without one")
    ap.add_argument("--arm")
    ap.add_argument("--video")
    ap.add_argument("--dir", help="upload / fetch-memory: the local results folder of this video and arm; relay: an outbox folder")
    ap.add_argument("--dest", help="download: local root; files land in <dest>/merit/<arm>/<video>/")
    ap.add_argument("--epic", help="earlier-answers: the epic-morph checkout whose v3/export_bank.py BANKS names the banks")
    ap.add_argument("--suffix", default="", help="earlier-answers: the variant suffix of the answers files (_<tag>, or empty)")
    a = ap.parse_args()
    if a.action == "relay":   # an outbox folder: upload it with the arguments its job's upload was given
        meta = json.load(open(os.path.join(a.dir, ".upload.json")))
        a.action, a.baseline, a.arm, a.video, a.memory = "upload", meta["baseline"], meta["arm"], meta["video"], meta["memory"]
    if not (a.arm and a.video):
        ap.error("--arm and --video are required")
    if a.memory:   # "none": a baseline with no memory (API models); its folder is answers + runs.jsonl
        MEMORY[:] = [] if a.memory == ["none"] else a.memory
    path = f"{a.baseline}/{a.arm}/{a.video}"
    if a.action == "earlier-answers":
        print("\n".join(earlier_answers(path, a)))
    elif a.action == "fetch-memory":
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
        name = os.environ.get("MERIT_OUTBOX")
        if name:   # copy to the outbox; a relay uploads it (it merges runs.jsonl and checks the memory then)
            dest, tmp = os.path.join(OUTBOX, name), os.path.join(OUTBOX, f".{name}.tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            for f in files:
                shutil.copy(os.path.join(a.dir, f), os.path.join(tmp, f))
            json.dump(dict(baseline=a.baseline, arm=a.arm, video=a.video, memory=MEMORY or ["none"]),
                      open(os.path.join(tmp, ".upload.json"), "w"))
            shutil.rmtree(dest, ignore_errors=True)   # an earlier run of the same task: this one replaces it
            os.rename(tmp, dest)
            print(f"put {len(files)} files in the outbox {dest} for the relay (not uploaded): {', '.join(files)}")
            return
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
                                      commit_message=f"{a.baseline} {a.arm} {a.video}" + (" (rebuilt memory)" if rebuilt else ""))
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
