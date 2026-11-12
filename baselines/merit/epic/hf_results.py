#!/usr/bin/env python3
"""MERIT results on the Hugging Face dataset (private, morph-graphs/epic-morph-results by default).

One video's results folder (memory, answers, runs.jsonl) lives at merit/<arm>/<video>/ in the dataset.
`upload` pushes a local folder there; `download` fetches one (e.g. the committed batch10 reference for
checking a new machine). The token comes from HF_TOKEN or the cached `hf auth login`; it needs write access
for upload. MERIT_HF_REPO picks another dataset.

    python epic/hf_results.py upload   --arm m8 --video P07-20240531-201703 --dir <local folder>
    python epic/hf_results.py download --arm m8 --video P07-20240531-201703 --dest <local root>
"""
import argparse
import os

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["upload", "download"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--dir", help="upload: the local results folder of this video and arm")
    ap.add_argument("--dest", help="download: local root; files land in <dest>/merit/<arm>/<video>/")
    a = ap.parse_args()
    path = f"merit/{a.arm}/{a.video}"
    if a.action == "upload":
        files = sorted(f for f in os.listdir(a.dir) if not f.startswith("."))
        HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=a.dir, path_in_repo=path,
                              ignore_patterns=[".*", "*.partial"], commit_message=f"MERIT {a.arm} {a.video}")
        print(f"uploaded {len(files)} files to {REPO}/{path}: {', '.join(files)}")
    else:
        snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=[f"{path}/*"], local_dir=a.dest)
        got = os.listdir(os.path.join(a.dest, path)) if os.path.isdir(os.path.join(a.dest, path)) else []
        print(f"downloaded {len(got)} files from {REPO}/{path} to {os.path.join(a.dest, path)}")
        if not got:
            raise SystemExit(f"nothing at {REPO}/{path}")


if __name__ == "__main__":
    main()
