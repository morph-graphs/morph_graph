#!/usr/bin/env python3
"""Make (or reopen) the epic_eval run folder of one API model config and one bank, without Slurm or git mirrors.

epic_eval's steps (prep_media, run_eval, score) only need a run folder holding reproducibility_config.json
and the bank files. The orchestrator builds one from pinned commits and submits Slurm jobs; this builds one
from a model config in baselines/api/configs/ (media, model, runtime: no paths) and a bank of an epic-morph
checkout (generated there by v3/export_bank.py, not committed), for running the steps directly, e.g. in the
Docker image. The bank files are copied into the run folder, so the run keeps the questions it was asked: a
bank file that has changed since stops here (answers are joined to questions by wording and ask time, which
would not notice edited options). A config whose model or media settings changed under the same name stops too.

    python baselines/api/epic/make_run.py --config baselines/api/configs/gpt-5-f1024.json --bank batch10_agree2 \
        --epic <epic-morph checkout> --videos <HD-EPIC videos> --work baselines/api/work
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from epic_eval.common import RUN_CONFIG, media_hash   # noqa: E402


def commit(repo, env):
    """The checkout's commit: from the environment (docker/run.sh passes it), else from git."""
    if os.environ.get(env):
        return os.environ[env]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="baselines/api/configs/<name>.json")
    ap.add_argument("--bank", required=True, help="bank name: <epic>/outputs/v3/banks/<bank>/*_questions.json")
    ap.add_argument("--epic", required=True, help="epic-morph checkout")
    ap.add_argument("--videos", required=True, help="HD-EPIC videos (<P>/<video>.mp4)")
    ap.add_argument("--work", required=True, help="work root: runs/<config>_<bank>/ and media/<hash>/")
    ap.add_argument("--media_dir", default=None, help="reuse a folder of clips encoded with the same media settings")
    a = ap.parse_args()

    name = os.path.splitext(os.path.basename(a.config))[0]
    run = os.path.join(os.path.abspath(a.work), "runs", f"{name}_{a.bank}")
    rel = f"outputs/v3/banks/{a.bank}"
    src = sorted(glob.glob(os.path.join(a.epic, rel, "*_questions.json")))
    if not src:
        sys.exit(f"no bank at {os.path.join(a.epic, rel)}: generate it in epic-morph with `python v3/export_bank.py {a.bank}`")
    dst = os.path.join(run, "questions_repo", rel)
    os.makedirs(dst, exist_ok=True)
    for f in src:
        d = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(d):
            shutil.copy(f, d)
        elif open(f, "rb").read() != open(d, "rb").read():
            sys.exit(f"{os.path.basename(f)} differs from the copy this run was asked ({d}): "
                     f"a changed bank needs its own run (another bank name, or another --work)")

    cfg = json.load(open(a.config))
    cfg = {"run": {"tag": name, "runs_root": os.path.dirname(run)},
           "code": {"commit": commit(REPO, "API_REPO_COMMIT")},
           "questions": {"glob": f"{rel}/*_questions.json", "commit": commit(a.epic, "API_EPIC_COMMIT")},
           "paths": {"video_dir": os.path.abspath(a.videos)},
           "media": cfg["media"], "model": cfg["model"], "runtime": cfg.get("runtime", {})}
    cfg["resolved"] = {"run_dir": run, "questions_dir": os.path.join(run, "questions_repo"), "bank": a.bank,
                       "media_dir": os.path.abspath(a.media_dir) if a.media_dir
                       else os.path.join(os.path.abspath(a.work), "media", media_hash(cfg))}
    path = os.path.join(run, RUN_CONFIG)
    if os.path.exists(path):
        old = json.load(open(path))
        for k in ("media", "model"):   # what changes answers; commits and paths are those of the first launch
            if old[k] != cfg[k]:
                sys.exit(f"{path}: '{k}' differs from {a.config}. A changed setting is another config name.")
        cfg["code"], cfg["questions"] = old["code"], old["questions"]
        if old == cfg:   # other videos of this run may be reading the file right now: leave it alone
            print(run)
            return
    with open(path + f".{os.getpid()}.tmp", "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(f.name, path)   # atomic, for the same reason
    print(run)


if __name__ == "__main__":
    main()
