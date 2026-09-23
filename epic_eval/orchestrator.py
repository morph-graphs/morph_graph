"""Sandboxed EPIC-Morph MCQ evaluation on SLURM.

Checks out the eval code (this repo) at ``--commit`` and the question bank
(epic-morph) at the commit pinned in the config into a run directory named
``<code sha8>_<tag>_cfg<confighash>``, then submits up to three array jobs:

    prep   one task per video: encode full video + [0, t_ask] clips   (skipped if media valid)
    eval   one task per video: query the model, write answers/<video>.jsonl
    score  one task: scores.json / scores.txt

Every job runs from the checked-out code, never from the working tree. The
orchestrator submits and exits; there is no login-node polling. Rerunning the
same command recomputes the work list from disk and submits only what is
missing, so it is also the recovery path after preemption or failures.

    python -m epic_eval.orchestrator --config epic_eval/configs/gemini-3.8-flash-high.json \\
        --commit <sha> --dryrun
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .common import (RUN_CONFIG, config_hash, load_json, load_questions, media_done_marker,
                     media_hash, media_path, group_by_cutoff)
from .run_eval import done_qids

NEXRUN = Path("~/.claude/bin/nexrun").expanduser()


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def mirror(runs_root: Path, url: str) -> Path:
    """One bare mirror per repo under <runs_root>/_mirrors, fetched on every run."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", url.rstrip("/").split("/")[-1].split(":")[-1])
    path = runs_root / "_mirrors" / name
    if not path.exists():
        print(f"mirroring {url} -> {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--mirror", url, str(path))
    else:
        git("remote", "set-url", "origin", url, cwd=path)
        git("fetch", "--prune", "origin", cwd=path)
    return path


def resolve(mirror_path: Path, ref: str) -> str:
    try:
        return git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=mirror_path)
    except subprocess.CalledProcessError:
        sys.exit(f"FATAL: {ref!r} is not a commit in {mirror_path} -- is it pushed?")


def checkout(mirror_path: Path, sha: str, dest: Path) -> None:
    if not dest.exists():
        git("clone", "--quiet", str(mirror_path), str(dest))
    else:
        git("fetch", "--quiet", str(mirror_path), cwd=dest)
    git("checkout", "--quiet", "--force", sha, cwd=dest)
    git("clean", "-fdq", cwd=dest)
    head = git("rev-parse", "HEAD", cwd=dest)
    if head != sha:
        sys.exit(f"FATAL: {dest} is at {head}, expected {sha}")
    print(f"  {dest.name} @ {sha[:10]}")


# ---------------------------------------------------------------- work lists

def media_ready(cfg: dict, video: str, cutoffs) -> bool:
    media_dir = Path(cfg["resolved"]["media_dir"])
    marker = media_done_marker(media_dir, video)
    if not marker.exists():
        return False
    listed = json.loads(marker.read_text())["files"]
    want = [media_path(media_dir, video, c) for c in cutoffs]
    return all(p.name in listed and p.exists() and p.stat().st_size == listed[p.name]["bytes"] for p in want)


# ---------------------------------------------------------------- slurm

def write_array(cfg: dict, step: str, commands: list[str], dependency: str | None) -> Path:
    run_dir, s = Path(cfg["resolved"]["run_dir"]), cfg["slurm"][step]
    slurm_dir, log_dir = run_dir / "slurm", run_dir / "slurm" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Timestamped: a pending array from an earlier submission keeps reading its own list.
    cmd_file = slurm_dir / f"{step}_{stamp}_commands.txt"
    cmd_file.write_text("\n".join(commands) + "\n")
    array = f"1-{len(commands)}" + (f"%{s['max_parallel']}" if s.get("max_parallel") else "")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=ee-{step}-{cfg['run']['tag']}"[:60],
        f"#SBATCH --output={log_dir}/{step}_%A_%a.out",
        f"#SBATCH --error={log_dir}/{step}_%A_%a.err",
        f"#SBATCH --account={s['account']}",
        f"#SBATCH --partition={s['partition']}",
        f"#SBATCH --qos={s['qos']}",
        f"#SBATCH --cpus-per-task={s['cpus']}",
        f"#SBATCH --mem={s['mem']}",
        f"#SBATCH --time={s['time']}",
        f"#SBATCH --array={array}",
        "#SBATCH --requeue",
    ]
    if s.get("gres"):
        lines.append(f"#SBATCH --gres={s['gres']}")
    if dependency:
        lines.append(f"#SBATCH --dependency=afterany:{dependency}")
    lines += [
        "",
        "set -eo pipefail",
        'echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} restarts=${SLURM_RESTART_COUNT:-0}"',
        "source /etc/profile.d/modules.sh",
        f"module load {cfg['paths']['ffmpeg_module']}",
        f"cd {cfg['resolved']['code_dir']}",
        "export PYTHONUNBUFFERED=1",
        f'CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {cmd_file})',
        'echo "$CMD"',
        'eval "$CMD"',
        "",
    ]
    script = slurm_dir / f"{step}_{stamp}.slurm"
    script.write_text("\n".join(lines))
    return script


def submit(cfg: dict, step: str, script: Path, n_tasks: int, note: str, dryrun: bool) -> str | None:
    s = cfg["slurm"][step]
    print(f"\n=== {step}: {n_tasks} task(s) | {s['partition']}/{s['qos']} {s['cpus']}cpu {s['mem']} "
          f"{s.get('gres') or 'no-gpu'} {s['time']} | {script}")
    print(script.read_text())
    cmds = Path(re.search(r"sed -n \S+ (\S+)\)", script.read_text()).group(1)).read_text().splitlines()
    print(f"--- commands ({len(cmds)}), first: {cmds[0]}")
    if dryrun:
        return f"DRYRUN-{step}"
    run_dir = cfg["resolved"]["run_dir"]
    out = subprocess.run([str(NEXRUN), "launch", "--tag", f"{cfg['run']['tag']}-{step}", "--out", run_dir,
                          "--note", note, "--", "sbatch", str(script)],
                         capture_output=True, text=True)
    print(out.stdout.strip(), out.stderr.strip())
    m = re.search(r"submitted (\d+)", out.stdout)
    if out.returncode != 0 or not m:
        sys.exit(f"FATAL: submission of {step} failed")
    return m.group(1)


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--commit", required=True, help="eval-code commit (any ref the mirror can resolve)")
    p.add_argument("--dryrun", action="store_true", help="check out and write scripts, but submit nothing")
    p.add_argument("--steps", default="prep,eval,score")
    p.add_argument("--videos", help="comma-separated subset (smoke tests)")
    p.add_argument("--limit", type=int, help="max questions per video per eval task (smoke tests)")
    p.add_argument("--code_repo_url", help="override config code.repo_url (e.g. a local path for testing)")
    p.add_argument("--questions_repo_url", help="override config questions.repo_url")
    p.add_argument("--env_name", help="override the auto-generated run directory name")
    p.add_argument("--note", default="", help="why this run exists (goes to the nexrun ledger)")
    a = p.parse_args()

    cfg = load_json(a.config)
    runs_root = Path(cfg["run"]["runs_root"])
    code_url = a.code_repo_url or cfg["code"]["repo_url"]
    q_url = a.questions_repo_url or cfg["questions"]["repo_url"]

    # Resolve both commits to full SHAs *before* hashing, so a short and a long
    # spelling of the same commit land in the same run directory.
    code_mirror, q_mirror = mirror(runs_root, code_url), mirror(runs_root, q_url)
    cfg["code"]["commit"] = resolve(code_mirror, a.commit)
    cfg["questions"]["commit"] = resolve(q_mirror, cfg["questions"]["commit"])
    cfg.pop("resolved", None)
    # slurm resources and runtime concurrency don't change answers, so they don't change the run
    hashed = {k: v for k, v in cfg.items() if k not in ("resolved", "slurm", "runtime")}
    # where encoded clips live is a location, not a result: sharing another run's media keeps the hash
    hashed["paths"] = {k: v for k, v in cfg["paths"].items() if k != "media_root"}
    env = a.env_name or f"{cfg['code']['commit'][:8]}_{cfg['run']['tag']}_cfg{config_hash(hashed)}"
    run_dir = runs_root / env
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")

    checkout(code_mirror, cfg["code"]["commit"], run_dir / "code_repo")
    checkout(q_mirror, cfg["questions"]["commit"], run_dir / "questions_repo")
    cfg["resolved"] = {
        "run_dir": str(run_dir),
        "code_dir": str(run_dir / "code_repo"),
        "questions_dir": str(run_dir / "questions_repo"),
        "media_dir": str(Path(cfg["paths"].get("media_root") or runs_root / "media") / media_hash(cfg))
                     if cfg["media"]["enabled"] else None,
        "code_repo_url": code_url,
        "questions_repo_url": q_url,
    }
    if not (run_dir / "code_repo" / "epic_eval").is_dir():
        sys.exit(f"FATAL: commit {cfg['code']['commit'][:10]} has no epic_eval/ package")
    prev = run_dir / RUN_CONFIG
    if prev.exists() and {k: v for k, v in load_json(prev).items() if k not in ("slurm", "runtime")} != \
            {k: v for k, v in cfg.items() if k not in ("slurm", "runtime")}:
        print(f"note: {prev} differs from this config outside 'slurm'/'runtime'; overwriting")
    prev.write_text(json.dumps(cfg, indent=2))

    questions = load_questions(cfg["resolved"]["questions_dir"], cfg["questions"]["glob"])
    videos = sorted({q["video"] for q in questions})
    if a.videos:
        wanted = a.videos.split(",")
        unknown = set(wanted) - set(videos)
        if unknown:
            sys.exit(f"FATAL: no questions for {sorted(unknown)}")
        videos = [v for v in videos if v in wanted]
    by_video = {v: [q for q in questions if q["video"] == v] for v in videos}
    n_stream = sum(q["streaming"] for q in questions)
    print(f"questions: {len(questions)} ({n_stream} streaming) over {len(by_video)} videos"
          f" @ epic-morph {cfg['questions']['commit'][:10]}")

    steps = a.steps.split(",")
    py = cfg["paths"]["python"]
    note = a.note or f"EPIC-Morph MCQ eval, {cfg['model']['name']}, video={cfg['media']['enabled']}"
    dep = None

    if "prep" in steps and cfg["media"]["enabled"]:
        todo = [v for v in videos if not media_ready(cfg, v, group_by_cutoff(by_video[v]).keys())]
        print(f"\nprep: {len(todo)}/{len(videos)} videos need media -> {cfg['resolved']['media_dir']}")
        if todo:
            cmds = [f"{py} -m epic_eval.prep_media --run_dir {shlex.quote(str(run_dir))} --video {v}" for v in todo]
            dep = submit(cfg, "prep", write_array(cfg, "prep", cmds, None), len(cmds), note, a.dryrun)

    if "eval" in steps:
        if cfg["model"]["backend"] == "gemini":
            from .backends.gemini import api_key
            if not api_key(cfg["model"]):
                msg = f"no Gemini key in ${cfg['model']['api_key_env']} or {cfg['model']['api_key_file']}"
                if not a.dryrun:
                    sys.exit(f"FATAL: {msg}")
                print(f"WARNING (dryrun): {msg}; a real submission would stop here")
        answers = run_dir / "answers"
        todo = [v for v in videos
                if a.limit or {q["qid"] for q in by_video[v]} - done_qids(answers / f"{v}.jsonl")]
        print(f"\neval: {len(todo)}/{len(videos)} videos have unanswered questions")
        if todo:
            lim = f" --limit {a.limit}" if a.limit else ""
            cmds = [f"{py} -m epic_eval.run_eval --run_dir {shlex.quote(str(run_dir))} --video {v}{lim}" for v in todo]
            dep = submit(cfg, "eval", write_array(cfg, "eval", cmds, dep),
                         len(cmds), note, a.dryrun) or dep

    if "score" in steps:
        cmds = [f"{py} -m epic_eval.score --run_dir {shlex.quote(str(run_dir))}"]
        submit(cfg, "score", write_array(cfg, "score", cmds, dep),
               1, note, a.dryrun)

    print(f"\n{'[DRYRUN] nothing submitted. ' if a.dryrun else ''}run dir: {run_dir}")


if __name__ == "__main__":
    main()
