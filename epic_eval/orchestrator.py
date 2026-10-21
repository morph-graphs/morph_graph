"""Sandboxed EPIC-Morph MCQ evaluation on SLURM.

Checks out the eval code (this repo) at ``--commit`` and the question bank
(epic-morph) at the commit pinned in the config into a run directory named
``<code sha8>_<tag>_cfg<confighash>``, then submits up to three array jobs:

    prep   one task per video: encode full video + [0, t_ask] clips   (skipped if media valid)
    eval   one task per video: query the model, write answers/<video>.jsonl
           (or, with slurm.eval.questions_per_task, one task per chunk of a video's
           pending questions, each writing answers/<video>.<shard>.jsonl)
    score  one task: scores.json / scores.txt

Every job runs from the checked-out code, never from the working tree. The
code comes from a mirror of ``code.repo_url`` or, when the config has none (the
local-model configs), from the clean checkout this orchestrator runs in, so
nothing has to be pushed first. The steps are chained with afterany
dependencies and submitted with sbatch (through nexrun where it exists); each
submission is logged to ``slurm/launches.jsonl``. The orchestrator submits and
exits; there is no login-node polling. Rerunning the same command recomputes
the work list from disk and submits only what is missing, so it is also the
recovery path after preemption or failures.

    python3 -m epic_eval.orchestrator --config epic_eval/configs/qwen3.8-27b-thinking.json \\
        --fps 1 --max_frames 768                       # HEAD of this checkout, all three steps
    python -m epic_eval.orchestrator --config epic_eval/configs/gemini-3.8-flash-high.json \\
        --commit <pushed sha> --dryrun                 # print the scripts, submit nothing
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

from .common import (BANKS, RUN_CONFIG, config_hash, load_json, load_questions, media_done_marker,
                     media_hash, media_path, group_by_cutoff, resolve_bank)
from .run_eval import skip_qids

NEXRUN = Path("~/.claude/bin/nexrun").expanduser()


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def mirror(runs_root: Path, url: str, ref: str) -> Path:
    """One bare mirror per repo under <runs_root>/_mirrors, fetched unless it already has `ref`.

    A full SHA already in the mirror needs no network, so a pinned question bank works
    from nodes that cannot reach GitHub. Any other ref (a branch) is always refetched.
    """
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", url.rstrip("/").split("/")[-1].split(":")[-1])
    path = runs_root / "_mirrors" / name
    if not path.exists():
        print(f"mirroring {url} -> {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--mirror", url, str(path))
    elif not (re.fullmatch(r"[0-9a-f]{40}", ref) and has_commit(path, ref)):
        git("remote", "set-url", "origin", url, cwd=path)
        git("fetch", "--prune", "origin", cwd=path)
    return path


def has_commit(repo: Path, ref: str) -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                          cwd=repo, capture_output=True).returncode == 0


def local_repo() -> Path:
    """The checkout this orchestrator runs from; refuses uncommitted changes to tracked files."""
    top = Path(git("rev-parse", "--show-toplevel", cwd=Path(__file__).resolve().parent))
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=top)
    if dirty:
        sys.exit(f"FATAL: {top} has uncommitted changes; commit them first (jobs run a clean checkout):\n{dirty}")
    return top


def resolve(repo: Path, ref: str) -> str:
    try:
        return git("rev-parse", "--verify", f"{ref}^{{commit}}", cwd=repo)
    except subprocess.CalledProcessError:
        sys.exit(f"FATAL: {ref!r} is not a commit in {repo}")


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


def eval_chunks(run_dir: Path, videos: list[str], by_video: dict, cfg: dict, per_task: int,
                limit: int | None) -> list[tuple[str, str, list[str]]]:
    """(video, shard, qids) per eval task: each video's pending questions in chunks of per_task.

    Chunks follow group order, so a task opens as few video contexts as possible. The
    shard name carries a timestamp: a relaunch never reuses a file a running task writes.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    chunks = []
    for v in videos:
        skip = skip_qids(run_dir, v, cfg)
        pending = [q["qid"] for g in group_by_cutoff([q for q in by_video[v] if q["qid"] not in skip]).values()
                   for q in g][:limit]
        for i in range(0, len(pending), per_task):
            chunks.append((v, f"{stamp}-{i // per_task:03d}", pending[i:i + per_task]))
    return chunks


# ---------------------------------------------------------------- slurm

def python_fn(cfg: dict) -> list[str]:
    """Bash lines defining `py`, the interpreter every task command calls.

    With a `container` block, py runs python inside the Apptainer image. --cleanenv keeps the
    host's module/compiler variables out (Triton breaks on them), so PYTHONPATH goes in via --env.
    Without one, py is `paths.python`.
    """
    c = cfg.get("container")
    if not c:
        return [f'py() {{ {cfg["paths"]["python"]} "$@"; }}']
    flags = ["--nv", "--cleanenv", *(f"--bind {shlex.quote(b)}" for b in c.get("binds", []))]
    if c.get("pythonpath"):
        flags.append(f"--env PYTHONPATH={shlex.quote(c['pythonpath'])}")
    # Zaratan ships apptainer as a module; without this a job only finds it if the
    # submitting shell happened to have it loaded (sbatch exports that environment).
    return ["command -v apptainer >/dev/null || module load apptainer",
            f"SIF={shlex.quote(c['sif'])}",
            f'py() {{ apptainer exec {" ".join(flags)} "$SIF" python "$@"; }}']


def write_array(cfg: dict, step: str, commands: list[str], dependency: str | None) -> Path:
    """One array job whose task i runs commands[i-1], spelled out in a case statement.

    Commands can use $RUN (the run directory) and py (see python_fn).
    """
    run_dir, s = Path(cfg["resolved"]["run_dir"]), cfg["slurm"][step]
    slurm_dir, log_dir = run_dir / "slurm", run_dir / "slurm" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
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
        # null when ffmpeg comes from elsewhere, e.g. inside the container
        *([f"module load {cfg['paths']['ffmpeg_module']}"] if cfg["paths"].get("ffmpeg_module") else []),
        "export PYTHONUNBUFFERED=1",
        "",
        f"RUN={shlex.quote(str(run_dir))}",
        *python_fn(cfg),
        'cd "$RUN/code_repo"',
        "",
        'case "$SLURM_ARRAY_TASK_ID" in',
        *(f"  {i}) {cmd} ;;" for i, cmd in enumerate(commands, 1)),
        '  *) echo "no task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;',
        "esac",
        "",
    ]
    # Timestamped: a pending array from an earlier submission keeps its own script.
    script = slurm_dir / f"{step}_{time.strftime('%Y%m%d-%H%M%S')}.slurm"
    script.write_text("\n".join(lines))
    return script


def submit(cfg: dict, step: str, script: Path, n_tasks: int, note: str, dryrun: bool) -> str | None:
    """sbatch the script (via nexrun where it exists) and log it to slurm/launches.jsonl."""
    s = cfg["slurm"][step]
    res = (f"{s['partition']}, {s.get('gres') or 'no gpu'}, {s['cpus']} cpu, {s['mem']}, {s['time']}"
           + (f", max {s['max_parallel']} at once" if s.get("max_parallel") else ""))
    if dryrun:
        print(f"\n=== {step}: {n_tasks} task(s) ({res}) | {script}\n{script.read_text()}")
        return f"DRYRUN-{step}"
    run_dir = Path(cfg["resolved"]["run_dir"])
    if NEXRUN.exists():
        cmd = [str(NEXRUN), "launch", "--tag", f"{cfg['run']['tag']}-{step}", "--out", str(run_dir),
               "--note", note, "--", "sbatch", str(script)]
        pattern = r"submitted (\d+)"
    else:
        cmd, pattern = ["sbatch", "--parsable", str(script)], r"^(\d+)"
    out = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(pattern, out.stdout.strip())
    if out.returncode != 0 or not m:
        sys.exit(f"FATAL: submission of {step} failed:\n{out.stdout}{out.stderr}")
    job = m.group(1)
    print(f"{step}: {n_tasks} task(s) -> job {job} ({res})")
    with open(run_dir / "slurm" / "launches.jsonl", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, "job": job,
                            "n_tasks": n_tasks, "script": str(script), "commit": cfg["code"]["commit"],
                            "argv": sys.argv, "note": note}) + "\n")
    return job


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--commit", default="HEAD",
                   help="eval-code ref (default HEAD): resolved in this checkout, which must be clean, "
                        "or in a mirror of code.repo_url if the config has one")
    p.add_argument("--dryrun", action="store_true", help="check out and write scripts, but submit nothing")
    p.add_argument("--steps", default="prep,eval,score")
    p.add_argument("--videos", help="comma-separated subset (smoke tests)")
    p.add_argument("--limit", type=int, help="max questions per video per eval task (smoke tests)")
    p.add_argument("--code_repo_url", help="override config code.repo_url (e.g. a local path for testing)")
    p.add_argument("--questions_repo_url", help="override config questions.repo_url")
    p.add_argument("--env_name", help="override the auto-generated run directory name")
    p.add_argument("--note", default="", help="why this run exists (goes to slurm/launches.jsonl)")
    p.add_argument("--fps", type=float, help="override model.fps (required for qwen38_vl)")
    p.add_argument("--max_frames", type=int, help="override model.max_frames (required for qwen38_vl)")
    a = p.parse_args()

    cfg = load_json(a.config)
    try:
        cfg["questions"], bank = resolve_bank(cfg["questions"])
    except ValueError as e:
        sys.exit(f"FATAL: {e}")
    if a.fps is not None:
        cfg["model"]["fps"] = a.fps
    if a.max_frames is not None:
        cfg["model"]["max_frames"] = a.max_frames
    if cfg["model"]["backend"] == "qwen38_vl":
        # Video sampling is a sweep axis, so it is chosen per launch and named in the run dir.
        fps, max_frames = cfg["model"].get("fps"), cfg["model"].get("max_frames")
        if fps is None or max_frames is None:
            sys.exit("FATAL: qwen38_vl needs --fps and --max_frames, e.g. --fps 1 --max_frames 768")
        cfg["run"]["tag"] += f"-fps{fps:g}-f{max_frames}"
    runs_root = Path(cfg["run"]["runs_root"])
    code_url = a.code_repo_url or cfg.get("code", {}).get("repo_url")
    q_url = a.questions_repo_url or cfg["questions"]["repo_url"]

    # Resolve both commits to full SHAs *before* hashing, so a short and a long
    # spelling of the same commit land in the same run directory.
    local = not code_url
    code_src = local_repo() if local else mirror(runs_root, code_url, a.commit)
    code_url = code_url or str(code_src)
    cfg.setdefault("code", {})["commit"] = resolve(code_src, a.commit)
    if local and not git("branch", "-r", "--contains", cfg["code"]["commit"], cwd=code_src):
        print(f"note: {cfg['code']['commit'][:10]} is not on any remote branch; push it to keep the run reproducible")
    q_mirror = mirror(runs_root, q_url, cfg["questions"]["commit"])
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

    checkout(code_src, cfg["code"]["commit"], run_dir / "code_repo")
    checkout(q_mirror, cfg["questions"]["commit"], run_dir / "questions_repo")
    cfg["resolved"] = {
        "run_dir": str(run_dir),
        "code_dir": str(run_dir / "code_repo"),
        "questions_dir": str(run_dir / "questions_repo"),
        "media_dir": str(Path(cfg["paths"].get("media_root") or runs_root / "media") / media_hash(cfg))
                     if cfg["media"]["enabled"] else None,
        "code_repo_url": code_url,
        "questions_repo_url": q_url,
        **({"bank": bank} if bank else {}),
    }
    if not (run_dir / "code_repo" / "epic_eval").is_dir():
        sys.exit(f"FATAL: commit {cfg['code']['commit'][:10]} has no epic_eval/ package")
    prev = run_dir / RUN_CONFIG
    if prev.exists() and {k: v for k, v in load_json(prev).items() if k not in ("slurm", "runtime")} != \
            {k: v for k, v in cfg.items() if k not in ("slurm", "runtime")}:
        print(f"note: {prev} differs from this config outside 'slurm'/'runtime'; overwriting")
    prev.write_text(json.dumps(cfg, indent=2))

    questions = load_questions(cfg["resolved"]["questions_dir"], cfg["questions"]["glob"])
    want = load_json(BANKS)[bank].get("questions") if bank else None
    if want is not None and len(questions) != want:
        sys.exit(f"FATAL: bank {bank} should hold {want} questions, found {len(questions)} "
                 f"at {cfg['questions']['commit'][:10]} -- check banks.json")
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
    note = a.note or f"EPIC-Morph MCQ eval, {cfg['model']['name']}, video={cfg['media']['enabled']}"
    dep = None

    if "prep" in steps and cfg["media"]["enabled"]:
        todo = [v for v in videos if not media_ready(cfg, v, group_by_cutoff(by_video[v]).keys())]
        print(f"\nprep: {len(todo)}/{len(videos)} videos need media -> {cfg['resolved']['media_dir']}")
        if todo:
            cmds = [f'py -m epic_eval.prep_media --run_dir "$RUN" --video {v}' for v in todo]
            dep = submit(cfg, "prep", write_array(cfg, "prep", cmds, None), len(cmds), note, a.dryrun)

    if "eval" in steps:
        if cfg["model"]["backend"] == "gemini":
            from .backends.gemini import api_key
            if not api_key(cfg["model"]):
                msg = f"no Gemini key in ${cfg['model']['api_key_env']} or {cfg['model']['api_key_file']}"
                if not a.dryrun:
                    sys.exit(f"FATAL: {msg}")
                print(f"WARNING (dryrun): {msg}; a real submission would stop here")
        run_eval = 'py -m epic_eval.run_eval --run_dir "$RUN"'
        per_task = cfg["slurm"]["eval"].get("questions_per_task")
        if per_task:
            chunks = eval_chunks(run_dir, videos, by_video, cfg, int(per_task), a.limit)
            qid_dir = run_dir / "slurm" / "qids"
            qid_dir.mkdir(parents=True, exist_ok=True)
            cmds = []
            for v, shard, qids in chunks:
                qfile = qid_dir / f"{v}.{shard}.txt"
                qfile.write_text("\n".join(qids) + "\n")
                cmds.append(f'{run_eval} --video {v} --qids_file "$RUN/{qfile.relative_to(run_dir)}" --shard {shard}')
            print(f"\neval: {sum(len(q) for _, _, q in chunks)} unanswered questions in {len(cmds)} tasks "
                  f"of <= {per_task} over {len({v for v, _, _ in chunks})}/{len(videos)} videos")
        else:
            todo = [v for v in videos
                    if a.limit or {q["qid"] for q in by_video[v]} - skip_qids(run_dir, v, cfg)]
            print(f"\neval: {len(todo)}/{len(videos)} videos have unanswered questions")
            lim = f" --limit {a.limit}" if a.limit else ""
            cmds = [f"{run_eval} --video {v}{lim}" for v in todo]
        if cmds:
            dep = submit(cfg, "eval", write_array(cfg, "eval", cmds, dep),
                         len(cmds), note, a.dryrun) or dep

    if "score" in steps:
        cmds = ['py -m epic_eval.score --run_dir "$RUN"']
        submit(cfg, "score", write_array(cfg, "score", cmds, dep),
               1, note, a.dryrun)

    print(f"\n{'[DRYRUN] nothing submitted. ' if a.dryrun else ''}run dir: {run_dir}")


if __name__ == "__main__":
    main()
