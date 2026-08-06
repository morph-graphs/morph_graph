"""Generate a resumable Slurm array for ``morph_graph.video_qa``.

This deliberately writes submission artifacts but only calls ``sbatch`` when
``--submit`` is supplied.  Cluster/conda setup is caller-provided instead of
being hard-coded from a developer checkout.

Resource presets compose ``{site?}{gpu}`` (e.g. ``a5000``, ``vulcan-a6000``,
``h100``) from a small site table × GPU gres map — same conventions as the
FluxMem / CML scavenger scripts.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from .benchmarks import BENCHMARKS, get_benchmark

# Site → account / partition / qos. ``None`` = default scavenger.
_SITES: dict[str | None, dict[str, str]] = {
    None: {
        "account": "scavenger",
        "partition": "scavenger",
        "qos": "scavenger",
    },
    "vulcan": {
        "account": "vulcan-abhinav",
        "partition": "vulcan-scavenger",
        "qos": "vulcan-scavenger",
    },
    "cml": {
        "account": "cml-scavenger",
        "partition": "cml-scavenger",
        "qos": "cml-scavenger",
    },
}

# Short name → Slurm gres type. Bare ``h100`` defaults to the cml site.
_GPU_TYPES: dict[str, str] = {
    "a4000": "rtxa4000",
    "a5000": "rtxa5000",
    "a6000": "rtxa6000",
    "h100": "h100",
}

_DEFAULT_SITE_FOR_GPU: dict[str, str | None] = {
    "h100": "cml",
}


def _preset_names() -> list[str]:
    names: set[str] = set()
    for gpu in _GPU_TYPES:
        names.add(gpu)
        for site in _SITES:
            if site is not None:
                names.add(f"{site}-{gpu}")
    return sorted(names)


def _parse_preset(name: str) -> tuple[str | None, str]:
    if name in _GPU_TYPES:
        return _DEFAULT_SITE_FOR_GPU.get(name), name
    for site in _SITES:
        if site is None:
            continue
        prefix = f"{site}-"
        if name.startswith(prefix):
            gpu = name.removeprefix(prefix)
            if gpu in _GPU_TYPES:
                return site, gpu
    raise argparse.ArgumentTypeError(
        f"Unknown preset {name!r}; choose from: {', '.join(_preset_names())}"
    )


def resolve_preset(
    name: str,
    *,
    n_gpu: int = 1,
    cpus: int = 4,
    mem: str = "20G",
    time: str = "4:00:00",
) -> list[str]:
    """Expand a preset name into ``#SBATCH`` directive bodies (no ``#SBATCH``)."""
    site, gpu = _parse_preset(name)
    cfg = _SITES[site]
    gres_type = _GPU_TYPES[gpu]
    # CML H100 scripts historically use untyped ``gpu:N``; typed elsewhere.
    gres = f"gpu:{n_gpu}" if site == "cml" and gpu == "h100" else f"gpu:{gres_type}:{n_gpu}"
    return [
        f"--account={cfg['account']}",
        f"--partition={cfg['partition']}",
        f"--qos={cfg['qos']}",
        f"--gres={gres}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        "--nodes=1",
        "--requeue",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--question-path", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--num-jobs", type=int, default=10)
    parser.add_argument("--python", default="python3")
    parser.add_argument(
        "--preset",
        choices=_preset_names(),
        help=(
            "Resource preset composed as [site-]gpu "
            f"(sites: {', '.join(s for s in _SITES if s)}; "
            f"gpus: {', '.join(_GPU_TYPES)}). "
            "Bare h100 → cml; bare a* → scavenger."
        ),
    )
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per task (with --preset)")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per task (with --preset)")
    parser.add_argument("--mem", default="32G", help="Memory (with --preset)")
    parser.add_argument("--time", default="4:00:00", help="Walltime (with --preset)")
    parser.add_argument(
        "--preamble",
        action="append",
        default=[],
        help="Shell setup line; may be supplied more than once",
    )
    parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Extra/raw #SBATCH directive; appended after --preset",
    )
    parser.add_argument(
        "--extra-cli-arg",
        action="append",
        default=[],
        help="Additional argument passed unchanged to video_qa.cli",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    benchmark = get_benchmark(
        args.benchmark, question_path=args.question_path, video_dir=args.video_dir
    )
    samples = sorted(benchmark.load(), key=lambda sample: str(sample.video_path))
    
    # Filter out already processed samples globally
    completed_ids = set()
    if args.out_dir.exists():
        for chunk_path in args.out_dir.glob("answers_chunk_*.json"):
            try:
                data = json.loads(chunk_path.read_text())
                for row in data:
                    if "sample_id" in row:
                        completed_ids.add(str(row["sample_id"]))
            except Exception:
                pass
    if completed_ids:
        samples = [s for s in samples if str(s.sample_id) not in completed_ids]
        print(f"Skipping {len(completed_ids)} already processed samples. {len(samples)} remaining.")

    if not samples:
        print("All samples are already processed. Exiting.")
        return
    if args.num_jobs < 1:
        raise ValueError("--num-jobs must be positive")
    if args.gpus < 1:
        raise ValueError("--gpus must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    chunks = [
        samples[start : start + max(1, (len(samples) + args.num_jobs - 1) // args.num_jobs)]
        for start in range(
            0, len(samples), max(1, (len(samples) + args.num_jobs - 1) // args.num_jobs)
        )
    ]

    commands_path = args.log_dir / "commands.txt"
    commands = []
    for index, chunk in enumerate(chunks):
        ids_path = args.log_dir / f"sample_ids_{index}.json"
        ids_path.write_text(json.dumps([sample.sample_id for sample in chunk]))
        command = [
            args.python,
            "-m",
            "morph_graph.video_qa.cli",
            "--benchmark",
            args.benchmark,
            "--output-json",
            str((args.out_dir / f"answers_chunk_{index}.json").resolve()),
            "--cache-dir",
            str((args.out_dir / f"cache_chunk_{index}").resolve()),
            "--sample-ids-json",
            str(ids_path.resolve()),
        ]
        if args.question_path:
            command += ["--question-path", str(args.question_path.resolve())]
        if args.video_dir:
            command += ["--video-dir", str(args.video_dir.resolve())]
        for extra in args.extra_cli_arg:
            command += shlex.split(extra)
        commands.append(" ".join(shlex.quote(part) for part in command))
    commands_path.write_text("\n".join(commands) + "\n")

    if args.preset:
        directives = resolve_preset(
            args.preset,
            n_gpu=args.gpus,
            cpus=args.cpus,
            mem=args.mem,
            time=args.time,
        )
        directives += args.sbatch_arg
    elif args.sbatch_arg:
        directives = list(args.sbatch_arg)
    else:
        directives = ["--gres=gpu:1", "--cpus-per-task=4", "--mem=32G", "--time=4:00:00"]

    submit_path = args.log_dir / "submit.slurm"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --array=1-{len(chunks)}",
        f"#SBATCH --output={args.log_dir.resolve()}/%A_%a.log",
        f"#SBATCH --error={args.log_dir.resolve()}/%A_%a.err",
        *(f"#SBATCH {directive}" for directive in directives),
        "set -euo pipefail",
        'eval "$(/fs/cfar-projects/mem_open_vocab/miniconda3/bin/conda shell.bash hook)"',
        "conda activate morph_graphs_qwen3_cu126",
        "source /etc/profile.d/modules.sh",
        "module load cuda/12.6.3",
        "module load ffmpeg/7.1",
        *args.preamble,
        f"CMD=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" {shlex.quote(str(commands_path.resolve()))})",
        'srun bash -lc "$CMD"',
    ]
    submit_path.write_text("\n".join(lines) + "\n")
    print(f"Generated {len(chunks)} jobs for {len(samples)} questions: {submit_path}")
    if args.preset:
        print(f"Preset {args.preset}: {' '.join(directives)}")
    if args.submit:
        subprocess.run(["sbatch", str(submit_path)], check=True)


if __name__ == "__main__":
    main()
