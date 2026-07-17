"""Generate a resumable Slurm array for ``morph_graph.video_qa``.

This deliberately writes submission artifacts but only calls ``sbatch`` when
``--submit`` is supplied.  Cluster/conda setup is caller-provided instead of
being hard-coded from a developer checkout.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from .benchmarks import BENCHMARKS, get_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--question-path", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--num-jobs", type=int, default=10)
    parser.add_argument("--python", default="python")
    parser.add_argument("--preamble", action="append", default=[],
                        help="Shell setup line; may be supplied more than once")
    parser.add_argument("--sbatch-arg", action="append", default=[],
                        help="Raw #SBATCH directive, e.g. '--gres=gpu:1'")
    parser.add_argument("--extra-cli-arg", action="append", default=[],
                        help="Additional argument passed unchanged to video_qa.cli")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    benchmark = get_benchmark(
        args.benchmark, question_path=args.question_path, video_dir=args.video_dir
    )
    samples = sorted(benchmark.load(), key=lambda sample: str(sample.video_path))
    if not samples:
        raise ValueError("Benchmark produced no samples")
    if args.num_jobs < 1:
        raise ValueError("--num-jobs must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    chunks = [
        samples[start : start + max(1, (len(samples) + args.num_jobs - 1) // args.num_jobs)]
        for start in range(0, len(samples), max(1, (len(samples) + args.num_jobs - 1) // args.num_jobs))
    ]

    commands_path = args.log_dir / "commands.txt"
    commands = []
    for index, chunk in enumerate(chunks):
        ids_path = args.log_dir / f"sample_ids_{index}.json"
        ids_path.write_text(json.dumps([sample.sample_id for sample in chunk]))
        command = [
            args.python, "-m", "morph_graph.video_qa.cli",
            "--benchmark", args.benchmark,
            "--output-json", str((args.out_dir / f"answers_chunk_{index}.json").resolve()),
            "--cache-dir", str((args.out_dir / f"cache_chunk_{index}").resolve()),
            "--sample-ids-json", str(ids_path.resolve()),
        ]
        if args.question_path:
            command += ["--question-path", str(args.question_path.resolve())]
        if args.video_dir:
            command += ["--video-dir", str(args.video_dir.resolve())]
        for extra in args.extra_cli_arg:
            command += shlex.split(extra)
        commands.append(" ".join(shlex.quote(part) for part in command))
    commands_path.write_text("\n".join(commands) + "\n")

    submit_path = args.log_dir / "submit.slurm"
    directives = args.sbatch_arg or ["--gres=gpu:1", "--cpus-per-task=8", "--mem=32G", "--time=4:00:00"]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --array=1-{len(chunks)}",
        f"#SBATCH --output={args.log_dir.resolve()}/%A_%a.log",
        f"#SBATCH --error={args.log_dir.resolve()}/%A_%a.err",
        *(f"#SBATCH {directive}" for directive in directives),
        "set -euo pipefail",
        *args.preamble,
        f"CMD=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" {shlex.quote(str(commands_path.resolve()))})",
        'srun bash -lc "$CMD"',
    ]
    submit_path.write_text("\n".join(lines) + "\n")
    print(f"Generated {len(chunks)} jobs for {len(samples)} questions: {submit_path}")
    if args.submit:
        subprocess.run(["sbatch", str(submit_path)], check=True)


if __name__ == "__main__":
    main()
