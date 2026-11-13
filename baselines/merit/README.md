# baselines/merit: MERIT on EPIC-Morph

[MERIT](https://github.com/choi-yeeun/MERIT) (Choi et al., arXiv 2608.07663; Apache-2.0) builds a memory of a
video once (a caption per 30 s clip, plus retrieval keys) and answers questions by retrieving from it over
several rounds. It is the first memory baseline of the EPIC-Morph benchmark paper. The batch10 numbers
(598 questions, epic-morph bank `batch10_agree2`) were made with this code: MERIT 8B 36.3%, MERIT 27B 49.0%.

## Layout

| path | what |
|---|---|
| `MERIT/` | submodule: [morph-graphs/MERIT](https://github.com/morph-graphs/MERIT), branch `epic-morph` = upstream `6edea6e` + our patches (one commit) |
| `epic/prepare_epic.py` | one HD-EPIC video -> 30 s clips (448 px), 1 fps frames, the clip list; a bank -> MERIT's question format |
| `epic/caption_clips_qwen.py` | MERIT's memory, step 1: one caption per clip with the arm's model |
| `epic/blind_qa.py` | MERIT's answer prompt with no context: what the model gets from question and options alone |
| `epic/collect_answers.py`, `epic/compare_answers.py` | merge a video's answer files (reusing those of unchanged questions) and list unanswered IDs; compare two runs |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |
| `run_merit.sh` | one video, one arm, one stage (prep, build, qa, blind, upload, all) |
| `env/` | the two environments, pinned: `wheels-*`, `requirements-*`, `install_requirements.sh` (shared by the image and `setup_env.sh`) |
| `docker/` | the image (`Dockerfile`), `build.sh`, `run.sh` (one job in a container), `image_tag.sh` |
| `build_order.txt` | the videos still without a MERIT memory, in the order to build them |

Our patches (in the submodule's `epic-morph` branch): Qwen3.8-27B as `qwen38-27b` (transformers 5.8, instruct
mode); loading from the local Hugging Face snapshot when cached; `MERIT_DEVICE_MAP` and, on several GPUs,
memory caps per GPU with a sequential layout (Qwen3.8-27B on 4x 24 GB); `MERIT_NO_FRAMES=1` (answer from
retrieved captions only, for memory probes). Settings are MERIT's EgoLife defaults except top-k 5, max 5
rounds, 512 image tokens, keys event + object + summary, 448 px.

**Where things live.** The code is this repo (`git clone --recurse-submodules`, or `git submodule update
--init baselines/merit/MERIT`). The questions are an epic-morph checkout, named by `EPIC_MORPH_DIR` as for
`epic_eval`: banks from its `outputs/v3/banks/` (generated there with `v3/export_bank.py <bank>` when missing:
banks are not committed, and the same commit gives the same bank byte for byte), videos from its
`data/HD-EPIC/videos/` (or `HDEPIC_VIDEOS`).
The results are not in git: they are written to `baselines/merit/results/<arm>/<video>/` (ignored) and the
`upload` stage puts them in the private Hugging Face dataset
[morph-graphs/epic-morph-results](https://huggingface.co/datasets/morph-graphs/epic-morph-results) at
`merit/<arm>/<video>/` (a member's token with write access: `HF_TOKEN`, or `hf auth login`).

## Two arms, two images

| arm | model (captions and answers) | GPU per job | environment |
|---|---|---|---|
| `m8` | Qwen/Qwen3-VL-8B-Instruct | one GPU with >= 40 GB (the 4B embedding model loads next to it) | transformers 4.57.3 |
| `m27` | Qwen/Qwen3.8-27B | one 80 GB GPU (H100); also runs on 2x 48 GB or 4x 24 GB | transformers 5.8.0 (5.2 loads it but generates garbage) |

**Off our clusters, run only in the Docker images** (rule in `AGENTS.md`): `ghcr.io/morph-graphs/merit:<arm>-<env
hash>`, built from `docker/Dockerfile` by `.github/workflows/baseline-images.yml`. The image holds the environment
(`env/`: prebuilt CUDA wheels, pinned requirements, causal-conv1d compiled for m27); the code is the mounted
checkout of `main`. `docker/run.sh` picks the image whose tag matches the checkout's environment files
(`docker/image_tag.sh`):

```bash
export EPIC_MORPH_DIR=<epic-morph checkout>
docker pull ghcr.io/morph-graphs/merit:$(bash baselines/merit/docker/image_tag.sh m27)
MERIT_GPU=0 bash baselines/merit/docker/run.sh build P03-20240218-190556 m27            # one H100
MERIT_GPU=1 bash baselines/merit/docker/run.sh build P03-20240218-190556 m8             # one GPU >= 40 GB
MERIT_GPU=2 bash baselines/merit/docker/run.sh qa P07-20240531-201703 m8 batch10_agree2
```
`docker/run.sh` mounts this checkout at `/repo`, the epic-morph checkout at `/epic-morph`, the Hugging Face
cache (`MERIT_HF_CACHE`, default `~/.cache/huggingface`, which also holds a `hf auth login` token) and the
videos read-only; it passes `HF_TOKEN` when set, runs as your user and shows the container only the GPU(s)
in `MERIT_GPU`. After a video's stages finish: `MERIT_GPU=0 bash baselines/merit/docker/run.sh upload <video> <arm>`. Download the models into that cache once
(`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.8-27B`, `Qwen/Qwen3-Embedding-4B`). `docker/build.sh <arm>` builds
the same image locally. On our own clusters, a virtualenv from `env/setup_env.sh <arm> <dir>` (Python 3.11;
CUDA_HOME for m27's causal-conv1d) and `PY=<dir>/bin/python bash baselines/merit/run_merit.sh ...` run the same
stack without a container.

## Running

**Build a memory once, reuse it everywhere.** A memory depends only on the video and the model, so `build`
first takes it from the dataset when it is there (CPU only, seconds; `MERIT_FETCH=0` to skip that) and every
bank, probe set and variant answers from it. Variants of the same bank get their own answer files:
`MERIT_TAG=<name>` writes `answers_<bank>_<name>.json.gz`, and `MERIT_NO_FRAMES=1` (answer from captions only,
for memory probes) defaults the tag to `noframes`. `upload` adds files to the dataset folder and merges
`runs.jsonl`; only a rebuilt memory replaces the folder (answers from the old memory are then removed).

`build_order.txt` lists the 146 videos with no memory yet: the 17 that already have object stories,
then the rest longest first. `build` makes the clips and frames first if needed. `qa` answers every question
of the bank for the video; a question MERIT fails on (it writes "Error", e.g. after a CUDA out-of-memory) is
retried by ID, up to three passes. Every stage skips finished work, so rerun the same command after a
failure.

**Question identity.** A question's ID is its bank `key` (files without keys, such as M2 probes, keep the
1-based position), and each answer row also records `key`, `qhash` (a hash of the wording and options) and
`pos` (its position in the bank file). `qa` reuses the answers file already in `MERIT_OUT` (from an earlier
tranche, or `hf_results.py download`): an answer is kept while its question's wording, options and gold letter
are unchanged, so after a tranche adds or edits questions of a video only those are answered. Answer files from
before keys (positional IDs) are matched by the question they carry, and read the same.

One job per GPU (or GPU set); stages of different videos can run side by side. The clips and frames
(~300 MB a video) and MERIT's run folders stay in `baselines/merit/work/` (not committed).

## What comes back (in the Hugging Face dataset)

Per video, at `merit/<arm>/<video>/` in morph-graphs/epic-morph-results (~0.3-0.8 MB a video):
- `<video>_30sec.json`, `<video>_30sec_4_key.json`: the memory (captions, keys);
- `answers_<bank>.json.gz`: one row per question, with MERIT's retrieval rounds (the M1 evaluation reads them);
- `blind_<bank>.json`, when the blind stage ran (`blind_<bank>_<tag>.json` when `MERIT_CONTEXT` gave it a context file, named by `MERIT_TAG`);
- `runs.jsonl`: one line per finished stage (this repo's commit, the MERIT and epic-morph commits, model,
  torch and transformers versions, GPUs, seconds).

The batch10 runs (10 videos x 2 arms, from our Nexus runs) are there in this layout.

## Checking a new machine first

Rebuild one batch10 video into a scratch folder and compare with our run, fetched from the dataset:
```bash
MERIT_GPU=0 MERIT_OUT=/repo/baselines/merit/work/check \
    bash baselines/merit/docker/run.sh all P07-20240531-201703 m27 batch10_agree2
python3 baselines/common/hf_results.py download --arm m27 --video P07-20240531-201703 --dest baselines/merit/work/ref
python3 baselines/merit/epic/compare_answers.py \
    baselines/merit/work/ref/merit/m27/P07-20240531-201703/answers_batch10_agree2.json.gz \
    baselines/merit/work/check/m27/P07-20240531-201703/answers_batch10_agree2.json.gz
```
(`hf_results.py` needs `huggingface_hub`, in the images: on the host, `pip install huggingface_hub` or run it
in the container.)
The committed P07 runs score 24/64 (m8) and 25/64 (m27), made on RTX A5000s (m8) and A5000s / a GH200 (m27).
Decoding is greedy and deterministic on one GPU model, but captions differ across GPU models (measured on
P07, 2026-11-12: this environment and the original container gave 8/8 identical captions on the same A6000,
and both only 2/8 identical to the A5000 run), and the answers follow the captions. So on another GPU model
compare accuracy, not answers: our check on an L40S gave 22/64 vs 24/64 for m8 (41/64 same option, McNemar
p = 0.80). A large gap means a setup problem (most often the wrong transformers version for m27).
