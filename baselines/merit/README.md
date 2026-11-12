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
| `epic/collect_answers.py`, `epic/compare_answers.py` | merge a video's answer files and list unanswered IDs; compare two runs |
| `run_merit.sh` | one video, one arm, one stage (prep, build, qa, blind, all) |
| `env/` | the two environments, pinned: `wheels-*`, `requirements-*`, `install_requirements.sh` (shared by the image and `setup_env.sh`) |
| `docker/` | the image (`Dockerfile`), `build.sh`, `run.sh` (one job in a container), `image_tag.sh` |
| `build_order.txt` | the videos still without a MERIT memory, in the order to build them |

Our patches (in the submodule's `epic-morph` branch): Qwen3.8-27B as `qwen38-27b` (transformers 5.8, instruct
mode); loading from the local Hugging Face snapshot when cached; `MERIT_DEVICE_MAP` and, on several GPUs,
memory caps per GPU with a sequential layout (Qwen3.8-27B on 4x 24 GB); `MERIT_NO_FRAMES=1` (answer from
retrieved captions only, for memory probes). Settings are MERIT's EgoLife defaults except top-k 5, max 5
rounds, 512 image tokens, keys event + object + summary, 448 px.

**Two checkouts.** The code is this repo (`git clone --recurse-submodules`, or `git submodule update --init
baselines/merit/MERIT`). The data is an epic-morph checkout, named by `EPIC_MORPH_DIR` as for `epic_eval`:
banks are read from its `outputs/v3/banks/`, videos from its `data/HD-EPIC/videos/` (or `HDEPIC_VIDEOS`), and
results are written to its `outputs/v3/merit/<arm>/<video>/`, which is where they are committed.

## Two arms, two images

| arm | model (captions and answers) | GPU per job | environment |
|---|---|---|---|
| `m8` | Qwen/Qwen3-VL-8B-Instruct | one GPU with >= 40 GB (the 4B embedding model loads next to it) | transformers 4.57.3 |
| `m27` | Qwen/Qwen3.8-27B | one 80 GB GPU (H100); also runs on 2x 48 GB or 4x 24 GB | transformers 5.8.0 (5.2 loads it but generates garbage) |

**Off our clusters, run only in the Docker images** (rule in `AGENTS.md`): `ghcr.io/morph-graphs/merit:<arm>-<env
hash>`, built from `docker/Dockerfile` by `.github/workflows/merit-images.yml`. The image holds the environment
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
cache (`MERIT_HF_CACHE`, default `~/.cache/huggingface`) and, if set, `HDEPIC_VIDEOS` read-only; it runs as
your user and shows the container only the GPU(s) in `MERIT_GPU`. Download the models into that cache once
(`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3.8-27B`, `Qwen/Qwen3-Embedding-4B`). `docker/build.sh <arm>` builds
the same image locally. On our own clusters, a virtualenv from `env/setup_env.sh <arm> <dir>` (Python 3.11;
CUDA_HOME for m27's causal-conv1d) and `PY=<dir>/bin/python bash baselines/merit/run_merit.sh ...` run the same
stack without a container.

## Running

`build_order.txt` lists the 146 videos with no committed memory yet: the 17 that already have object stories,
then the rest longest first. `build` makes the clips and frames first if needed. `qa` answers every question
of the bank for the video; a question MERIT fails on (it writes "Error", e.g. after a CUDA out-of-memory) is
retried by ID, up to three passes. Every stage skips finished work, so rerun the same command after a
failure. One job per GPU (or GPU set); stages of different videos can run side by side. The clips and frames
(~300 MB a video) and MERIT's run folders stay in `baselines/merit/work/` (not committed).

## What comes back (in the epic-morph checkout)

Per video, in `outputs/v3/merit/<arm>/<video>/` (~0.3-0.8 MB a video), committed to epic-morph:
- `<video>_30sec.json`, `<video>_30sec_4_key.json`: the memory (captions, keys);
- `answers_<bank>.json.gz`: one row per question, with MERIT's retrieval rounds (the M1 evaluation reads them);
- `blind_<bank>.json`, when the blind stage ran;
- `runs.jsonl`: one line per finished stage (this repo's commit, the MERIT and epic-morph commits, model,
  torch and transformers versions, GPUs, seconds).

The batch10 runs are committed in epic-morph in this layout (10 videos x 2 arms).

## Checking a new machine first

Rebuild one batch10 video into a scratch folder and compare with the committed run:
```bash
MERIT_GPU=0 MERIT_OUT=/repo/baselines/merit/work/check \
    bash baselines/merit/docker/run.sh all P07-20240531-201703 m27 batch10_agree2
python3 baselines/merit/epic/compare_answers.py \
    $EPIC_MORPH_DIR/outputs/v3/merit/m27/P07-20240531-201703/answers_batch10_agree2.json.gz \
    baselines/merit/work/check/m27/P07-20240531-201703/answers_batch10_agree2.json.gz
```
The committed P07 runs score 24/64 (m8) and 25/64 (m27). Generation is greedy but not bit-identical across
GPUs and library builds, so expect a few questions to flip; the accuracies should be within a few questions
of each other and the McNemar p well above 0.05. A large gap means a setup problem (most often the wrong
transformers version for m27).
