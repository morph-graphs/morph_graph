# baselines/vgent: Vgent on EPIC-Morph

[Vgent](https://github.com/xiaoqian-shen/Vgent) (NeurIPS 2025, arXiv 2510.14032; the repository carries no
license file) is a training-free graph RAG pipeline around a video LLM. Offline it cuts the video (1 fps) into
64-frame chunks, has the video LLM list each chunk's entities, actions and scenes as JSON, and links chunks
that share an entity (text similarity, BAAI/bge-large-en-v1.5). For a question it extracts keywords, retrieves
up to 20 chunks by similarity to the graph's entities and chunk descriptions, checks them with yes/no
sub-questions ("structured reasoning"), and answers from the frames of the best 5 chunks.

It is run here **as released**: the same video LLM for every step, the same prompts and settings
(`chunk_size` 64, `n_retrieval` 20, `n_refine` 5, 1 fps, `total_pixels` 16384, `uniform_frame` 450). No
model is swapped and no paid API is involved: Vgent uses only open weights.

## Read first: what the released code does

All of this is upstream's behaviour, and the default here keeps it. It decides what a Vgent score means.

- **Under 1280 frames (21 min 20 s) there is no graph.** Upstream skips the graph when the video has fewer
  than `chunk_size * n_retrieval` frames and answers from up to 450 uniformly sampled frames: plain
  Qwen2.5-VL. A streaming question asked before 21:20 is answered that way. 116 of the 156 HD-EPIC videos
  are shorter than that.
- **`refine_nodes` never runs its sub-questions for ordinary questions.** It starts with `info = None` and
  stores the generated sub-questions with `info.update(...)`, which raises; the bare `except` retries five
  times and gives up, so the retrieved chunks go to the answer step unverified (top 5 by similarity).
  Only "order", "action counting" and "how many" questions, which build `info` by rule, get verified.
  We run it as released and do not fix it.
- Entity linking compares a new entity with the first graph key only (`max(range(len(entity_sim)))` runs
  over a 1-row matrix), so in practice chunks are linked by identical lower-cased names. Retrieval does not
  use the edges. Not changed.
- The extraction prompt asks for actions as `{"entity name": "action description"}` and the parser wants
  the keys `entity name` and `action description`, so actions are kept only when the backbone writes both.

## Layout

| path | what |
|---|---|
| `Vgent/` | submodule: upstream [xiaoqian-shen/Vgent](https://github.com/xiaoqian-shen/Vgent) at `c35c4b7` (2025-10-27), unpatched (`git submodule update --init baselines/vgent/Vgent`) |
| `epic/prepare_frames.py` | the video's 1 fps frames at Vgent's size, as one memory-mapped file |
| `epic/build_graph.py` | the offline graph of one video (upstream `vgent_graph.py`), each chunk's extraction cached; helpers for `run_qa.py` |
| `epic/run_qa.py` | answers one video's questions of a bank (upstream `vgent_rag.py`), each from the video up to its `t_ask`; `--blind` for the no-memory floor |
| `run_vgent.sh` | one video, one arm, one stage (prep, build, qa, blind, upload, all) |
| `env/` | the environment, pinned: `wheels.txt`, `requirements.txt`, `install_requirements.sh`, `setup_env.sh` |
| `../merit/epic/prepare_epic.py` | (MERIT's) the bank's questions as rows with their key and hash |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |

Upstream's two entry scripts loop over MLVU / VideoMME / LongVideoBench under `torchrun`; `epic/` calls the
same `Vgent` class (`utils/vgent.py`) for one video on one GPU instead. There is no Docker image yet.

## Arms

An arm is Vgent's backbone, named as in upstream's `MODEL_MAP`:

| arm | backbone (every step) | GPU |
|---|---|---|
| `qwenvl25_7b` | Qwen/Qwen2.5-VL-7B-Instruct (upstream's default; the paper's best configuration) | one GPU >= 40 GB |
| `qwenvl25_3b`, `qwenvl2_7b`, `qwenvl2_2b` | the paper's other Qwen-VL backbones; same wrapper, not run here yet | |

The paper's remaining backbones (InternVL2.5-2B, LLaVA-Video-7B, LongVU-7B) need their own packages, which
upstream's `requirements.txt` does not list; `run_vgent.sh` rejects them until an environment exists.

```bash
bash baselines/vgent/env/setup_env.sh <venv dir>   # Python 3.11 (PYTHON=...), CUDA 12.4 wheels
```

Models, from the Hugging Face cache: the arm's backbone and `BAAI/bge-large-en-v1.5`.

## Running

```bash
export PY=<venv>/bin/python EPIC_MORPH_DIR=<epic-morph checkout>
bash baselines/vgent/run_vgent.sh prep  P05-20240425-201444 qwenvl25_7b                  # CPU
bash baselines/vgent/run_vgent.sh build P05-20240425-201444 qwenvl25_7b                  # the graph, no questions needed
bash baselines/vgent/run_vgent.sh qa    P05-20240425-201444 qwenvl25_7b batch10_agree2
bash baselines/vgent/run_vgent.sh blind P05-20240425-201444 qwenvl25_7b batch10_agree2
bash baselines/vgent/run_vgent.sh upload P05-20240425-201444 qwenvl25_7b
```

Every stage skips finished work. The frames stay in `baselines/vgent/work/` (`VGENT_WORK`, about 1.1 MB per
second of video), results go to `baselines/vgent/results/<arm>/<video>/` (`VGENT_OUT`) and `upload` puts them
in the private dataset morph-graphs/epic-morph-results at `vgent/<arm>/<video>/`.

**The memory is question-independent.** `build` runs the one backbone call per 64 s chunk and caches it
(`<video>_nodes.jsonl`); the graph itself is assembled from those by upstream's `construct_graph`, which
adds chunks in time order. So the graph a question may see is exactly the graph upstream would build from the
video cut at its `t_ask`: `qa` runs `construct_graph` over the frames at or before `t_ask` with the cached
chunks, and only a last chunk cut short by `t_ask` costs a new backbone call (cached per cut in
`<video>_cutoff_nodes.jsonl`). Nothing after `t_ask` reaches a question: not a frame, not a chunk description.

## What differs from running upstream's scripts

- **Frames on disk.** Upstream holds the whole video in memory at full resolution (about 37 GB per hour of
  HD-EPIC). `prep` writes the same frames (upstream's indices and target size, bicubic) as uint8, so the
  floats are rounded, and the answer step downsizes these (616 px for HD-EPIC) instead of the 1408 px originals.
- **Frame size under a cut.** Frames keep the size computed for the whole video; upstream on a video cut at
  `t_ask` would size them from the cut's length (a few percent larger when it is not a multiple of 64 frames).
- **Text similarity is cached.** Each string is embedded once on the GPU instead of at every comparison on
  the CPU; same model, same pooling, values equal up to float rounding.
- **Questions** are shown in upstream's MLVU format (candidates `A. <option>`, prompt
  `Question: ...\nOptions:\n(A) ...`). Banks have 2 to 4 options; upstream answers the third option when it
  cannot read a letter from the backbone, which for a 2-option question is recorded as `?` (wrong).
- **No subtitles.** Upstream feeds LongVideoBench / VideoMME subtitles and, for MLVU, transcripts whose
  extraction is not in the repository. HD-EPIC has no transcript here, so `subtitles=None` (a code path
  upstream supports).

## Measured (one video, 2026-11-13)

`P05-20240425-201444` (27 min, 1630 frames), `qwenvl25_7b`, one RTX A6000, bank `batch10_agree2` (59 questions):
prep 4.6 min (4 CPU threads; frames 616x616, 1.86 GB); build 12 min (26 chunks, about 25 s each; graph of 26
nodes, 351 edges, 56 entity keys); qa 39 min: 5 questions asked before 21:20 with no graph (2 backbone calls,
about 10 s each) and 54 with the graph (7 calls and about 36 s for an ordinary question, 5 of the calls being
the failed sub-question attempts above; 22 calls and about 160 s for the 3 the rules verify); blind 41 s. Peak
GPU memory 24.3 GB. Of the 156 HD-EPIC videos (41.3 h), 116 are shorter than 1280 s, so as released Vgent
builds a graph for 40 of them.

## What comes back (in the Hugging Face dataset)

Per video, at `vgent/<arm>/<video>/`:
- the memory: `<video>_nodes.jsonl` (a header, then per chunk its entities, actions, scenes) and
  `<video>_graph.pkl` (upstream's format: `video_graph`, a networkx DiGraph, and `entity_graph`);
- `<video>_cutoff_nodes.jsonl`: the chunks cut short by a question's `t_ask`;
- `answers_<bank>.json.gz`: one row per question in bank order, keyed by the bank's `key` (plus `qhash`), with
  Vgent's letter (`response`), the backbone's text (`raw`), `evaluate`, `t_ask`, the frames it could see
  (`cutoff_frames`), whether a graph existed, the keyword step's output (`llm_info`), the retrieved and
  the used chunks, the sub-question checks, each backbone call's frames and seconds, `version` and the
  `memory_hash` the answer was made from;
- `blind_<bank>.json`: the same rows from the backbone with the question only;
- `runs.jsonl`: one line per finished stage (commits of this repo, upstream and epic-morph; backbone; torch
  and transformers versions; GPU; seconds).
