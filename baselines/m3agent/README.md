# baselines/m3agent: M3-Agent on EPIC-Morph

[M3-Agent](https://github.com/ByteDance-Seed/m3-agent) ("Seeing, Listening, Remembering, and Reasoning: A Multimodal
Agent with Long-Term Memory", Long et al., ICLR 2026, arXiv 2508.09736; Apache-2.0) watches a video in 30 s clips
and writes an entity-centred memory graph: faces (insightface) and voices (ERes2NetV2) become nodes, and a fine-tuned
Qwen2.5-Omni ([M3-Agent-Memorization](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization)) writes episodic
and semantic memories that refer to them. A question is answered by
[M3-Agent-Control](https://huggingface.co/ByteDance-Seed/M3-Agent-Control) (Qwen3-32B trained with RL) in up to
5 rounds of reasoning, each either an answer or a search that returns the memories of the top-2 clips.

## Layout

| path | what |
|---|---|
| `m3-agent/` | submodule: [morph-graphs/m3-agent](https://github.com/morph-graphs/m3-agent), branch `epic-morph` = upstream `0e3e419` + our patches (two commits: the port, then the API arm's plain-key calls) |
| `epic/m3_epic.py` | one HD-EPIC video -> 30 s clips with audio; a bank -> the control model's input (before_clip from t_ask); control output -> the answers file |
| `run_m3agent.sh` | one video, one arm, one stage (prep, build, qa, blind, upload, all) |
| `env/` | the two environments, pinned: `requirements-*`, `wheels-mem.txt`, `install_requirements.sh` (shared by `setup_env.sh` and a future image) |
| `VERSION` | the adapter version, in every answer row and `runs.jsonl` line; bumped by hand when the outputs' meaning changes |
| `../common/hf_results.py` | (shared by all baselines) upload a video's results to the Hugging Face dataset, or download them |

## Two environments

Upstream needs two transformers versions that cannot share an environment, so there are two virtualenvs
(Python 3.11, torch 2.6.0 with CUDA 12.4), made with `env/setup_env.sh <mem|ctl> <dir>`:

| env | runs | key pins |
|---|---|---|
| `mem` | prep, build | transformers at upstream's commit `f742a644` (Qwen2.5-Omni before its release), qwen-omni-utils 0.0.4, flash-attn 2.7.4.post1, insightface 0.7.3 + onnxruntime 1.22.1 (CPU), hdbscan 0.8.40, speakerlab (3D-Speaker `065629c`), faster-whisper 1.1.1 |
| `ctl` | qa, blind, upload; the embedding model | vllm 0.8.4, transformers 4.51.0, numpy 1.26.4, sentence-transformers 4.1.0 |

`setup_env.sh` needs git and a C compiler (insightface and hdbscan build from source); ffmpeg comes from
imageio-ffmpeg when the system has none. The scripts take the two pythons as `PY_MEM` and `PY_CTL`.

Models (`M3AGENT_MODELS`, one folder, or one variable each; see the header of `run_m3agent.sh`):
`ByteDance-Seed/M3-Agent-Memorization` (18 GB), `ByteDance-Seed/M3-Agent-Control` (66 GB), `Qwen/Qwen3-Embedding-4B`,
a CTranslate2 conversion of Whisper large-v3-turbo (e.g. `deepdml/faster-whisper-large-v3-turbo-ct2`),
`pretrained_eres2netv2.ckpt` from ModelScope `iic/speech_eres2netv2_sv_zh-cn_16k-common` (the upstream README's
link), and `insightface/` (insightface downloads `buffalo_l` there on first use).

## GPUs

| stage | needs | measured (P05-20240425-201444, 27 min) |
|---|---|---|
| build | one GPU with >= 40 GB: the memorization model (~18 GB in bf16) with a 30 s clip of video and audio (~10k tokens), the embedding model (8 GB, a helper process), Whisper and ERes2NetV2; faces run on the CPU (8 cores). A 24 GB card runs out of memory in the first clip's prefill even with the embedding model on the CPU (measured on an A5000) | see "Checked" below |
| qa, blind | the 32.8B control model on vLLM: 2x 48 GB with tensor parallel 2 (upstream's setting), or one 80 GB GPU (`M3AGENT_TP=1`), plus the embedding model on the first GPU | see "Checked" below |

## Running

```bash
export PY_MEM=<mem venv>/bin/python PY_CTL=<ctl venv>/bin/python EPIC_MORPH_DIR=<epic-morph checkout> M3AGENT_MODELS=<models>
CUDA_VISIBLE_DEVICES=0   bash baselines/m3agent/run_m3agent.sh build P05-20240425-201444 m3
CUDA_VISIBLE_DEVICES=0,1 bash baselines/m3agent/run_m3agent.sh qa    P05-20240425-201444 m3 batch10_agree2
CUDA_VISIBLE_DEVICES=0,1 bash baselines/m3agent/run_m3agent.sh blind P05-20240425-201444 m3 batch10_agree2
bash baselines/m3agent/run_m3agent.sh upload P05-20240425-201444 m3
# the API arm: GEMINI_API_KEY and OPENAI_API_KEY set, arm m3api (see "The API arm")
```

**Build once, answer many times.** `build` needs no questions, so every video's memory can be built before its
questions exist. It first takes the memory from the dataset when it is there (`M3AGENT_FETCH=0` to skip that), and
resumes after the last finished clip when interrupted (the graph is saved after every clip). `qa` truncates the
memory per question at its `t_ask`: a question asked at time t sees clips 0 .. floor(t/30) - 1, the clips
that had ended by then (upstream's `before_clip`, as in M3-Bench's annotations; MERIT's runs round t up first,
which can admit a clip up to 1 s early).
Questions that are not streaming see the whole video. `M3AGENT_TAG=<name>` writes a variant's answers to
`answers_<bank>_<name>.json.gz`. Every stage skips finished work: rerun the same command after a failure.

## What comes back (in the Hugging Face dataset, `m3agent/<arm>/<video>/`)

- `<video>_m3graph.pkl`: the memory graph (upstream's `VideoGraph`, pickled; load it with the submodule on the
  path), and `<video>_memories.json`: its text memories per clip, readable, with the character mappings;
- `answers_<bank>.json.gz`: one row per question, keyed by the bank's `key` (plus the positional `ID`): the
  options, the correct and the chosen letter, `correct`, the model's answer text, `before_clip`, each search
  round's query and the clips it returned with their start and end times (evidence retrieval), the full
  conversation, the memory's hash and the adapter version;
- `blind_<bank>.json`: the same rows from the control model with no memory;
- `runs.jsonl`: one line per finished stage (commits of this repo, the submodule and epic-morph, versions, GPUs,
  seconds).

The clips, the per-clip face and voice outputs and the control runs stay in `M3AGENT_WORK`.

## How this differs from the published method

In arm `m3`, local models replace upstream's closed APIs (`M3AGENT_BACKEND=local`); arm `m3api` calls the APIs
instead (see "The API arm" below):

1. **ASR / "diarization": faster-whisper large-v3-turbo instead of Gemini-1.5-Pro.** Upstream asks Gemini for the
   speech segments of each clip (start, end, text), split at speaker turns but without speaker labels; who speaks
   comes from ERes2NetV2 voice embeddings in both versions. Whisper (VAD filter on, segments with no-speech
   probability > 0.6 dropped) splits at pauses instead of speaker turns, so one segment can hold two speakers.
2. **Text embeddings: Qwen3-Embedding-4B instead of OpenAI text-embedding-3-large**, for the memories, the
   semantic-memory merging (similarity > 0.85) and the search queries (with Qwen3-Embedding's query instruction).
   The control model's search threshold stays at upstream's 0.5, though the two models' similarity scales differ.
3. **No answer judge.** Upstream grades open answers with GPT-4o; EPIC-Morph is multiple choice, so the options
   are listed after the question ("Answer with the letter of the correct option") and the chosen letter is read
   from the answer (a leading letter, else the option text it contains) and compared with the key.

Other changes, all in the submodule's first commit:

4. Clips are cut as upstream (clip k = [30k, 30k + 30) s) but re-encoded (H.264, AAC, long side 720 px) rather
   than copied with `-c copy`, so each cut is exact instead of snapping to the source's keyframes; HD-EPIC's
   1408 px frames are scaled to 720 px; a last sliver under 2 s is dropped.
5. Clips are processed in clip order (upstream's unsorted `glob` made node ids, and so the truncation by clip,
   depend on directory order), and `truncate_memory_by_clip` truncates correctly when the cut-off clip has no
   text memory (upstream then kept the whole memory, future included).
6. The memorization output is accepted under `video_description` as well as `video_descriptions` (the prompt asks
   for the first, upstream's parser reads the second), and its retries are capped at 3 instead of 20: decoding is
   near-greedy (temperature 1e-6), so a retry repeats the same output.
7. Control: every request seeded (sampling stays at upstream's 0.6 / 0.95 / 20), searches run in one process on a
   cached truncated graph instead of a process pool reloading the pickle, max_model_len 16384 and
   gpu_memory_utilization 0.8 (to leave room for the embedding model), answers resumable. Parsing: the action is
   read after `</think>` only (upstream also matched inside reasoning cut off at 1024 tokens) and without a
   greedy match across brackets. No search runs in the round that must answer (upstream ran one whose result
   nobody read).
   **Fallback answer round:** the control model often outputs another [Search] in the round that must answer
   (with no memory, the blind stage, nearly always: 54 of 59 questions on P05). Upstream counts such a question
   as unanswered (wrong). We give it one more round in which no search is possible and that asks for the
   option letter; `answered_in_loop` in each row keeps upstream's outcome, and the summary line prints both
   accuracies (`--no_fallback_answer` in `control.py` restores upstream's protocol).
8. flash-attn 2.7.4.post1 instead of 2.6.3 (no torch 2.6 build of 2.6.3); Python 3.11 instead of 3.9; the packages
   only used for training (trl, peft) or by the Omni talker are not installed.

Known approximations: a face or voice node keeps embeddings added by later clips, and later semantic memories can
reinforce the edges of earlier ones; truncation removes later nodes but not these updates. They can change which
face-voice equivalence wins (and so the character ids), and an edge weakened to zero by a later clip is missing
from a search restricted to a character; the text memories themselves and their embeddings are never touched.

## The API arm (`m3api`)

`run_m3agent.sh <stage> <video> m3api [bank]` runs the same two checkpoints with upstream's API calls in place of
the local ASR and embedding models (`M3AGENT_BACKEND=api`, set by the arm). Its memory, answers and work files are
kept apart from `m3`'s: `results/m3api/<video>/`, `work/m3api/` (the clips are shared) and `m3agent/m3api/<video>/`
in the dataset, so neither arm can pick up the other's memory or per-clip voice outputs.

| variable | needed by | what |
|---|---|---|
| `GEMINI_API_KEY` | `build` | a Gemini Developer API (AI Studio) key; sent only in the `x-goog-api-key` header |
| `OPENAI_API_KEY` | `build`, `qa` | an OpenAI key, for `text-embedding-3-large` (memories and search queries) |
| `M3AGENT_GEMINI_MODEL` | `build` (optional) | the speech-segment model, default `gemini-2.5-pro`; recorded in `runs.jsonl` |

`blind` needs neither key (no memory, no search). A missing key stops the stage before any work.

What the calls do (submodule, `mmagent/utils/chat_api.py`, `mmagent/voice_processing.py`):

- **Speech segments:** each clip's audio (16 kHz WAV, inline) goes to Gemini's REST `generateContent` with
  upstream's ASR/segmentation prompt, unchanged, and a response schema that forces a JSON list of
  `{start_time, end_time, asr}`; parsing is upstream's. Upstream sent the clip's video to `gemini-1.5-pro-002`,
  which is retired. Retries back off on 429, 5xx and network errors. Unlike upstream (20 attempts, then the build
  stops), a clip with no usable answer after 3 attempts counts as having no speech, with a warning in the log
  (decoding is near-greedy, so a retry mostly repeats the answer). Segments without text or whose times are not
  `MM:SS` are dropped.
- **Embeddings:** `openai.OpenAI()` with upstream's model name `text-embedding-3-large`, 64 texts per request. The
  search threshold stays at upstream's 0.5, the value upstream used with this model.
- Still on upstream's Azure config (`configs/api_config.json`), and not used by these scripts: the GPT-4o judge
  (`control.py --judge`) and the planning call in `retrieve.answer_with_retrieval`.

**Model availability (checked 2026-11-13):** our AI Studio key lists `gemini-2.5-pro`, but `generateContent`
returns 404 ("no longer available to new users", suggesting `gemini-3.1-pro-preview`). A key from an account that
used 2.5 Pro before may still reach it; otherwise set `M3AGENT_GEMINI_MODEL`.

Checked (2026-11-13):

- **Live, Gemini:** the real `process_voices` (memorization environment, Nexus) on the first 12 s of P05 clip 31,
  with `M3AGENT_GEMINI_MODEL=gemini-3.1-pro-preview` (see above), speaker embeddings stubbed (ERes2NetV2 needs a
  GPU and is unchanged): one call returned `[{"start_time": "00:04", "end_time": "00:06", "asr": "Nine minutes."}]`
  (Whisper in arm `m3`: 00:05-00:08 "Nine minutes"), kept as one 2 s segment in one voice node.
- **Mocked, not live:** the OpenAI embedding path. We have no OpenAI key. Checked against a stub client and the
  real `openai` 1.75.0 SDK on a mocked HTTP transport, in both environments: request URL, model and input, batches
  of 64, order kept, empty texts padded. The same mocks cover the missing-key errors, Gemini 429/503 retries, a
  400 raising, the key in the header only, blocked or `{}` answers (3 attempts, then no speech), malformed
  segments dropped, and `m3` still embedding locally.
- **Not run:** a full `m3api` build or qa on a video.

## Checked (2026-11-12/13, P05-20240425-201444, bank batch10_agree2: 59 questions, all streaming)

- Both environments build from `env/` (Python 3.11.11, CPU job, 12 and 15 min); `pip check` is clean for `mem` and
  reports only the expected transformers pin for `ctl`.
- `prep`: the 27.2 min video (1408 px, AAC stereo) gives 55 clips of exactly 30.0 s (video and audio), the last
  9.8 s; questions map to before_clip 8..53.
- `build` on one RTX A6000: 72 min (2.65 GPU-hours per video-hour), ~60 s a clip, a few clips up to 6 min
  (memorization retries). Memory: 55 clips, 1299 nodes, 13.5 MB pickle + 0.12 MB JSON; 0 face nodes (egocentric
  video: no face passes the quality filter) and 1 voice node (Whisper with VAD found speech twice in the video;
  without VAD it hallucinates "Thank you." on kitchen noise).
- `qa` on 4x RTX A5000 (TP 4, embedding model on the CPU): 14/59 right (23.7%), every question answered within
  the 5 rounds; 138 searches, 37 returned nothing (threshold 0.5); all 171 retrieved clips ended before t_ask.
  67 min, mostly the CPU embedding model running on one thread (fixed since: it now uses the job's cores).
- `blind` (same hardware): 12/59 (20.3%), 4 answered within the round, 55 through the fallback round.
- For context, MERIT on the same 59 questions: 8B 21/59, 27B 30/59. Chance over the options is ~25%; M3-Agent's
  14/59 and 12/59 are both within one standard deviation of it.
