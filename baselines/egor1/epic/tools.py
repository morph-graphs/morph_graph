"""Ego-R1's three tools (rag, video_llm, vlm) on one HD-EPIC video, in process.

The tool code is upstream's, imported from the checkout (with ../epic-morph.patch: the Gemini model id from the
environment, plain OpenAI keys): r1rag.ragagent.RagAgent.query (GPT-4o picks the most relevant log entry),
egor1_videollm_gemini.video_llm_sync (stitches the 30 s clips of a range, 1 fps 448 px, and asks Gemini 2.5 Pro,
in place of the paper's retired 1.5 Pro; arm `llava`: egor1_videollm_llava, LLaVA-Video-7B behind upstream's
llava_video.py server) and egor1_vlm.vlm (one frame, GPT-4o). Upstream runs each behind a FastAPI server; here
they are called directly, and the dispatch and its error strings follow Ego-R1-Agent/utils/process.py tool_call.

What this port adds is the streaming cutoff: a question asked at t_ask must not see the video after it.
- rag searches a view of the memory cut at the question's time (cutoff_logs): the windows that ended by then,
  and the unfinished 10 min / hour / day windows summarized again from the captions before it (GPT-4.1, cached
  per cutoff, shared by every question asked at that second). Upstream's day and hour searches would otherwise
  read summaries that run past the question.
- video_llm ranges are clipped to the cutoff; a range or a vlm frame after it, or a stamp that is not
  DAYX_HHMMSSFF, gets an error string instead of a call (check()).
"""
import contextlib
import importlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.environ.get("EGOR1_SRC") or os.path.join(HERE, "..", "Ego-R1")
sys.path.insert(0, HERE)
import apis                                              # noqa: E402
from build_memory import LEVELS, build_levels, jload, jsave, secs   # noqa: E402

LATE = "error: invalid search. The time should not be later than the query time"


def upstream(folder, name):
    """Import <upstream>/<folder>/<name>.py. Every tool folder (and Ego-R1-Agent) has its own `utils`, so the
    cached one is dropped before and after."""
    path = os.path.join(UP, folder)
    for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[k]
    sys.path.insert(0, path)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(path)
        for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
            del sys.modules[k]


def stamp(sec):
    sec = int(sec)
    return f"DAY1_{sec // 3600:02d}{sec % 3600 // 60:02d}{sec % 60:02d}00"


def stamp_secs(ts):
    """Seconds of a DAYX_HHMMSSFF stamp; None when it is not one."""
    m = re.fullmatch(r"DAY(\d)_(\d{6})(\d{2})?", str(ts).strip())
    return (int(m.group(1)) - 1) * 86400 + secs(m.group(2) + "00") if m else None


def cutoff_logs(memory, video, cutoff, video_end):
    """Paths {"1day", "1hour", "10min"} of the logs a question asked at `cutoff` (seconds) may search."""
    full = {n: os.path.join(memory, f"{video}_{n}.json") for n, _ in LEVELS}
    if cutoff >= video_end:
        return full
    d = os.path.join(memory, "cutoffs", f"c{cutoff:06d}")
    view = {n: os.path.join(d, f"{n}.json") for n, _ in LEVELS}
    if not all(os.path.exists(p) for p in view.values()):
        apis.need("summary")
        caps = [e for e in jload(os.path.join(memory, f"{video}_30sec.json")) if secs(e["end_time"]) <= cutoff]
        os.makedirs(d, exist_ok=True)
        for n, log in build_levels(caps, reuse={n: jload(p) for n, p in full.items()}).items():
            jsave(log, view[n])
    return view


BAD_STAMP = "error: invalid time format. Use DAYX_HHMMSSFF"
CLIP = 30   # seconds per clip file (prepare_clips.py)


def check(call, cutoff):
    """The streaming rule on a parsed tool call: (call, None), with a video_llm range clipped to the cutoff,
    or (call, error string) when it asks for video after the question or its stamps are not DAYX_HHMMSSFF
    (upstream parses stamps more loosely than this check could vouch for). Calls no API.
    Upstream cuts a range at its end second, except that an end on a clip boundary takes that whole next clip
    (config_videos, trim_e == 0), and vlm reads one of the first 20 frames of the clip holding the stamp."""
    if call is None or not isinstance(call.get("arguments"), dict):
        return call, None
    name, args = str(call["name"]).lower(), call["arguments"]
    if "video_llm" in name and "range" in args:
        se = [stamp_secs(x) for x in str(args["range"]).split("-")]
        if len(se) != 2 or None in se:
            return call, BAD_STAMP
        end = min(se[1], cutoff)
        if end % CLIP == 0 and end + CLIP > cutoff:
            end -= 1
        if se[0] >= end:
            return call, LATE
        if end != se[1]:
            call = {**call, "arguments": {**args, "range": f"{stamp(se[0])}-{stamp(end)}"}}
    elif "vlm" in name and "timestamp" in args:
        t = stamp_secs(args["timestamp"])
        if t is None:
            return call, BAD_STAMP
        if t // CLIP * CLIP + 1 > cutoff:      # the frame comes from the first second of that clip
            return call, LATE
    return call, None


def is_api(e):
    """A failed API call (network, key, quota, Gemini file processing) or frame read (cv2), as opposed to a tool call upstream
    rejects (bad timestamp or range, refused content): those are the agent's to deal with."""
    return type(e).__module__.split(".")[0] in ("openai", "google", "httpx", "httpcore", "requests", "cv2") \
        or isinstance(e, (TimeoutError, ConnectionError, RuntimeError))


API_ERRORS = []   # failed API calls of the current question: it is then not counted as answered (retried later)
VIDEO_FAIL = "API error, fail to parse the response from video_llm."   # what process.tool_call returns on any failure


class Tools:
    def __init__(self, arm, clips, logs, cutoff, cache_dir):
        """arm: gemini25pro or llava (the video_llm); clips: <clip_root>/<video>; logs: cutoff_logs(); cutoff: seconds."""
        self.arm, self.clips, self.cutoff, self.cache = arm, clips, cutoff, cache_dir
        rag = upstream("api/rag", "r1rag.ragagent")
        if not hasattr(rag, "egor1_gpt4"):     # call_gpt4 returns "error" after 5 failed tries: note it
            rag.egor1_gpt4 = rag.call_gpt4

            def call_gpt4(*a, **k):
                r = rag.egor1_gpt4(*a, **k)
                if r == "error":
                    API_ERRORS.append("rag: call_gpt4 failed")
                return r
            rag.call_gpt4 = call_gpt4
        self.rag_agent = rag.RagAgent(day_log=logs["1day"], hour_log=logs["1hour"], min_log=logs["10min"])
        os.environ["EGOR1_GEMINI_MODEL"] = apis.MODELS["video_llm"]      # read by upstream's video_llm (our patch)
        self.video = upstream("api/visual_tools/egor1_videollm_" + ("llava" if arm == "llava" else "gemini"), "video_llm")
        self.image = upstream("api/visual_tools/egor1_vlm", "vlm")
        os.makedirs(os.path.join(cache_dir, "cache"), exist_ok=True)
        apis.track()

    def _quiet(self, fn, *a, **k):
        """Run upstream code in the cache folder (vlm writes ./cache/<stamp>.png) with its prints dropped (they
        include the tail of the API key)."""
        cwd = os.getcwd()
        os.chdir(self.cache)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return fn(*a, **k)
        finally:
            os.chdir(cwd)

    def call(self, call):
        """process.tool_call for one wearer, in process. Returns the text put between <information> tags."""
        call, err = check(call, self.cutoff)
        self.sent = None if err else call      # what goes on to upstream's tool, after the streaming check
        if call is None:
            return ""
        if err:
            return err
        name, args = str(call["name"]).lower(), call["arguments"]
        if not isinstance(args, dict):     # upstream's parser can hand back a set or a string
            return "Invalid query arguments."
        apis.ROLE = "rag" if "rag" in name else "video_llm" if "video_llm" in name else "vlm"
        if "rag" in name:
            if any(k not in args for k in ("level", "keywords", "start_time", "query_time")):
                return "Invalid query arguments."
            try:
                return str(self._quiet(self.rag_agent.query, level=args["level"], keywords=args["keywords"],
                                       query_time=args["query_time"], start_time=args["start_time"]))
            except Exception:      # upstream: the server answers 500 and the agent gets an empty result
                return ""
        if "video_llm" in name:
            if "question" not in args or "range" not in args:
                return "Invalid query arguments."
            try:
                if self.arm != "llava":
                    r = self._quiet(self.video.video_llm_sync, args["question"], args["range"], "A1",
                                    os.path.join(self.cache, "video"), os.environ["GEMINI_API_KEY"], self.clips)
                else:
                    r = self._quiet(self.video.video_llm, question=args["question"], range=args["range"], identity="A1",
                                    cache_dir=os.path.join(self.cache, "video"), data_dir=self.clips)
                return r["answer"]
            except Exception as e:
                if is_api(e):
                    API_ERRORS.append(f"video_llm: {type(e).__name__}: {e}")
                return VIDEO_FAIL
        if "vlm" in name:
            if "question" not in args or "timestamp" not in args:
                return "Invalid query arguments."
            try:
                return self._quiet(self.image.vlm, question=args["question"], timestamp=args["timestamp"], identity="A1",
                                   data_dir=self.clips)["answer"]
            except Exception as e:
                if is_api(e):
                    API_ERRORS.append(f"vlm: {type(e).__name__}: {e}")
                return "API error."
        return f"Invalid tool name: {call['name']}."
