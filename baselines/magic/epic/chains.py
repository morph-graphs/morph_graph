#!/usr/bin/env python3
"""MAGIC-Video's Narrative Memory Chain (topic + event chains) at one question cutoff, for EPIC-Morph.

MAGIC distils its chains by scanning a whole video offline, which would leak the future into a streaming
question. So the chains are built per question cutoff (the query time) from the memory that ends before it:
captions, 10 min summaries and semantic-extraction groups ending <= the cutoff (stricter than the paper, which
builds them once per video). They are made by the qa stage
(run_qa.py), cached per cutoff under <memory dir>/chains/c<seconds>/ and reused by every bank that asks at
the same second. Every chain prompt is memoised in chains/prompt_cache.json (keyed by the whole prompt, so a key
holds only video before the cutoff that made it): a later cutoff pays only for prompts whose input changed. With the
topic prompts in 50-caption chunks and the event step 1 once per day (as upstream's EgoLife chains; days: Days), the
cutoffs of a video sorted in time build incrementally: each one adds the topic chunks and the day that grew since the
last, plus the event merge (step 2) and its enrichment (step 3), which read the whole video so far and are redone
per cutoff.

They are the memory role's work (upstream: extract_{topic,event}_chains.py --model openai/gpt-oss-120b). With
the reported arms that is a paid API, so cutoffs another machine already built are first taken from the dataset
(MAGIC_MEMORY_REPO_PATH, the memory's folder there), and a cutoff with a failed call is not kept.
"""
import copy
import json
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import roles
from build_memory import TAG, jload, jsave, llm, secs

log = logging.getLogger("chains")

# MAGIC's EgoLife chain prompts (preprocess/egolife/extract_{topic,event}_chains.py) with the
# EgoLife cast (A1_JAKE and housemates) replaced by the EPIC setting: one camera wearer ("I")
# cooking in their own kitchen. Question families follow the EPIC-Morph bank's.
SYSTEM_PROMPT = "You are a precise video fact extractor. Output valid JSON only."

TOPIC_PROMPT = """In a first-person (egocentric) kitchen video, the entity "{entity}" appears in multiple episodes at different times.
The camera wearer is referred to as "I" in the captions; write facts about the camera wearer as "the camera wearer".

Below are the 30-second episode captions where "{entity}" is mentioned:

{episodes_text}

Extract facts about "{entity}" that help answer these types of questions:

1. **Where & When**: Where was "{entity}" taken from, and where was it put? When was it first/last used?
2. **State changes**: Was "{entity}" opened, closed, cut, washed, cooked, emptied, filled, mixed?
3. **Contents & Containers**: What was put into or taken out of "{entity}"? What container holds "{entity}" now?
4. **Steps & Order**: Which cooking step used "{entity}", and what happened before/after?
5. **Counts & Repetition**: How many times was "{entity}" used or moved?

Rules:
- Write each fact as a complete sentence naming "{entity}", the action, and the location or container.
- Include the WHERE (fridge, cupboard, drawer, hob, counter, sink, which bowl or pan) when relevant.
- Skip trivial mentions where "{entity}" appears in background without meaningful action.

Good facts:
- "The camera wearer took the {entity} out of the fridge and put it on the counter"
- "The camera wearer poured the {entity} into the large pan on the hob"

Bad facts (skip these):
- "The {entity} was on the counter" (static description, no action)

Output a JSON array (empty array [] if no meaningful facts):
[
  {{"time": "DAY1 00:05", "fact": "descriptive fact with action, object and location"}},
  {{"time": "DAY1 00:12", "fact": "another fact"}}
]"""

EVENT_STEP1 = """You are analyzing a first-person (egocentric) kitchen video. The camera wearer is referred to as "I" in the captions.

Below are 10-minute summary segments from {date}.

For each major activity in the video so far, extract:
- name: a short descriptive name (e.g., "Making the sauce", "Washing up")
- time_range: approximate start and end time (e.g., "00:03:00-00:09:00")
- summary: 2-3 sentences describing what happened: WHAT concrete actions were taken (take, put, open, pour, cut, stir, wash ...), with WHICH objects and containers, and WHAT the outcome was. Refer to the camera wearer as "the camera wearer".
- key_entities: important objects and places involved

Return a JSON list:
[
  {{
    "name": "activity name",
    "time_range": "HH:MM:SS-HH:MM:SS",
    "summary": "detailed description with specific actions, objects and outcomes",
    "key_entities": ["object1", "place1"]
  }}
]

Focus on significant, goal-directed activities (preparing an ingredient, cooking a dish, cleaning), not trivial actions. Aim for 3-12 activities.

Segments:
{segments}
"""

EVENT_STEP2 = """Below are activities from a first-person kitchen video.

Find activities that are related across different time periods. This includes:
- Different stages of the same dish or task (e.g., "chopped onions" -> "fried onions" -> "added onions to the stew")
- Repeated occurrences of the same activity (e.g., "washed a pan" at several times)
- Ongoing work done intermittently (e.g., stirring a pot between other tasks)

Only output events that have 2 or more related activities at different times. Ignore standalone activities. Be thorough and find as many as possible.

For each event, output:
- name: specific name
- steps: list of {{day, start_time, end_time, description}}  (day is {day_rule}; times are HH:MM:SS)
  * description must state concrete actions, objects and outcomes, not vague summaries
  * description should explain how this step relates to the previous step (continuation, completion, etc.)
- key_entities: important objects and places only

Return JSON list.

Activities:
{daily_activities}
"""

EVENT_STEP3 = """You are analyzing a first-person (egocentric) kitchen video. The camera wearer is referred to as "I" in the captions.

This is one step of the event chain "{chain_name}".
Time: {day} {start_time} - {end_time}

Below are the 30-second episode captions from this time period. Extract ONLY information relevant to "{chain_name}". Ignore unrelated content.

Captions:
{captions_text}

Write 2-5 sentences describing what happened in this step, focusing on:
- WHAT concrete actions were taken, with WHICH objects and containers, and what the outcome was
- WHERE things were taken from and put
Refer to the camera wearer as "the camera wearer".

Output a single JSON string (just the description text):
{{"description": "detailed description here"}}"""


def parse_json(raw):
    """extract_*_chains.call_llm's parsing: strip fences / think tags, else the first balanced {...} or [...]."""
    if not raw:
        return None
    clean = re.sub(r"```json\s*|```\s*", "", raw)
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        for sc, ec in [("[", "]"), ("{", "}")]:
            st = raw.find(sc)
            if st < 0:
                continue
            depth = 0
            for i in range(st, len(raw)):
                depth += (raw[i] == sc) - (raw[i] == ec)
                if depth == 0:
                    try:
                        return json.loads(raw[st:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


LENGTH_RETRY_TOKENS = 16000   # the budget for a prompt whose reply came back empty at its cap (below)


def call(model, prompt, system=None, max_tokens=4096, tries=3, cache=None):
    """One chain prompt at temperature 0; a parsed reply is kept in cache (the prompt cache) and reused. A reply that
    stopped at its cap with no parsable JSON (empty, or cut off) because the model spent the cap on reasoning (roles.hit_length: GPT-OSS-120B on a few topic
    prompts at 4,096, the same on every retry, 2026-11-17) is asked again with LENGTH_RETRY_TOKENS, as STEP1_TOKENS
    and MERGE_TOKENS already are for every call; a prompt answered within its cap is asked and cached as before."""
    key = json.dumps([system, prompt, max_tokens])
    if cache is not None and key in cache:
        return copy.deepcopy(cache[key])   # a caller that edits its reply never edits the cache
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    for _ in range(tries):
        try:
            r = parse_json(model.generate(msgs, max_tokens=max_tokens, temperature=0.0))
            if r is not None:
                if cache is not None:
                    cache[key] = r
                return r
        except Exception as e:
            log.warning(f"chain LLM error: {e}")
        if roles.hit_length() and max_tokens < LENGTH_RETRY_TOKENS:
            log.warning(f"chain reply stopped at max_tokens={max_tokens} (finish_reason length): again at {LENGTH_RETRY_TOKENS}")
            max_tokens = LENGTH_RETRY_TOKENS
    return None


PERSONS = {"i", "me", "my", "we", "us", "everyone", "narrator", "person", "people", "true", "false", "yes", "no",
           "he", "she", "they", "it", "someone", "something", "anything", "nothing", "things", "stuff",
           "camera wearer", "the camera wearer", "user", "the user"}


MERGE_TOKENS = 16000 if roles.ROUTED else 8000   # upstream's 16000 for its API model (reasoning tokens count too)
STEP1_TOKENS = 16000   # upstream's 4,000 truncated the reply on a long day; parse_json's repair stays the last resort


class Days:
    """Time labels for the chain prompts. An HD-EPIC video is one recording: DAY1 and the video's own clock, as before.
    A stitched video (<kitchen>-stitched) is a kitchen's recordings in order, each after a 4 s date card; its session
    table (the dataset's stitched/sessions/<kitchen>_sessions.json: each session's start date and time, duration and
    second in the stream) gives every stream second its real day (DAY<n>, n = the date's rank among the kitchen's dates)
    and clock time, as EgoLife's chains are labelled. The stored chains stay in stream time ("DAY1" + the stream's
    HH:MM:SS), which is what the qa stage matches them against."""

    def __init__(self, table=None):
        self.ses = []
        if table:
            from datetime import datetime
            dates = sorted({x["start"][:10] for x in table["sessions"]})
            for x in table["sessions"]:
                t0 = datetime.strptime(x["start"], "%Y-%m-%d %H:%M:%S")
                self.ses.append({"day": dates.index(x["start"][:10]) + 1, "at": x["video_at_s"], "card": x["card_at_s"],
                                 "dur": x["duration_s"], "clock": t0.hour * 3600 + t0.minute * 60 + t0.second})

    def day_clock(self, sec):
        """(day, clock seconds) of a stream second; a date card shows its session's start."""
        if not self.ses:
            return 1, int(sec)
        x = [x for x in self.ses if x["card"] <= sec][-1:] or self.ses[:1]
        x = x[0]
        return x["day"], x["clock"] + int(max(0, min(sec - x["at"], x["dur"])))

    def day(self, sec):
        return self.day_clock(sec)[0]

    def label(self, sec, hm=False):
        """'DAY<n> HH:MM' (hm) or 'HH:MM:SS' for an HD-EPIC video (the earlier labels); 'DAY<n> HH:MM[:SS]' when stitched."""
        d, c = self.day_clock(sec)
        t = f"{c // 3600:02d}:{c % 3600 // 60:02d}" + ("" if hm else f":{c % 60:02d}")
        return f"DAY{d} {t}" if hm or self.ses else t

    def stream(self, day, clock):
        """The stream second of a day and clock time a reply names (None: not a time). Between two of a day's sessions,
        or after its last, the end of the session before; before its first, that session's start. (No session of the
        nine kitchens crosses midnight, 2026-11-16.)"""
        if clock is None:
            return None
        if not self.ses:
            return clock
        on = [x for x in self.ses if x["day"] == day] if day else []
        if not on:
            return None
        prev = [x for x in on if x["clock"] // 60 * 60 <= clock]   # HH:MM (topic labels) names the session's first minute
        if not prev:
            return on[0]["at"]
        x = prev[-1]
        return x["at"] + max(0, min(clock - x["clock"], x["dur"]))


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def parse_time(t):
    """(day or None, clock seconds or None) of 'DAY3 18:08[:47]', '18:08:47' or '18:08'."""
    d = re.search(r"DAY\s*(\d+)", str(t))
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(t))
    return (int(d.group(1)) if d else None,
            int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0) if m else None)


def chains_at(cut, caps, m10, sem_triples, model, cache, days=Days()):
    """Topic + event chains from the video before `cut` seconds only."""
    def restamp(text, d):   # clock times a stitched video's reply writes in its text -> stream time, as stored
        if not days.ses or not isinstance(text, str):
            return text

        def one(m):   # "DAY<n> HH:MM[:SS]" -> "DAY1 <stream time>"; a bare time is on the reply's own day d
            t = days.stream(int(m.group(1)) if m.group(1) else d, int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4) or 0))
            return m.group(0) if t is None else ("DAY1 " if m.group(1) else "") + hms(min(t, cut))
        return re.sub(r"(?:\bDAY\s*(\d+)\s+)?\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", one, text)

    caps = [c for c in caps if secs(c["end_time"]) <= cut]
    m10 = [c for c in m10 if secs(c["end_time"]) <= cut]
    # topic chains: entities in >= 2 semantic-extraction groups that ended before the cutoff
    ent_ts = defaultdict(set)
    for ts, tl in sem_triples.items():
        if secs(ts[1:]) > cut:
            continue
        for t in tl:
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                for e in (t[0].lower().strip(), t[2].lower().strip()):
                    if e not in PERSONS and len(e) >= 2:
                        ent_ts[e].add(ts)
    cands = sorted(e for e, s in ent_ts.items() if len(s) >= 2)

    def topic(ent):   # chunks of 50 captions in time order: an earlier cutoff's full chunks are cache hits
        lines = [f"[{days.label(secs(c['start_time']), hm=True)}] {c['text']}" for c in caps if ent in c["text"].lower()]
        facts = []
        for i in range(0, len(lines), 50):
            r = call(model, TOPIC_PROMPT.format(entity=ent, episodes_text="\n\n".join(lines[i:i + 50])), system=SYSTEM_PROMPT,
                     cache=cache)
            if isinstance(r, list):
                facts += [f for f in r if isinstance(f, dict) and f.get("fact")]
        if days.ses:   # stored in stream time, as the qa stage reads it; never after the cutoff
            for f in facts:
                d, c = parse_time(f.get("time", ""))
                t = days.stream(d, c)
                f["time"] = f"DAY1 {hms(min(t, cut))[:5]}" if t is not None else ""
                f["fact"] = restamp(f["fact"], d)
        return ent, {"entity": ent, "has_lifecycle": bool(facts), "facts": facts}
    with ThreadPoolExecutor(32) as ex:
        topics = dict(ex.map(topic, cands))

    # event chains as upstream's EgoLife ones: step 1 once per day over its 10 min summaries (a summary belongs to the
    # day it starts on; the cutoff's day only up to the cutoff), step 2 links the days' activities (+ round 2), step 3
    # enriches each step from the captions
    enriched = []
    if m10:
        by_day = defaultdict(list)
        for c in m10:
            by_day[days.day(secs(c["start_time"]))].append(c)
        daily = ""
        for d in sorted(by_day):   # a finished day's prompt is the same at every later cutoff: cached
            seg = "\n\n".join(f"[{days.label(secs(c['start_time']))}] {c['text']}" for c in by_day[d])
            acts = [x for x in call(model, EVENT_STEP1.format(date=f"DAY{d}", segments=seg), max_tokens=STEP1_TOKENS,
                                    cache=cache) or [] if isinstance(x, dict)]
            daily += f"\n=== DAY{d} ===\n" + "".join(
                f"- [{x.get('time_range', '')}] {x.get('name', '')}: {x.get('summary', '')}\n"
                + (f"  Entities: {', '.join(map(str, x['key_entities']))}\n" if x.get("key_entities") else "") for x in acts)
        rule = '"DAY1"' if len(by_day) == 1 else 'the activity\'s day, "DAY1" to "DAY{}"'.format(max(by_day))
        chains = call(model, EVENT_STEP2.format(daily_activities=daily, day_rule=rule), max_tokens=MERGE_TOKENS, cache=cache) or []
        chains = [c for c in chains if isinstance(c, dict) and isinstance(c.get("steps"), list)]
        if chains:
            found = "\n".join(f"- {c.get('name', '')}" for c in chains)
            r2 = call(model, "Below are activities from a first-person kitchen video, and a list of cross-time event chains already found.\n"
                      "Your task: find ADDITIONAL cross-time event chains that were MISSED in the first round. Do not repeat any chain already found.\n\n"
                      f"Already found chains (do NOT repeat these):\n{found}\n\nActivities:\n{daily}\n\n"
                      "For each NEW event chain, output:\n- name: specific name\n- steps: list of {day, start_time, end_time, description}\n"
                      "- key_entities: important objects and places only\n\nReturn JSON list. If no additional chains found, return [].",
                      max_tokens=MERGE_TOKENS, cache=cache)
            names = {c.get("name", "").lower() for c in chains}
            chains += [c for c in (r2 or []) if isinstance(c, dict) and isinstance(c.get("steps"), list)
                       and c.get("name", "").lower() not in names]

        def enrich(ec):
            ec = copy.deepcopy(ec)
            for st in ec["steps"]:
                if not isinstance(st, dict):
                    continue
                d = parse_time(st.get("day", ""))[0]
                s0 = days.stream(d, parse_time(st.get("start_time", ""))[1])
                s1 = days.stream(d, parse_time(st.get("end_time", ""))[1])
                st["day"] = "DAY1"   # stored in stream time (the qa stage's clock)
                if "description" in st:
                    st["description"] = restamp(st["description"], d)
                if s0 is None or s1 is None:
                    if days.ses:   # a day and clock that name no stream time: not a time the qa stage can read
                        st["start_time"] = st["end_time"] = ""
                    continue
                if s0 > cut:   # a step the model placed after the cutoff cannot be grounded; drop it
                    st["start_time"] = st["end_time"] = ""
                    continue
                shown = (days.label(s0).split()[0] if days.ses else "DAY1", st["start_time"], st["end_time"])
                if s1 > cut:   # nor can its part after the cutoff: the step ends at the cutoff at the latest
                    s1 = cut
                    shown = shown[:2] + (days.label(cut).split()[-1],)
                    st["end_time"] = hms(cut)
                if days.ses:
                    st["start_time"], st["end_time"] = hms(s0), hms(s1)
                txt = [f"[{days.label(secs(c['start_time']))}] {c['text']}" for c in caps if s0 <= secs(c["start_time"]) <= s1]
                if not txt:
                    continue
                r = call(model, EVENT_STEP3.format(chain_name=ec.get("name", ""), day=shown[0], start_time=shown[1],
                                                   end_time=shown[2], captions_text="\n\n".join(txt)), max_tokens=1024, cache=cache)
                if isinstance(r, dict) and r.get("description"):
                    st["description"] = restamp(r["description"], d)
            return ec
        with ThreadPoolExecutor(16) as ex:
            enriched = list(ex.map(enrich, chains))
    return topics, enriched


def fetch_chains(cdir, memory_hash, todo):
    """Copy the cutoffs in todo that the dataset holds for this same memory into cdir (best effort)."""
    from huggingface_hub import snapshot_download
    path = os.environ["MAGIC_MEMORY_REPO_PATH"]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            snapshot_download(os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results"), repo_type="dataset",
                              allow_patterns=[f"{path}/chains/*"], local_dir=tmp)
        except Exception as e:
            log.warning(f"could not look for chains in the dataset ({type(e).__name__}: {str(e)[:160]})")
            return
        src = os.path.join(tmp, path, "chains")
        if not os.path.exists(os.path.join(src, "memory_hash")) or open(os.path.join(src, "memory_hash")).read().strip() != memory_hash:
            return
        got = [c for c in todo if os.path.exists(os.path.join(src, f"c{c:05d}", "step3_enriched_chains.json"))]
        for c in got:
            shutil.copytree(os.path.join(src, f"c{c:05d}"), os.path.join(cdir, f"c{c:05d}"), dirs_exist_ok=True)
        if os.path.exists(os.path.join(src, "prompt_cache.json")) and not os.path.exists(os.path.join(cdir, "prompt_cache.json")):
            shutil.copy(os.path.join(src, "prompt_cache.json"), cdir)
        log.info(f"chains for {len(got)} of {len(todo)} missing cutoffs taken from the dataset ({path})")


def sessions(cdir, video):
    """A stitched video's session table (kept in cdir, from the dataset's stitched/sessions/); None for other videos."""
    if not video.endswith("-stitched"):
        return None
    f = os.path.join(cdir, "sessions.json")
    if not os.path.exists(f):
        from huggingface_hub import hf_hub_download
        k = video[:-len("-stitched")]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                p = hf_hub_download(os.environ.get("MERIT_HF_REPO", "morph-graphs/epic-morph-results"),
                                    f"stitched/sessions/{k}_sessions.json", repo_type="dataset", local_dir=tmp)
            except Exception as e:
                raise SystemExit(f"{video}: no session table (stitched/sessions/{k}_sessions.json in the dataset; or put it "
                                 f"at {f}): {type(e).__name__}: {str(e)[:160]}")
            shutil.copy(p, f)
    return jload(f)


def ensure_chains(mem, video, cuts, memory_hash):
    """Chains for every cutoff (seconds) in cuts, built when missing; returns {cutoff: chain dir}. Chains made
    from another memory (a rebuild) are removed first."""
    cdir, stamp = os.path.join(mem, "chains"), os.path.join(mem, "chains", "memory_hash")
    if os.path.isdir(cdir) and (not os.path.exists(stamp) or open(stamp).read().strip() != memory_hash):
        log.info(f"chains in {cdir} were made from another memory: removing them")
        shutil.rmtree(cdir)
    os.makedirs(cdir, exist_ok=True)
    open(stamp, "w").write(memory_hash + "\n")
    missing = lambda: [c for c in sorted(set(cuts)) if not os.path.exists(os.path.join(cdir, f"c{c:05d}", "step3_enriched_chains.json"))]
    todo = missing()
    if todo and os.environ.get("MAGIC_MEMORY_REPO_PATH") and os.environ.get("MAGIC_FETCH", "1") != "0":
        fetch_chains(cdir, memory_hash, todo)
        todo = missing()
    if todo and roles.ROUTED and roles.missing(["memory"]):
        raise SystemExit(f"chains for {len(todo)} cutoffs of {video} are neither here nor in the dataset, and building "
                         f"them calls a paid API: {roles.missing(['memory'])[0]}. Run this bank's qa first where the key is.")
    if todo:
        caps, m10 = jload(os.path.join(mem, f"{video}_30sec.json")), jload(os.path.join(mem, f"{video}_10min.json"))
        days = Days(sessions(cdir, video))
        sem = jload(os.path.join(mem, f"semantic_extraction_results_{TAG}.json"))["semantic_triples"]
        model, cache_f = llm(), os.path.join(cdir, "prompt_cache.json")
        cache = jload(cache_f) if os.path.exists(cache_f) else {}
        for cut in todo:   # in time order: each cutoff reuses the earlier ones' prompts
            failed = roles.errors()
            topics, events = chains_at(cut, caps, m10, sem, model, cache, days)
            if roles.errors() > failed:   # a failed call would be kept as an empty chain
                raise SystemExit(f"{roles.errors() - failed} LLM calls failed building the chains at {cut}s"
                                 f"{': ' + roles.FATAL[0] if roles.FATAL else ''}; not kept, rerun to retry")
            d = os.path.join(cdir, f"c{cut:05d}")
            jsave(topics, os.path.join(d, "topic_chains.json"))
            jsave(events, os.path.join(d, "step3_enriched_chains.json"))   # written last: marks the cutoff done
            jsave(cache, cache_f)
            log.info(f"chains @ {cut}s: {sum(t['has_lifecycle'] for t in topics.values())}/{len(topics)} topic chains, "
                     f"{len(events)} event chains")
    return {c: os.path.join(cdir, f"c{c:05d}") for c in cuts}
