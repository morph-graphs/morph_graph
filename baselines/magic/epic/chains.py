#!/usr/bin/env python3
"""MAGIC-Video's Narrative Memory Chain (topic + event chains) at one question cutoff, for EPIC-Morph.

MAGIC distils its chains by scanning a whole video offline, which would leak the future into a streaming
question. So the chains are built per question cutoff (ceil(t_ask)) from the memory that ends before it:
captions, 3 min summaries and semantic-extraction groups ending <= the cutoff. They are made by the qa stage
(run_qa.py), cached per cutoff under <memory dir>/chains/c<seconds>/ and reused by every bank that asks at
the same second. A topic prompt whose input did not change since an earlier cutoff is answered from
chains/topic_cache.json.

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

Below are 3-minute summary segments from {date}.

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
- steps: list of {{day, start_time, end_time, description}}  (day is "DAY1"; times are HH:MM:SS)
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


def call(model, prompt, system=None, max_tokens=4096, tries=3):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    for _ in range(tries):
        try:
            r = parse_json(model.generate(msgs, max_tokens=max_tokens, temperature=0.0))
            if r is not None:
                return r
        except Exception as e:
            log.warning(f"chain LLM error: {e}")
    return None


def fmt_hm(ts):
    s = str(ts).zfill(8)
    return f"DAY1 {s[0:2]}:{s[2:4]}"


def fmt_hms(ts):
    s = str(ts).zfill(8)
    return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"


PERSONS = {"i", "me", "my", "we", "us", "everyone", "narrator", "person", "people", "true", "false", "yes", "no",
           "he", "she", "they", "it", "someone", "something", "anything", "nothing", "things", "stuff",
           "camera wearer", "the camera wearer", "user", "the user"}


MERGE_TOKENS = 16000 if roles.ROUTED else 8000   # upstream's 16000 for its API model (reasoning tokens count too)


def chains_at(cut, caps, m3, sem_triples, model, topic_cache):
    """Topic + event chains from the video before `cut` seconds only."""
    caps = [c for c in caps if secs(c["end_time"]) <= cut]
    m3 = [c for c in m3 if secs(c["end_time"]) <= cut]
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

    def topic(ent):
        lines = [f"[{fmt_hm(c['start_time'])}] {c['text']}" for c in caps if ent in c["text"].lower()]
        key = json.dumps([ent, lines])
        if key in topic_cache:           # same entity, same captions as an earlier cutoff -> same prompt
            return ent, topic_cache[key]
        facts = []
        for i in range(0, len(lines), 50):
            r = call(model, TOPIC_PROMPT.format(entity=ent, episodes_text="\n\n".join(lines[i:i + 50])), system=SYSTEM_PROMPT)
            if isinstance(r, list):
                facts += [f for f in r if isinstance(f, dict) and f.get("fact")]
        res = {"entity": ent, "has_lifecycle": bool(facts), "facts": facts}
        topic_cache[key] = res
        return ent, res
    with ThreadPoolExecutor(32) as ex:
        topics = dict(ex.map(topic, cands))

    # event chains: step 1 over 3 min summaries (EgoLife: 10 min per day), step 2 merge (+ round 2), step 3 enrich
    enriched = []
    if m3:
        seg = "\n\n".join(f"[{fmt_hms(c['start_time'])}] {c['text']}" for c in m3)
        acts = call(model, EVENT_STEP1.format(date="DAY1", segments=seg), max_tokens=4000) or []
        acts = [x for x in acts if isinstance(x, dict)]
        daily = "\n=== DAY1 ===\n" + "".join(
            f"- [{x.get('time_range', '')}] {x.get('name', '')}: {x.get('summary', '')}\n"
            + (f"  Entities: {', '.join(map(str, x['key_entities']))}\n" if x.get("key_entities") else "") for x in acts)
        chains = call(model, EVENT_STEP2.format(daily_activities=daily), max_tokens=MERGE_TOKENS) or []
        chains = [c for c in chains if isinstance(c, dict) and isinstance(c.get("steps"), list)]
        if chains:
            found = "\n".join(f"- {c.get('name', '')}" for c in chains)
            r2 = call(model, "Below are activities from a first-person kitchen video, and a list of cross-time event chains already found.\n"
                      "Your task: find ADDITIONAL cross-time event chains that were MISSED in the first round. Do not repeat any chain already found.\n\n"
                      f"Already found chains (do NOT repeat these):\n{found}\n\nActivities:\n{daily}\n\n"
                      "For each NEW event chain, output:\n- name: specific name\n- steps: list of {day, start_time, end_time, description}\n"
                      "- key_entities: important objects and places only\n\nReturn JSON list. If no additional chains found, return [].",
                      max_tokens=MERGE_TOKENS)
            names = {c.get("name", "").lower() for c in chains}
            chains += [c for c in (r2 or []) if isinstance(c, dict) and isinstance(c.get("steps"), list)
                       and c.get("name", "").lower() not in names]

        def to_s(t):
            m = re.match(r"(\d{1,2}):(\d{2}):?(\d{2})?", str(t))
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0) if m else None

        def enrich(ec):
            ec = copy.deepcopy(ec)
            for st in ec["steps"]:
                if not isinstance(st, dict):
                    continue
                st["day"] = "DAY1"
                s0, s1 = to_s(st.get("start_time", "")), to_s(st.get("end_time", ""))
                if s0 is None or s1 is None:
                    continue
                if s0 > cut:   # a step the model placed after the cutoff cannot be grounded; drop it
                    st["start_time"] = st["end_time"] = ""
                    continue
                txt = [f"[{fmt_hms(c['start_time'])}] {c['text']}" for c in caps if s0 <= secs(c["start_time"]) <= s1]
                if not txt:
                    continue
                r = call(model, EVENT_STEP3.format(chain_name=ec.get("name", ""), day="DAY1", start_time=st["start_time"],
                                                   end_time=st["end_time"], captions_text="\n\n".join(txt)), max_tokens=1024)
                if isinstance(r, dict) and r.get("description"):
                    st["description"] = r["description"]
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
        if os.path.exists(os.path.join(src, "topic_cache.json")) and not os.path.exists(os.path.join(cdir, "topic_cache.json")):
            shutil.copy(os.path.join(src, "topic_cache.json"), cdir)
        log.info(f"chains for {len(got)} of {len(todo)} missing cutoffs taken from the dataset ({path})")


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
        caps, m3 = jload(os.path.join(mem, f"{video}_30sec.json")), jload(os.path.join(mem, f"{video}_3min.json"))
        sem = jload(os.path.join(mem, f"semantic_extraction_results_{TAG}.json"))["semantic_triples"]
        model, cache_f = llm(), os.path.join(cdir, "topic_cache.json")
        cache = jload(cache_f) if os.path.exists(cache_f) else {}
        for cut in todo:
            failed = roles.errors()
            topics, events = chains_at(cut, caps, m3, sem, model, cache)
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
