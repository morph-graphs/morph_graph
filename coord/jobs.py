"""The job board's store: the private dataset morph-graphs/epic-morph-jobs (DESIGN.md).

    tasks.json          the master's list: [{id, baseline, arm, stage, video, bank, pools, priority, est_gpu_h, deps}]
    pools/<pool>.json   one file per pool, written only by that pool's worker:
                        {pool, host, gpus, updated, claims: {task_id: {status, lease_until, claimed, started,
                         finished, seconds, gpu, exit, note}}}   status: running | relay | done | failed
                        relay: finished, its results wait in the worker's outbox for a relay to upload them
                        (worker.py --outbox); held like done, but not yet done for the tasks that depend on it
    master.json         the master's last cycle (time, counts)

A worker claims by committing its own pool file with parent_commit = the head it read: if anyone committed
since, the Hub refuses (412) and the worker re-reads, so two pools never hold the same task. A claim whose
lease ran out (worker gone, machine preempted) counts as free again.
"""
import json
import os
import time

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO = os.environ.get("COORD_JOBS_REPO", "morph-graphs/epic-morph-jobs")
LEASE_S = 1800


def now():
    return time.time()


def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t or now()))


def read():
    """(head sha, tasks, {pool: pool file}) from the dataset's current head."""
    api = HfApi()
    head = api.dataset_info(REPO).sha
    d = snapshot_download(REPO, repo_type="dataset", revision=head)
    tasks = json.load(open(os.path.join(d, "tasks.json"))) if os.path.exists(os.path.join(d, "tasks.json")) else []
    pools = {}
    pdir = os.path.join(d, "pools")
    for f in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
        if f.endswith(".json"):
            pools[f[:-5]] = json.load(open(os.path.join(pdir, f)))
    return head, tasks, pools


def taken(pools, t=None):
    """{task_id: (pool, claim)} for claims that hold a task now: done or relay, or running with a live lease."""
    t = t or now()
    out = {}
    for name, p in pools.items():
        for tid, c in p.get("claims", {}).items():
            if c["status"] in ("done", "relay") or (c["status"] == "running" and c.get("lease_until", 0) > t):
                if out.get(tid, (None, {}))[1].get("status") != "done":   # a done claim wins over a running one
                    out[tid] = (name, c)
    return out


def failures(pools):
    n = {}
    for p in pools.values():
        for tid, c in p.get("claims", {}).items():
            if c["status"] == "failed":
                n[tid] = n.get(tid, 0) + 1
    return n


def commit(files, head, message):
    """Write {path_in_repo: obj} in one commit on top of head. False when someone committed since (re-read)."""
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=json.dumps(o, indent=1).encode()) for p, o in files.items()]
    try:
        HfApi().create_commit(REPO, repo_type="dataset", operations=ops, commit_message=message, parent_commit=head)
        return True
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code in (409, 412):
            return False
        raise
