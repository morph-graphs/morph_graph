#!/usr/bin/env python3
"""An answer arm (g5m-q8, g5m-q27, g5m-g51) keeps a copy of the g5m memory. When the dataset's memory is another one
(rebuilt since), the arm must take it again and drop what it made from the old copy, not answer from that copy.
run_merit.sh's own `build` for an answer arm, with no GPU and no network: the Hub is a stand-in module serving a folder.

    python baselines/merit/epic/test_stale_memory.py      (or pytest baselines/merit/epic/test_stale_memory.py)
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

MERIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V, ARM = "P01-20240101-000001", "g5m-q8"
FILES = [f"{V}_30sec.json", f"{V}_30sec_4_key.json"]
HUB = '''import hashlib, os
from .utils import EntryNotFoundError, LocalEntryNotFoundError
ROOT = os.environ["FAKE_HUB"]
class constants:
    HF_HUB_CACHE = os.path.join(ROOT, "cache")
class Info:
    def __init__(self, path):
        d = open(os.path.join(ROOT, path), "rb").read()
        self.path, self.lfs, self.blob_id = path, None, hashlib.sha1(b"blob %d\\0" % len(d) + d).hexdigest()
class HfApi:
    def get_paths_info(self, repo, paths, repo_type=None):
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise OSError("offline")
        return [Info(p) for p in paths if os.path.isfile(os.path.join(ROOT, p))]
def hf_hub_download(repo, path, repo_type=None, revision=None, **kw):
    # a job node without the Hub: its cache has the files only at the commit they were prefetched at ("C1"), and the
    # cache's "main" has moved on since
    if os.environ.get("HF_HUB_OFFLINE") == "1" and revision != "C1":
        raise LocalEntryNotFoundError(path)
    if not os.path.isfile(os.path.join(ROOT, path)):
        raise EntryNotFoundError(path)
    return os.path.join(ROOT, path)
snapshot_download = None
'''


def write(folder, text):
    os.makedirs(folder, exist_ok=True)
    for f in FILES:
        open(os.path.join(folder, f), "w").write(json.dumps([{"file": f, "memory": text}]))


def ref(folder):   # as run_merit.sh memory_ref writes it
    return json.dumps({"memory_arm": "g5m", "sha256": {f: hashlib.sha256(open(os.path.join(folder, f), "rb").read()).hexdigest()
                                                         for f in FILES}}, sort_keys=True) + "\n"


def test_answer_arm_takes_a_rebuilt_memory():
    with tempfile.TemporaryDirectory() as tmp:
        # run_merit.sh and what its build stage of an answer arm touches, copied: the submodule is only checked for
        repo = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(repo, "baselines/merit/MERIT/eval"))
        open(os.path.join(repo, "baselines/merit/MERIT/eval/evaluate_egolife.py"), "w").close()
        shutil.copy(os.path.join(MERIT, "run_merit.sh"), os.path.join(repo, "baselines/merit"))
        shutil.copytree(os.path.join(MERIT, "..", "common"), os.path.join(repo, "baselines/common"))
        os.makedirs(os.path.join(tmp, "stub/huggingface_hub"))
        open(os.path.join(tmp, "stub/huggingface_hub/__init__.py"), "w").write(HUB)
        open(os.path.join(tmp, "stub/huggingface_hub/utils.py"), "w").write("class EntryNotFoundError(Exception):\n    pass\n"
                                                                    "class LocalEntryNotFoundError(EntryNotFoundError):\n    pass\n")
        hub, out, work = os.path.join(tmp, "hub/merit/g5m", V), os.path.join(tmp, "results"), os.path.join(tmp, "work")
        mem, mine = os.path.join(out, "g5m", V), os.path.join(out, ARM, V)
        made = [os.path.join(mine, "answers_tb.json.gz"), os.path.join(work, f"cache/{ARM}_{V}/emb"),
                os.path.join(work, f"runs/{ARM}/{V}/tb/x_final.json")]   # what the arm made from its memory
        os.makedirs(os.path.join(tmp, "epic"))
        env = dict(os.environ, PY=sys.executable, EPIC_MORPH_DIR=os.path.join(tmp, "epic"), MERIT_WORK=work, MERIT_OUT=out,
                   PYTHONPATH=os.path.join(tmp, "stub"), FAKE_HUB=os.path.join(tmp, "hub"), CUDA_VISIBLE_DEVICES="0")
        env.pop("MERIT_FETCH", None)

        def build(ok=True, **more):
            for f in made:
                os.makedirs(os.path.dirname(f), exist_ok=True)
                open(f, "w").write("made from the memory here")
            p = subprocess.run(["bash", os.path.join(repo, "baselines/merit/run_merit.sh"), "build", V, ARM],
                               env=dict(env, **more), capture_output=True, text=True)
            assert (p.returncode == 0) == ok, p.stdout + p.stderr
            return p.stdout

        def state():
            return (open(os.path.join(mem, FILES[0])).read(), [os.path.exists(f) for f in made],
                    open(os.path.join(mine, f"{V}_memory_ref.json")).read() == ref(mem))

        write(hub, "frameless")
        write(mem, "frameless")
        os.makedirs(mine)
        open(os.path.join(mine, "runs.jsonl"), "w").write("{}\n")
        open(os.path.join(mine, f"{V}_memory_ref.json"), "w").write(ref(mem))
        # the copy is the dataset's memory: nothing is fetched, nothing dropped
        assert "taking it again" not in build() and state() == (open(os.path.join(hub, FILES[0])).read(), [True] * 3, True)
        # the dataset's memory was rebuilt: MERIT_FETCH=0 keeps the local one (a memory built here, never the dataset's)
        write(hub, "rebuilt with frames")
        assert "taking it again" not in build(MERIT_FETCH="0") and "frameless" in state()[0] and state()[1] == [True] * 3
        # a node that cannot reach the Hub and has no prefetched copy cannot tell: the task stops, nothing is touched
        assert "could not reach the dataset" in build(ok=False, HF_HUB_OFFLINE="1")
        assert "frameless" in state()[0] and state()[1] == [True] * 3
        # ... also when the worker's prefetch listed the folder (it is in the dataset, only not in this cache)
        seen = os.path.join(tmp, "hub/cache/prefetched", f"{V}__g5m.json")
        os.makedirs(os.path.dirname(seen))
        json.dump(dict(commit="C0", files=[f"merit/g5m/{V}/{f}" for f in FILES]), open(seen, "w"))
        build(ok=False, HF_HUB_OFFLINE="1")
        # ... and goes on with the local memory when the prefetch saw no such memory in the dataset (a folder with
        # run records only is not one)
        json.dump(dict(commit="C1", files=[f"merit/g5m/{V}/runs.jsonl", f"magic/g5m/{V}/graph.pkl"]), open(seen, "w"))
        assert "no memory in the dataset yet" in build(HF_HUB_OFFLINE="1") and "frameless" in state()[0]
        # ... and takes the dataset's memory at the commit its prefetch saw, where the cache has it
        json.dump(dict(commit="C1", files=[f"merit/g5m/{V}/{f}" for f in FILES]), open(seen, "w"))
        assert "taking it again" in build(HF_HUB_OFFLINE="1") and "rebuilt" in state()[0] and state()[1:] == ([False] * 3, True)
        os.remove(seen)
        write(mem, "frameless")   # back to the stale copy, for the same without the offline detour
        open(os.path.join(mine, f"{V}_memory_ref.json"), "w").write(ref(mem))
        # ... otherwise the arm takes the dataset's, and its answers, embeddings and answering runs of the old one go
        assert "taking it again" in build()
        assert state() == (open(os.path.join(hub, FILES[0])).read(), [False] * 3, True) and "rebuilt" in state()[0]
        assert os.path.exists(os.path.join(mine, "runs.jsonl")) and sorted(os.listdir(mem)) == sorted(FILES)
        assert "taking it again" not in build() and state()[1] == [True] * 3            # and then leaves it alone
        # a dataset without the memory (never uploaded): the local one stays
        shutil.rmtree(hub)
        build()
        assert "rebuilt" in state()[0] and state()[1:] == ([True] * 3, True)


if __name__ == "__main__":
    test_answer_arm_takes_a_rebuilt_memory()
    print("ok test_answer_arm_takes_a_rebuilt_memory")
