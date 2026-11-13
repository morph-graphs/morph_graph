#!/usr/bin/env bash
# Download the models run_m3agent.sh expects into one folder (M3AGENT_MODELS), at the revisions the port was
# checked with (2026-11-12, Nexus). Skips what is already there; rerun to resume.
#   PY_MEM=... PY_CTL=... bash baselines/m3agent/fetch_models.sh <models dir>     (in the image: docker/run.sh models)
# M3AGENT_BACKEND=api skips the two local stand-ins (Whisper and Qwen3-Embedding), which the API calls replace.
# The two M3-Agent checkpoints are ~84 GB; HF_TOKEN is not needed (public repos) but raises rate limits.
set -euo pipefail
D=${1:?models dir}; : "${PY_MEM:?}" "${PY_CTL:?}"
mkdir -p "$D"; D=$(cd "$D" && pwd)
REPOS=("ByteDance-Seed/M3-Agent-Memorization M3-Agent-Memorization 01480de764d07144bc4716c5610ef97684a27b58"
       "ByteDance-Seed/M3-Agent-Control M3-Agent-Control b06d8a332f547b04add06b59f8687afc9c39e409")
[ "${M3AGENT_BACKEND:-local}" = api ] || REPOS+=(
       "Qwen/Qwen3-Embedding-4B Qwen3-Embedding-4B 5cf2132abc99cad020ac570b19d031efec650f2b"
       "dropbox-dash/faster-whisper-large-v3-turbo faster-whisper-large-v3-turbo 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf")
for r in "${REPOS[@]}"; do
    set -- $r
    "$PY_CTL" -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1], revision=sys.argv[3], local_dir=sys.argv[2], max_workers=8)" \
        "$1" "$D/$2" "$3"
    echo "ok $2 ($1 @ ${3:0:7})"
done
# ERes2NetV2 speaker embeddings (ModelScope; the link in upstream's README)
if [ ! -s "$D/pretrained_eres2netv2.ckpt" ]; then
    curl -sSfL -o "$D/pretrained_eres2netv2.ckpt.part" \
        https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt
    mv "$D/pretrained_eres2netv2.ckpt.part" "$D/pretrained_eres2netv2.ckpt"
fi
echo "ok pretrained_eres2netv2.ckpt"
# insightface's buffalo_l (faces), downloaded by insightface itself, now rather than by the first build
"$PY_MEM" -c "import sys; from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root=sys.argv[1], providers=['CPUExecutionProvider'])" \
    "$D/insightface" >/dev/null
echo "ok insightface/models/buffalo_l"
