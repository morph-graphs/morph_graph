#!/usr/bin/env bash
# The image tag: <arm>-<12 hex of a hash over the files that define the image>. Any commit with the same
# environment files maps to the same image, so a code-only commit needs no rebuild. One image (gpu) serves
# both of EGAgent's arms; an arm name is accepted so callers can pass theirs.
#   bash baselines/egagent/docker/image_tag.sh [gpu|gpt41|gemini25pro]
set -euo pipefail
ARM=${1:-gpu}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case $ARM in gpu|gpt41|gemini25pro) ;; *) echo "arm must be gpu (the image), gpt41 or gemini25pro" >&2; exit 1 ;; esac
H=$(cd "$M" && cat docker/Dockerfile env/install_requirements.sh env/setup_env.sh env/requirements-client.txt \
    | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "gpu-$H"
