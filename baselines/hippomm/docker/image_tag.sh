#!/usr/bin/env bash
# The image tag for the arm: <arm>-<12 hex of a hash over the files that define the image>. Any commit with the
# same environment files maps to the same image, so a code-only commit needs no rebuild. As MERIT's.
#   bash baselines/hippomm/docker/image_tag.sh q25vl7b-gpt4o
set -euo pipefail
ARM=${1:?q25vl7b-gpt4o}; M=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# one image for both arms (q25vl7b-gpt41 differs only in the model the API is asked for): it carries the first arm's name
case $ARM in q25vl7b-gpt4o|q25vl7b-gpt41) ARM=q25vl7b-gpt4o ;; *) echo "arm must be q25vl7b-gpt4o or q25vl7b-gpt41" >&2; exit 1 ;; esac
FILES=(docker/Dockerfile env/install_requirements.sh env/setup_env.sh env/requirements-client.txt env/requirements-vllm.txt)
H=$(cd "$M" && cat "${FILES[@]}" | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$H"
