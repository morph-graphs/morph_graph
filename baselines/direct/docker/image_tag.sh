#!/usr/bin/env bash
# The image tag an arm (or model) of the direct baseline runs in. Direct has no image of its own: each model runs in
# the MERIT environment of env/setup_env.sh, so this is MERIT's tag for it (ghcr.io/morph-graphs/merit:<tag>,
# baselines/merit/docker/image_tag.sh). Used by run.sh; a new model adds its name to the case below.
#   bash baselines/direct/docker/image_tag.sh qwen3vl-8b-f128      -> m8-<environment hash>
set -euo pipefail
ARM=${1:?arm (<model>-f<N>) or model}; D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case ${ARM%-f[0-9]*} in qwen3vl-8b) ENV=m8 ;; qwen38-27b) ENV=m27 ;; *) echo "unknown model in $ARM" >&2; exit 1 ;; esac
exec bash "$D/../../merit/docker/image_tag.sh" "$ENV"
