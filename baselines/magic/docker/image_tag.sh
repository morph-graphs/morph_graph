#!/usr/bin/env bash
# The image tag for an arm: <arm>-<12 hex of a hash over the files that define the image> (m27 also takes
# MERIT's m27 environment files, which it installs). Any commit with the same environment files maps to the
# same image, so a code-only commit on main needs no rebuild. Used by build.sh, run.sh and
# .github/workflows/baseline-images.yml.
#   bash baselines/magic/docker/image_tag.sh <m8|m27>
set -euo pipefail
ARM=${1:?m8 or m27}; B=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # baselines/
case $ARM in m8|m27) ;; *) echo "arm must be m8 or m27" >&2; exit 1 ;; esac
F=(magic/docker/Dockerfile magic/env/install_requirements.sh magic/env/wheels-client.txt magic/env/requirements-client.txt)
[ "$ARM" = m27 ] && F+=(merit/env/install_requirements.sh merit/env/wheels-m27.txt merit/env/requirements-m27.txt
                        merit/env/requirements-m27-nodeps.txt)
H=$(cd "$B" && cat "${F[@]}" | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
echo "$ARM-$H"
