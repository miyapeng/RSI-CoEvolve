#!/usr/bin/env bash
# Build the unmodified pinned upstream Dockerfile; optionally publish to TCR.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
repo=${VISION2WEB_REPO:-"$project_dir/../references/Vision2Web"}
engine=${CONTAINER_ENGINE:-podman}
image=${1:?Usage: build_vision2web_image.sh modelbest.tencentcloudcr.com/NAMESPACE/vision2web:TAG [--push]}
pin=d80a4a3d0ea7326541ad87314294269a633789e4
[[ "$image" == modelbest.tencentcloudcr.com/*:* ]] || { echo 'Use an explicit tagged TCR image' >&2; exit 2; }
[[ "$engine" == podman || "$engine" == docker ]] || { echo 'CONTAINER_ENGINE must be podman or docker' >&2; exit 2; }
[[ $# -eq 1 || ( $# -eq 2 && "$2" == --push ) ]] || exit 2
[[ $(git -C "$repo" rev-parse HEAD) == "$pin" ]] || { echo 'Unexpected Vision2Web commit' >&2; exit 2; }
[[ -z $(git -C "$repo" status --porcelain --untracked-files=no) ]] || { echo 'Tracked upstream source is dirty' >&2; exit 2; }
"$engine" build --network host --label "org.opencontainers.image.revision=$pin" \
  --tag "$image" --file "$repo/docker/Dockerfile" "$repo/docker"
if [[ ${2:-} == --push ]]; then
  "$engine" push "$image"
  # Registry authentication stays in REGISTRY_AUTH_FILE / engine credential store.
  digest=$(skopeo inspect --format '{{.Digest}}' "docker://$image")
  printf 'VISION2WEB_SANDBOX_IMAGE=%s@%s\n' "$image" "$digest"
fi
