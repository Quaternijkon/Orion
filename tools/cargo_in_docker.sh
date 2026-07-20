#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ORION_RUST_TOOL_IMAGE:-orion-rust-test:1.94-proto}"
dockerfile="$repo_root/tools/docker/Dockerfile.rust-tools"
target_dir="${CARGO_TARGET_DIR:-/proj/intelisys-PG0/exp/orion-distributed/cargo-target-native}"

if ! sudo -n docker image inspect "$image" >/dev/null 2>&1; then
    sudo -n docker build \
        --file "$dockerfile" \
        --tag "$image" \
        "$repo_root"
fi

mounts=(
    --volume "orion-cargo-cache:/usr/local/cargo/registry"
    --volume "$repo_root:$repo_root"
)

for shared_path in /proj/intelisys-PG0 /users/dry/orion-distributed; do
    if [[ -e "$shared_path" ]]; then
        mounts+=(--volume "$shared_path:$shared_path")
    fi
done

exec sudo -n docker run --rm \
    "${mounts[@]}" \
    --network host \
    --workdir "$repo_root" \
    --env "CARGO_TARGET_DIR=$target_dir" \
    "$image" \
    cargo "$@"
