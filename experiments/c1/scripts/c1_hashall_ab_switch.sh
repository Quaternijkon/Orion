#!/usr/bin/env bash
set -euo pipefail

variant=${1:?usage: c1_hashall_ab_switch.sh official|clean|instrumented}
storage_root=${2:-/users/dry/orion-distributed/hashall-ab-20260824}
case "$variant" in
  official)
    image=qdrant/qdrant:v1.17.1
    hardware_reporting=false
    extra_env=()
    ;;
  clean)
    image=orion-method4:fe61fdac6760-source-cb53046429a7
    hardware_reporting=false
    extra_env=()
    ;;
  instrumented)
    image=orion-c1:20260821-deterministic-seed
    hardware_reporting=true
    extra_env=(-e QDRANT_HNSW_GRAPH_BUILD_SEED=20260821)
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

nodes=(10.10.1.1 10.10.1.2 10.10.1.3 10.10.1.4)
names=(hashall-ab-node0 hashall-ab-node1 hashall-ab-node2 hashall-ab-node3)
storage=(node0 node1 node2 node3)

for index in 0 1 2 3; do
  host=${nodes[$index]}
  name=${names[$index]}
  if [[ $index -eq 0 ]]; then
    sudo -n docker rm -f "$name" >/dev/null 2>&1 || true
  else
    ssh "$host" "sudo -n docker rm -f '$name' >/dev/null 2>&1 || true"
  fi
done

common=(
  --network host
  --cpuset-cpus=0-19
  --ulimit nofile=65536:65536
  -e QDRANT__CLUSTER__ENABLED=true
  -e QDRANT__CLUSTER__P2P__PORT=6335
  -e QDRANT__SERVICE__HTTP_PORT=6333
  -e QDRANT__SERVICE__GRPC_PORT=6334
  -e "QDRANT__SERVICE__HARDWARE_REPORTING=$hardware_reporting"
  -e QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=16
  -e QDRANT__STORAGE__PERFORMANCE__OPTIMIZER_CPU_BUDGET=4
)

sudo -n docker run -d --name hashall-ab-node0 \
  "${common[@]}" "${extra_env[@]}" \
  -v "$storage_root/node0/storage:/qdrant/storage" \
  "$image" ./qdrant --uri http://10.10.1.1:6335 >/dev/null

for _ in $(seq 1 60); do
  if curl -fsS --max-time 1 http://10.10.1.1:6333/ >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS --max-time 2 http://10.10.1.1:6333/ >/dev/null

for index in 1 2 3; do
  host=${nodes[$index]}
  name=${names[$index]}
  store=${storage[$index]}
  ssh "$host" sudo -n docker run -d --name "$name" \
    "${common[@]}" "${extra_env[@]}" \
    -v "$storage_root/$store/storage:/qdrant/storage" \
    "$image" ./qdrant --bootstrap http://10.10.1.1:6335 \
    --uri "http://$host:6335" >/dev/null &
done
wait

for _ in $(seq 1 90); do
  if curl -fsS --max-time 2 http://10.10.1.1:6333/cluster | python -c '
import json, sys
d = json.load(sys.stdin)["result"]
expected = {f"http://10.10.1.{i}:6335" for i in range(1, 5)}
actual = {row["uri"].rstrip("/") for row in d["peers"].values()}
raise SystemExit(0 if actual == expected and d["raft_info"]["pending_operations"] == 0 else 1)
'; then
    curl -fsS http://10.10.1.1:6333/
    exit 0
  fi
  sleep 1
done

curl -sS http://10.10.1.1:6333/cluster | python -m json.tool >&2 || true
exit 1
