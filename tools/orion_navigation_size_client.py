#!/usr/bin/env python3
"""Native SearchBatch client; routing, MultiEP and merging remain on the server.

The controller starts this only after an explicit --execute. Every response is
validated and scored; fixed-window completions (including batch size) define QPS.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
from pathlib import Path
import time


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def varint(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def scalar(number, value):
    return varint(number << 3) + varint(value)


def field(number, value):
    return varint((number << 3) | 2) + varint(len(value)) + value


class Client:
    def __init__(self, spec):
        import numpy as np
        self.spec = spec
        with np.load(spec["queries_path"]) as data:
            self.queries, self.truth = data["queries"], data["truth"]
        self.prefix = field(1, spec["collection"].encode())
        self.suffix = scalar(4, 60)
        self.bodies = [field(2, self.prefix + field(2, np.asarray(q, dtype="<f4").tobytes())
                           + scalar(4, 10) + field(6, scalar(1, 0)) + field(11, scalar(1, 0)))
                       for q in self.queries]

    async def __aenter__(self):
        import grpc
        from qdrant_client.grpc import SearchBatchResponse
        self.channel = grpc.aio.insecure_channel(
            f'{self.spec["host"]}:{self.spec["port"] + 1}',
            options=[("grpc.enable_http_proxy", 0), ("grpc.max_receive_message_length", 64 << 20)])
        await asyncio.wait_for(self.channel.channel_ready(), 30)
        self.call = self.channel.unary_unary("/qdrant.Points/SearchBatch",
                                            response_deserializer=SearchBatchResponse.FromString)
        return self

    async def __aexit__(self, *exc):
        await self.channel.close()

    async def search(self, qids):
        import numpy as np
        response = await self.call(self.prefix + b"".join(self.bodies[i] for i in qids) + self.suffix,
                                   timeout=65)
        if len(response.result) != len(qids):
            raise RuntimeError("Incomplete SearchBatch response")
        ids = []
        for result in response.result:
            row = [p.id.num for p in result.result]
            if (len(row) != 10 or len(set(row)) != 10 or
                    any(p.id.WhichOneof("point_id_options") != "num" for p in result.result) or
                    any(i >= self.spec["rows"] for i in row) or
                    any(not np.isfinite(p.score) for p in result.result)):
                raise RuntimeError("Invalid global Top10 response")
            ids.append(row)
        return np.asarray(ids, dtype=np.int64)


async def audit(spec, destination):
    import numpy as np
    indices = np.asarray(spec["query_ids"], dtype=np.int64)
    answers = np.empty((len(indices), 10), dtype=np.int64)
    gate = asyncio.Semaphore(8)
    async with Client(spec) as client:
        async def batch(start):
            async with gate:
                ids = indices[start:start + spec["batch_size"]]
                answers[start:start + len(ids)] = await client.search(ids)
        await asyncio.gather(*(batch(i) for i in range(0, len(indices), spec["batch_size"])))
        hits = (answers[:, :, None] == client.truth[indices, None, :]).any(axis=2).sum(axis=1)
    np.savez_compressed(str(destination) + ".npz", query_ids=indices, answers=answers, hits=hits)
    result = {"recall": float(hits.sum() / (10 * len(indices))), "queries": len(indices),
              "hits": int(hits.sum())}
    save(str(destination) + ".json", result)
    return result


async def worker(spec, wid):
    import numpy as np
    dest = Path(spec["directory"])
    counts = np.zeros(len(spec["query_ids"]), dtype=np.int64)
    hits = np.zeros_like(counts)
    errors, batches = [], []
    marks = []
    async with Client(spec) as client:
        order = np.random.default_rng(spec["seed"]).permutation(spec["query_ids"])
        # Counts are stored in permutation order and mapped back at aggregation.
        save(dest / f"w{wid}.ready", {"pid": os.getpid()})
        deadline = time.monotonic() + 300
        while not (dest / "start.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Controller did not release the measurement barrier")
            await asyncio.sleep(.1)
        origin = json.loads((dest / "start.json").read_text())["start_monotonic"]
        warm, end = spec["warmup_seconds"], spec["warmup_seconds"] + spec["seconds"]

        async def mark_cpu():
            for at in (warm, end):
                await asyncio.sleep(max(0, origin + at - time.monotonic()))
                marks.append([time.monotonic(), time.process_time()])

        async def lane(number):
            await asyncio.sleep(max(0, origin - time.monotonic()))
            step = 0
            while time.monotonic() < origin + end:
                # Contiguous batches ensure coverage even when workers*lanes
                # shares factors with 2000/8000 (striding by that product would not).
                positions = (wid * 997 + number * 97 + step * spec["batch_size"]
                             + np.arange(spec["batch_size"])) % len(order)
                qids = order[positions]
                began = time.monotonic() - origin
                try:
                    answers = await client.search(qids)
                    ended = time.monotonic() - origin
                    matched = (answers[:, :, None] == client.truth[qids, None, :]).any(axis=2).sum(axis=1)
                    batches.append((began, ended, len(qids)))
                    if warm <= ended < end:
                        np.add.at(counts, positions, 1)
                        np.add.at(hits, positions, matched)
                except Exception as exc:
                    errors.append(repr(exc))
                    return
                step += 1
        await asyncio.gather(mark_cpu(), *(lane(i) for i in range(spec["lanes"])))
    np.savez_compressed(dest / f"w{wid}.npz", batches=np.asarray(batches).reshape(-1, 3),
                        query_ids=order, counts=counts, hits=hits)
    save(dest / f"w{wid}.json", {"errors": errors, "cpu_marks": marks,
                                "affinity": sorted(os.sched_getaffinity(0))})


def run_worker(spec, wid):
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    os.sched_setaffinity(0, set(spec["client_cpus"]))
    asyncio.run(worker(spec, wid))


def host_main(spec):
    os.sched_setaffinity(0, set(spec["client_cpus"]))
    children = []
    context = multiprocessing.get_context("spawn")
    try:
        for i in range(spec["workers_per_host"]):
            wid = spec["host_index"] * spec["workers_per_host"] + i
            child = context.Process(target=run_worker, args=(spec, wid))
            child.start()
            children.append(child)
        for child in children:
            child.join()
        if any(child.exitcode for child in children):
            raise RuntimeError("A load worker failed; window is invalid")
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    host_main(json.loads(args.spec.read_text()))
