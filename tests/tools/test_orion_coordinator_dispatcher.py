"""Exercise the reference sender over HTTP, including a lost response on a reused socket."""
import contextlib
import json
import select
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def coordinator(handle_search):
    attempts = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, *_):
            pass

        def reply(self, status, data, unstarted=False):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if unstarted:
                self.send_header("X-Orion-Admission", "not-started")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/permits"):
                self.reply(200, {"result": {
                    "count": body["count"], "lease": str(uuid.uuid4()), "ttl_ms": 500,
                    "weight": 4.0, "limit": 32, "generation": 1,
                    "artifact_sha256": "graph", "policy_sha256": "policy",
                }})
                return
            self.query = body["search"]
            attempts.append(body)
            handle_search(self, len(attempts))

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, args=(0.02,), daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/collections/test", attempts
        finally:
            server.shutdown()
            thread.join()


class CoordinatorClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("g++") and shutil.which("pkg-config")):
            raise unittest.SkipTest("requires g++ and pkg-config")
        flags = subprocess.run(
            ["pkg-config", "--cflags", "--libs", "libcurl >= 7.80", "json-c"],
            capture_output=True, text=True,
        )
        if flags.returncode:
            raise unittest.SkipTest("requires libcurl >= 7.80 and json-c")
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.binary = Path(tmp.name) / "sender"
        subprocess.run([
            "g++", "-std=c++17", "-pthread", str(ROOT / "examples/orion_direct_search.cpp"),
            "-o", str(cls.binary), *shlex.split(flags.stdout),
        ], check=True)

    def send(self, url, count):
        body = json.dumps({"vector": [1, 2], "limit": 10}) + "\n"
        return subprocess.run([self.binary, url], input=body * count, text=True,
                              capture_output=True, timeout=8)

    def test_only_explicit_unstarted_response_allows_reselection(self):
        for marker in (False, True):
            with self.subTest(marker=marker):
                def handle(request, number):
                    if number == 1:
                        request.reply(412, {}, unstarted=marker)
                    else:
                        request.reply(200, {"result": []})

                with coordinator(handle) as (url, attempts):
                    process = self.send(url, 1)
                self.assertEqual(process.returncode == 0, marker)
                self.assertEqual(len(attempts), 2 if marker else 1)
                if marker:
                    self.assertNotEqual(attempts[0]["permit"], attempts[1]["permit"])
                    self.assertEqual(attempts[0]["search"], attempts[1]["search"])

    def test_waiting_input_runs_concurrently_and_replies_keep_input_order(self):
        # The first request can only finish after the second has reached its coordinator.
        second_arrived = threading.Event()

        def handle(request, _):
            label = request.query["vector"][0]
            if label == 1:
                if not second_arrived.wait(timeout=3):
                    request.reply(500, {"error": "sender did not use the available window"})
                    return
            else:
                second_arrived.set()
            request.reply(200, {"result": label})

        with coordinator(handle) as (url, attempts):
            lines = [json.dumps({"vector": [i, 2], "limit": 1}) for i in (1, 2)]
            process = subprocess.run([self.binary, url], input="\n".join(lines) + "\n",
                                     text=True, capture_output=True, timeout=8)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual([json.loads(line)["result"] for line in process.stdout.splitlines()], [1, 2])
        self.assertEqual(len(attempts), 2)

    def test_lost_reply_on_reused_connection_keeps_original_owner(self):
        addresses = []

        def handle(request, number):
            addresses.append(request.client_address)
            if number == 2:
                request.close_connection = True
                request.connection.shutdown(socket.SHUT_RDWR)
                request.connection.close()
            else:
                request.reply(200, {"result": []})

        # Feed the second line only after the first reply. This exercises both
        # a live input pipe and libcurl's automatic replay path on a reused socket.
        with coordinator(handle) as (url, attempts):
            line = json.dumps({"vector": [1, 2], "limit": 10}) + "\n"
            with subprocess.Popen([self.binary, url], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) as process:
                try:
                    process.stdin.write(line)
                    process.stdin.flush()
                    self.assertTrue(select.select([process.stdout], [], [], 5)[0])
                    self.assertEqual(json.loads(process.stdout.readline()), {"result": []})
                    later, _ = process.communicate(input=line, timeout=5)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(later, "")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(addresses[0], addresses[1])


if __name__ == "__main__":
    unittest.main()
