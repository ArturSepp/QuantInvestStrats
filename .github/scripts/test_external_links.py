"""Offline regressions for confirmed failures, remote outages, and builder crashes."""

import json
import contextlib
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import check_external_links as checker


class Handler(BaseHTTPRequestHandler):
    """Serve real HTTP outcomes without contacting external websites."""

    def do_GET(self):
        code = {"/gone": 410, "/missing": 404, "/outage": 504, "/blocked": 403}.get(
            self.path, 200
        )
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b'<html><h1 id="present">Reference</h1></html>')

    def log_message(self, *args):
        pass


class ExternalLinkTests(unittest.TestCase):
    """A server outage is advisory; a confirmed 404, 410, or missing anchor is fatal."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.address = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_real_http_outcomes_and_anchor_confirmation(self):
        paths = ["/missing", "/gone", "/outage", "/blocked", "/ok#absent", "/ok#present"]
        records = [{"uri": self.address + path, "status": "broken"} for path in paths]
        results = checker.confirm_results(records)
        self.assertEqual(
            [r["status"] for r in results],
            ["broken", "broken", "deferred", "deferred", "broken", "ok"],
        )
        self.assertEqual(results[2]["sphinx"], records[2])

    def test_sphinx_service_unavailable_remains_visible(self):
        record = {"uri": self.address + "/outage", "status": "ignored",
                  "info": "service unavailable"}
        self.assertEqual(checker.confirm_results([record])[0]["status"], "deferred")

    def run_builder(self, records, returncode=1):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def fake_run(command, check):
                (output / "output.json").write_text(
                    "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
                )
                return type("Completed", (), {"returncode": returncode})()

            with patch.object(checker.subprocess, "run", fake_run), contextlib.redirect_stdout(
                io.StringIO()
            ):
                return checker.main(["--output-dir", str(output)])

    def test_real_sphinx_report_keeps_broken_links_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "conf.py").write_text('project = "Link regression"\n', encoding="utf-8")
            (source / "index.rst").write_text(
                "References\n==========\n\n"
                f"`Missing <{self.address}/missing>`_\n\n"
                f"`Outage <{self.address}/outage>`_\n",
                encoding="utf-8",
            )
            output = Path(temporary) / "build"
            with contextlib.redirect_stdout(io.StringIO()):
                result = checker.main(["--source", str(source), "--output-dir", str(output)])
            self.assertEqual(result, 1)
            report = json.loads((output / "confirmed.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(r["status"] for r in report), ["broken", "deferred"])

    def test_unavailability_does_not_fail_run(self):
        self.assertEqual(self.run_builder([
            {"uri": self.address + "/outage", "status": "broken"}
        ]), 0)

    def test_confirmed_missing_link_fails_even_with_an_outage(self):
        self.assertEqual(self.run_builder([
            {"uri": self.address + "/outage", "status": "broken"},
            {"uri": self.address + "/missing", "status": "broken"},
        ]), 1)

    def test_builder_error_is_never_overridden(self):
        with self.assertRaises(SystemExit):
            self.run_builder([], returncode=2)

    def test_stale_report_cannot_hide_builder_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "output.json").write_text('{"status":"working"}\n', encoding="utf-8")
            completed = type("Completed", (), {"returncode": 2})()
            with patch.object(checker.subprocess, "run", return_value=completed):
                with self.assertRaises(SystemExit):
                    checker.main(["--output-dir", str(output)])


if __name__ == "__main__":
    unittest.main()
