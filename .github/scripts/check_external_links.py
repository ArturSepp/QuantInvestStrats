"""Check external links without treating a remote outage as a documentation defect.

The separate HTML build remains strict. Sphinx collects every link; failed targets
are confirmed with the same policy used by the introduced-reference PR gate.
Unavailable targets remain visible in annotations and the retained JSON report.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_new_references import inspect_url  # noqa: E402


def confirm_results(records):
    """Retain Sphinx evidence and independently confirm each failed target."""
    results = []
    for record in records:
        if record["status"] in {"broken", "timeout"}:
            result = inspect_url(record["uri"])
        elif record["status"] == "ignored" and record.get("info") == "service unavailable":
            result = {"url": record["uri"], "status": "deferred", "reason": record["info"]}
        else:
            continue
        results.append({**result, "sphinx": record})
    return results


def build_links(source, output):
    """Apply typed Sphinx overrides, including its floating-point rate-limit budget."""
    from sphinx.application import Sphinx

    application = Sphinx(
        srcdir=str(source), confdir=str(source), outdir=str(output),
        doctreedir=str(output / ".doctrees"), buildername="linkcheck", freshenv=True,
        confoverrides={
            "linkcheck_workers": 2, "linkcheck_retries": 2, "linkcheck_timeout": 15,
            "linkcheck_rate_limit_timeout": 30.0,
            "linkcheck_report_timeouts_as_broken": True,
        },
    )
    application.build(force_all=True)
    return application.statuscode


def main(argv=None):
    """Run bounded Sphinx checks and fail on confirmed broken links or builder errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.build:
        try:
            return build_links(args.source, args.output_dir)
        except Exception:
            # A partial report cannot make an unexpected builder crash advisory.
            traceback.print_exc()
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "output.json"
    # A crashed builder must never reuse a successful report from an earlier run.
    output.unlink(missing_ok=True)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--build",
        "--source", str(args.source), "--output-dir", str(args.output_dir),
    ]
    completed = subprocess.run(command, check=False)
    if not output.is_file():
        raise SystemExit("Sphinx produced no link report; the documentation check failed")
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if completed.returncode not in {0, 1} or (
        completed.returncode and not any(r["status"] in {"broken", "timeout"} for r in records)
    ):
        raise SystemExit(completed.returncode or 1)
    results = confirm_results(records)
    report = args.output_dir / "confirmed.json"
    report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        level = "error" if result["status"] == "broken" else "warning"
        if result["status"] == "ok":
            print(f"Recovered: {result['url']}")
        else:
            message = f"{result['url']}: {result.get('reason', '')}"
            message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::{level}::{message}")
    broken = sum(result["status"] == "broken" for result in results)
    deferred = sum(result["status"] == "deferred" for result in results)
    print(f"{len(records)} links checked; {broken} confirmed broken; {deferred} unavailable")
    return int(broken > 0)


if __name__ == "__main__":
    raise SystemExit(main())
