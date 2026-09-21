"""Check newly added public references without gating on unrelated server outages."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urldefrag
from urllib.request import Request, urlopen

MAX_BODY = 4 * 1024 * 1024


class Anchors(HTMLParser):
    """Collect static anchors, using the same rendered IDs a link checker can see."""

    def __init__(self):
        super().__init__()
        self.names = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"id", "name"} and value:
                self.names.add(value)


def added_urls(patch):
    """Extract external URLs only from added Markdown/RST lines."""
    urls = set()
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for candidate in re.findall(r'https?://[^\s<>"\x27]+', line[1:]):
            # Preserve balanced parentheses inside URLs, removing Markdown's closing wrapper.
            candidate = candidate.rstrip(".,;]}")
            while candidate.endswith(")") and candidate.count(")") > candidate.count("("):
                candidate = candidate[:-1]
            if (
                "{" not in candidate
                and "PACKAGE" not in candidate
                and "REPOSITORY" not in candidate
            ):
                urls.add(candidate)
    return sorted(urls)


def inspect_url(url):
    """Classify a confirmed broken reference separately from temporary unavailability."""
    address, fragment = urldefrag(url)
    last = "unavailable"
    for attempt in range(2):
        try:
            request = Request(address, headers={"User-Agent": "OSS-documentation-check/1.0"})
            with urlopen(request, timeout=15) as response:
                kind = response.headers.get("Content-Type", "")
                if not fragment or "html" not in kind:
                    return {"url": url, "status": "ok"}
                body = response.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                return {"url": url, "status": "deferred", "reason": "large HTML requires review"}
            parser = Anchors()
            parser.feed(body.decode("utf-8", "replace"))
            if unquote(fragment) in parser.names:
                return {"url": url, "status": "ok"}
            last = f"static anchor #{fragment} not found"
        except HTTPError as error:
            if error.code not in {404, 410}:
                return {"url": url, "status": "deferred", "reason": f"HTTP {error.code}"}
            last = f"HTTP {error.code}"
        except (URLError, TimeoutError, OSError) as error:
            return {"url": url, "status": "deferred", "reason": str(error)}
        if attempt == 0:
            time.sleep(0.5)
    return {"url": url, "status": "broken", "reason": last}


def main():
    """Check introduced references; emit a machine-readable maintenance report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patch = subprocess.check_output(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--unified=0",
            args.base,
            args.head,
            "--",
            "*.md",
            "*.rst",
        ],
        text=True,
        encoding="utf-8",
    )
    results = [inspect_url(url) for url in added_urls(patch)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['status']}: {result['url']} {result.get('reason', '')}")
    deferred = sum(item["status"] == "deferred" for item in results)
    print(
        f"Checked {len(results)} introduced references; {deferred} need external-health follow-up."
    )
    raise SystemExit(1 if any(item["status"] == "broken" for item in results) else 0)


if __name__ == "__main__":
    main()
