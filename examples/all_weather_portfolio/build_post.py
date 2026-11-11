"""Build a standalone HTML edition of the Markdown post with embedded figures and downloads."""

import argparse
import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
CSS = """
:root {color-scheme:light;--ink:#172831;--muted:#465b64;--rule:#d4dfe2;--accent:#15586b}
* {box-sizing:border-box}
body {margin:0;color:var(--ink);background:#fff;font:18px/1.68 'STIX Two Text',Georgia,serif}
main {max-width:58rem;margin:auto;padding:4.8rem clamp(1rem,5vw,4.2rem)}
h1,h2,h3,h4,summary,th {font-family:'Source Sans 3',Arial,sans-serif;line-height:1.2}
h1 {font-size:clamp(2.2rem,5.5vw,3.7rem);letter-spacing:-.045em;max-width:18em;margin:0 0 .6em}
h2 {font-size:2rem;margin:2.5em 0 .65em;padding-top:.7em;border-top:2px solid var(--ink)}
h3 {font-size:1.45rem;margin:2.2em 0 .6em} p {margin:.85em 0}
a {color:var(--accent);text-underline-offset:3px} a:hover {text-decoration-thickness:2px}
img {display:block;max-width:100%;height:auto;margin:1.4em auto .3em}
p:has(img)+p:has(>em:only-child) {color:var(--muted);font-size:.93em;margin-top:.35em}
pre,code {font-family:'IBM Plex Mono',Consolas,monospace;font-size:.88em}
pre {overflow:auto;background:#f3f6f6;border-left:3px solid #9ab9be;padding:1.1em;line-height:1.55}
pre code {font-size:1em} blockquote {margin:1em 0;padding:0 1.25em;border-left:3px solid #a6bbc1}
.table-wrap {overflow-x:auto;margin:1.2em 0}
table {border-collapse:collapse;font: .86em/1.45 'Source Sans 3',Arial,sans-serif;width:100%}
td,th {border-bottom:1px solid var(--rule);padding:.55em .65em;vertical-align:top}
th {text-align:left;background:#edf3f4} tr:nth-child(even) td {background:#fafcfc}
details {margin:1.25em 0;border:1px solid var(--rule);padding:.8em 1em} summary {cursor:pointer}
@media(max-width:650px){main{padding:2rem 1rem}body{font-size:16px}td,th{min-width:5.5em}}
@media print {main{padding:0;max-width:none}h2,h3{break-after:avoid}
img,table{break-inside:avoid}pre{white-space:pre-wrap}}
"""


def convert(source):
    source = re.sub(r"<!--.*?-->", "", source, flags=re.S)
    source = source.replace("<details>", '<details markdown="1">')
    return markdown.markdown(
        source, extensions=["tables", "fenced_code", "md_in_html", "toc", "sane_lists"]
    )


def render(post, output):
    root = post.resolve().parent
    soup = BeautifulSoup(convert(post.read_text(encoding="utf-8")), "html.parser")
    downloads = soup.new_tag("p")
    guide_link = soup.new_tag("a", href="README.md")
    guide_link.string = "Download the reproduction guide and run commands"
    downloads.append(guide_link)
    soup.append(downloads)

    for tag, attr in [(a, "href") for a in soup.select("a[href]")] + [
        (image, "src") for image in soup.select("img[src]")
    ]:
        link = urlsplit(tag[attr])
        if link.scheme or link.netloc or not link.path:
            continue
        target = (root / unquote(link.path)).resolve()
        target.relative_to(root)
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        payload = base64.b64encode(target.read_bytes()).decode("ascii")
        tag[attr] = f"data:{mime};base64,{payload}"
        if attr == "href":
            tag["download"] = target.name
    for table in soup.select("table"):
        table.wrap(soup.new_tag("div", attrs={"class": "table-wrap"}))
    title = soup.h1.get_text()
    output.write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{CSS}</style></head><body><main>{soup}</main></body></html>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post", type=Path, default=HERE / "post.md")
    parser.add_argument("--output", type=Path, default=HERE / "post.html")
    args = parser.parse_args()
    render(args.post, args.output)
