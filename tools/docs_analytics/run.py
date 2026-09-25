"""Build and validate complete offline documentation bundles; never publish into docs/images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'tools/docs_analytics/manifest.json'


def digest(path: Path) -> str:
    """Return the SHA-256 content fingerprint of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write stable, strict JSON with no NaN or Infinity values."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')


def relative_path(value: str) -> Path:
    """Reject absolute paths, traversal, Windows drive paths, and nonportable separators."""
    path = Path(value)
    if not value or '\\' in value or ':' in value or path.anchor:
        raise ValueError(f'Expected portable relative path: {value!r}')
    if any(part in ('.', '..') for part in value.split('/')):
        raise ValueError(f'Unsafe relative path: {value!r}')
    return path


def image_references(text: str) -> list[str]:
    """Find Markdown, reference-style, HTML and MyST images, excluding code examples."""
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    visible = []
    references = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^\s*(\x60{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
            continue
        if match:
            fence = match[1]
            directive = re.match(r'\{(?:image|figure)\}\s+(\S+)', match[2])
            if directive:
                references.append(directive[1])
            continue
        colon_directive = re.match(r'^\s*:{3,}\{(?:image|figure)\}\s+(\S+)', line)
        if colon_directive:
            references.append(colon_directive[1])
        visible.append(line)
    prose = '\n'.join(visible)
    references += re.findall(r'!\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+[^)]*)?\)', prose)
    class ImageParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == 'img':
                src = dict(attrs).get('src')
                if src:
                    references.append(src)

    ImageParser().feed(prose)
    definitions = {
        key.strip().casefold(): url for key, url in re.findall(
            r'^\s*\[([^\]]+)\]:\s*<?([^\s>]+)>?', prose, flags=re.M)
    }
    for match in re.finditer(r'!\[([^\]]+)\](?:\[([^\]]*)\])?(?!\()', prose):
        label = (match[2] or match[1]).strip().casefold()
        if label not in definitions:
            raise ValueError(f'Unresolved image reference: {label}')
        references.append(definitions[label])
    return references


def load_manifest(path: Path = MANIFEST, root: Path = ROOT) -> dict:
    """Validate the coverage ledger before importing any analytics producer."""
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported analytics manifest schema')
    producers = manifest['producers']
    ids, paths = set(), set()
    if not producers or not manifest['assets']:
        raise ValueError('The manifest must contain producers and assets')
    for name, spec in producers.items():
        if spec['module'] != f'tools.docs_analytics.{name}' or spec['callable'] != 'produce':
            raise ValueError(f'Unrecognised producer entry point: {name}')
        if not re.fullmatch(r'[a-z_]+', name):
            raise ValueError(f'Unsafe producer name: {name}')
        for field in ('fixture', 'parameters', 'conventions', 'tables', 'checks'):
            if not spec[field]:
                raise ValueError(f'Missing producer field: {name}.{field}')
        for field in ('tables', 'checks'):
            entries = spec[field]
            if len(entries) != len(set(entries)):
                raise ValueError(f'Duplicate {field}: {name}')
            if any(not re.fullmatch(r'[a-z_]+', item) for item in entries):
                raise ValueError(f'Unsafe {field}: {name}')
    for asset in manifest['assets']:
        if asset['producer'] not in producers:
            raise ValueError(f'Unknown producer: {asset["producer"]}')
        path_value = asset['path']
        path_obj = relative_path(path_value)
        doc = relative_path(asset['document'])
        if path_obj.parent != Path('docs/images') or path_obj.suffix != '.png':
            raise ValueError(f'Image outside the preview allowlist directory: {path_value}')
        if not (root / doc).is_file():
            raise ValueError(f'Missing consumer: {doc}')
        if asset['id'] in ids or path_value in paths:
            raise ValueError(f'Duplicate image identity/path: {path_value}')
        ids.add(asset['id'])
        paths.add(path_value)
    if set(producers) != {asset['producer'] for asset in manifest['assets']}:
        raise ValueError('Unused producer in manifest')
    check_coverage(manifest, root)
    return manifest


def check_coverage(manifest: dict, root: Path = ROOT) -> None:
    """Require a producer or explicit non-analytics classification for every displayed image."""
    expected = {(item['document'], item['path']) for item in manifest['assets']}
    expected |= {(item['document'], item['url']) for item in manifest['non_analytics']}
    observed = set()
    for doc in (root / 'docs').rglob('*.md'):
        rel_doc = doc.relative_to(root)
        if any(part.startswith(('_', '.')) for part in rel_doc.parts[1:]):
            continue  # build-time mirrors and generated API pages have separate source owners
        for url in image_references(doc.read_text(encoding='utf-8')):
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                target = url
            else:
                resolved = (doc.parent / unquote(parsed.path)).resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError(f'Image escapes source tree: {rel_doc}: {url}')
                target = resolved.relative_to(root.resolve()).as_posix()
            observed.add((rel_doc.as_posix(), target))
    if observed != expected:
        raise ValueError(f'Image coverage mismatch: unregistered={sorted(observed - expected)}, '
                         f'unused={sorted(expected - observed)}')


def output_boundary(output: Path, root: Path = ROOT, *, new: bool = True) -> Path:
    """Require C-local task output and reject overwrites, repository output and link escapes."""
    local_setting = os.environ.get('AGENT_LOCAL_ROOT')
    if not local_setting:
        raise ValueError('Set AGENT_LOCAL_ROOT to the C-local generated-output root first')
    local = Path(local_setting).absolute()
    candidate = output.absolute()
    for item in (local, candidate):
        for parent in (item, *item.parents):
            if parent.is_symlink() or getattr(parent, 'is_junction', lambda: False)():
                raise ValueError(f'Linked output path is not allowed: {parent}')
        if any(part.lower().startswith('onedrive') for part in item.parts):
            raise ValueError('Generated output must be outside OneDrive')
    local, candidate = local.resolve(), candidate.resolve()
    if os.name == 'nt' and local.drive.upper() != 'C:':
        raise ValueError('AGENT_LOCAL_ROOT must be on C:')
    if candidate == local or not candidate.is_relative_to(local):
        raise ValueError('Output must be in a task directory below AGENT_LOCAL_ROOT')
    if candidate.is_relative_to(root.resolve()):
        raise ValueError('Output must be outside the source tree')
    if new and candidate.exists():
        raise ValueError(f'Refusing to overwrite existing output: {candidate}')
    return candidate


def source_fingerprint(root: Path = ROOT) -> dict:
    """Identify the effective source independently of Git or an installed version label."""
    paths = set((root / 'src/qis').rglob('*.py'))
    paths.update((root / 'tools/docs_analytics').glob('*.py'))
    paths.update((root / 'tools/docs_analytics').glob('*.json'))
    paths.add(root / 'examples/portfolios/model_layer_attribution_simulated.py')
    paths.add(root / 'examples/models/bootstrap_convention.py')
    paths.add(root / 'pyproject.toml')
    for pattern in ('*lock*', 'requirements*.txt'):
        paths.update(root.glob(pattern))
    files = {path.relative_to(root).as_posix(): digest(path)
             for path in sorted(paths) if path.is_file()}
    content = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'sha256': content, 'files': files,
            'identity': 'Effective source content, including dirty export files'}


@contextmanager
def offline():
    """Forbid socket connections and name resolution for the entire generation step."""
    def denied(*args, **kwargs):
        raise RuntimeError('Documentation analytics must run offline')
    patches = [(socket.socket, 'connect'), (socket.socket, 'connect_ex'),
               (socket, 'create_connection'), (socket, 'getaddrinfo')]
    originals = [(obj, name, getattr(obj, name)) for obj, name in patches]
    try:
        for obj, name, _ in originals:
            setattr(obj, name, denied)
        yield
    finally:
        for obj, name, value in originals:
            setattr(obj, name, value)


def expected_files(manifest: dict) -> set[str]:
    """Return all required bundle paths, excluding the provenance file itself."""
    files = {f'images/{Path(item["path"]).name}' for item in manifest['assets']}
    for name, spec in manifest['producers'].items():
        files.update(f'tables/{name}/{table}.csv' for table in spec['tables'])
    return files


def file_record(path: Path) -> dict:
    """Read an output back from disk and record its content and shape."""
    record = {'sha256': digest(path), 'bytes': path.stat().st_size}
    if not record['bytes']:
        raise ValueError(f'Empty output: {path}')
    if path.suffix == '.png':
        from PIL import Image
        with Image.open(path) as picture:
            picture.verify()
        with Image.open(path) as picture:
            picture.load()
            record['width'], record['height'] = picture.size
            if min(picture.size) < 100 or picture.convert('L').getextrema()[0] == (
                    picture.convert('L').getextrema()[1]):
                raise ValueError(f'Empty or undersized figure: {path}')
    else:
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.reader(stream))
        if len(rows) < 2 or not rows[0]:
            raise ValueError(f'Missing table data: {path}')
        record['rows_including_header'] = len(rows)
        record['columns'] = len(rows[0])
    return record


def validate_bundle(bundle: Path, manifest: dict, root: Path = ROOT) -> dict:
    """Reject missing, altered, unregistered or source-stale outputs."""
    bundle = output_boundary(bundle, root, new=False)
    record = json.loads((bundle / 'analytics_manifest.json').read_text(encoding='utf-8'))
    if record.get('schema_version') != 1 or record.get('status') != 'complete':
        raise ValueError('Bundle is not marked complete')
    if record['manifest'] != manifest:
        raise ValueError('Bundle manifest differs from the current coverage ledger')
    if record['source'] != source_fingerprint(root):
        raise ValueError('Bundle source fingerprint differs from the current source')
    wanted = expected_files(manifest)
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob('*') if path.is_file()}
    if actual != wanted | {'analytics_manifest.json'} or set(record['outputs']) != wanted:
        raise ValueError('Incomplete bundle or unregistered output')
    if set(record['producers']) != set(manifest['producers']):
        raise ValueError('Missing producer result')
    for name, spec in manifest['producers'].items():
        checks = record['producers'][name]['checks']
        if set(checks) != set(spec['checks']) or not all(
                value is True for value in checks.values()):
            raise ValueError(f'Missing or failed numerical checks: {name}')
    for rel in sorted(wanted):
        path = bundle / relative_path(rel)
        if not path.resolve().is_relative_to(bundle) or path.is_symlink():
            raise ValueError(f'Output link escapes bundle: {rel}')
        if file_record(path) != record['outputs'][rel]:
            raise ValueError(f'Output content/shape mismatch: {rel}')
    return record


def generate(output: Path, manifest: dict, root: Path = ROOT) -> Path:
    """Build in a staging directory; expose a complete bundle only after validation."""
    output = output_boundary(output, root)
    check_coverage(manifest, root)
    versions = {name: importlib.metadata.version(name) for name in manifest['dependencies']}
    source = source_fingerprint(root)
    staging = output.with_name(f'.{output.name}-building-{uuid.uuid4().hex}')
    staging.mkdir(parents=True, exist_ok=False)
    record = {'schema_version': 1, 'status': 'complete', 'manifest': manifest,
              'source': source, 'python': platform.python_version(), 'dependencies': versions,
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'producers': {}, 'outputs': {}}
    try:
        with offline():
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            import qis
            from tools.docs_analytics.style import render_context, save_figure

            imported = Path(qis.__file__).resolve()
            if not imported.is_relative_to((root / 'src/qis').resolve()):
                raise ValueError(f'qis imported outside this source export: {imported}')
            record['qis_import_path'] = str(imported)
            project = (root / 'pyproject.toml').read_text(encoding='utf-8')
            project = project.split('[project]', 1)[1].split('\n[', 1)[0]
            version = re.search(r'^version\s*=\s*["\x27]([^"\x27]+)', project, flags=re.M)
            if version is None:
                raise ValueError('Missing static qis source version in pyproject.toml')
            record['qis_source_version'] = version[1]
            record['version_note'] = (
                'dependencies.qis is installed distribution metadata; qis_source_version and '
                'the content fingerprint identify the imported source export')
            record['backend'] = matplotlib.get_backend()
            try:
                with render_context():
                    for name, spec in manifest['producers'].items():
                        print(f'Generating {name} ...', flush=True)
                        module = importlib.import_module(spec['module'])
                        producer = getattr(module, spec['callable'])
                        result = producer(spec)
                        figures = result.pop('figures')
                        tables = result.pop('tables')
                        wanted = {Path(asset['path']).name for asset in manifest['assets']
                                  if asset['producer'] == name}
                        if set(figures) != wanted or set(tables) != set(spec['tables']):
                            raise ValueError(f'Producer output differs from manifest: {name}')
                        if set(result['checks']) != set(spec['checks']) or not all(
                                value is True for value in result['checks'].values()):
                            raise ValueError(f'Producer checks failed: {name}')
                        for filename, fig in figures.items():
                            dest = staging / 'images' / filename
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            save_figure(fig, dest)
                            plt.close(fig)
                        for label, table in tables.items():
                            dest = staging / 'tables' / name / f'{label}.csv'
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            table.to_csv(dest, float_format='%.17g', lineterminator='\n')
                        record['producers'][name] = result
            finally:
                plt.close('all')
        record['outputs'] = {rel: file_record(staging / rel)
                             for rel in sorted(expected_files(manifest))}
        write_json(staging / 'analytics_manifest.json', record)
        validate_bundle(staging, manifest, root)
        staging.rename(output)
        return output
    except BaseException as error:
        (staging / 'FAILED.txt').write_text(f'{type(error).__name__}: {error}\n', encoding='utf-8')
        print(f'Incomplete diagnostic output retained at {staging}', file=sys.stderr)
        raise


def main(argv: list[str] | None = None) -> int:
    """Run one of the manifest inventory, complete generation, or validation commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true')
    action.add_argument('--all', action='store_true')
    action.add_argument('--validate-bundle', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.all != (args.output_dir is not None):
        parser.error('--output-dir is required only with --all')
    try:
        manifest = load_manifest()
        if args.list:
            for asset in manifest['assets']:
                print(f'{asset["id"]}: {asset["producer"]} -> {asset["path"]}')
        elif args.all:
            print(f'Complete bundle: {generate(args.output_dir, manifest)}')
        else:
            result = validate_bundle(args.validate_bundle, manifest)
            print(f'Valid complete bundle: {len(result["outputs"])} outputs')
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as error:
        print(f'Analytics error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
