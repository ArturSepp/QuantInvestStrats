"""Publish a reviewed complete bundle, or verify the recorded documentation previews."""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tools.docs_analytics.run import (
    file_record, load_manifest, output_boundary, relative_path, validate_bundle, write_json,
)


def _destination(root, relative):
    """Reject links and path escapes before writing an allowlisted deliverable."""
    path = root / relative_path(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Publication path escapes repository: {relative}')
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
            raise ValueError(f'Publication path uses a link: {item}')
        if item == root:
            break
    if path.exists() and not path.is_file():
        raise ValueError(f'Publication destination is not a file: {relative}')
    return path


def verify_published(root: Path, manifest: dict | None = None) -> dict:
    """Check published image bytes against provenance; source remains a dated snapshot."""
    root = root.absolute()
    manifest = manifest or load_manifest(root / 'tools/docs_analytics/manifest.json', root)
    path = _destination(root, 'docs/images/analytics_manifest.json')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema_version') != 1 or record.get('status') != 'complete':
        raise ValueError('Published provenance is not a complete bundle record')
    if record['manifest'] != manifest:
        raise ValueError('Published manifest differs from the current coverage ledger')
    paths = sorted(asset['path'] for asset in manifest['assets'])
    if record.get('publication', {}).get('paths') != paths:
        raise ValueError('Published preview set is incomplete')
    for relative in paths:
        output = f'images/{Path(relative).name}'
        if file_record(_destination(root, relative)) != record['outputs'][output]:
            raise ValueError(f'Published image does not match provenance: {relative}')
    return record


def publish_bundle(bundle: Path, root: Path, manifest: dict | None = None) -> dict:
    """Validate before any write; preserve a C-local backup and roll back on failure."""
    root = root.absolute()
    manifest = manifest or load_manifest(root / 'tools/docs_analytics/manifest.json', root)
    record = validate_bundle(bundle, manifest, root)
    if record['producers']['gallery'].get('presentation') == 'complete reports':
        raise ValueError('Publish the reviewed focused previews, not full-report output')
    paths = sorted(asset['path'] for asset in manifest['assets'])
    record = dict(record, publication={
        'published_at_utc': datetime.now(timezone.utc).isoformat(),
        'paths': paths,
        'note': 'Only allowlisted PNGs are published. Supporting CSVs remain in the build bundle.',
    })
    writes = {relative: (bundle / 'images' / Path(relative).name).read_bytes()
              for relative in paths}
    provenance = 'docs/images/analytics_manifest.json'
    writes[provenance] = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
                          + '\n').encode('utf-8')
    destinations = {relative: _destination(root, relative) for relative in writes}
    backup = output_boundary(bundle.parent / f'publication-backup-{uuid.uuid4().hex}', root)
    backup.mkdir()
    pending = backup / 'pending'
    pending.mkdir()
    previous = {}
    for relative, destination in destinations.items():
        previous[relative] = destination.read_bytes() if destination.exists() else None
        if previous[relative] is not None:
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_bytes(previous[relative])
        (pending / Path(relative).name).write_bytes(writes[relative])
    write_json(backup / 'backup.json', {
        'target': str(root), 'previously_present': [
            relative for relative, content in previous.items() if content is not None],
    })
    print(f'Publication backup: {backup}', flush=True)
    replaced = []
    try:
        for relative, destination in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pending / Path(relative).name, destination)
            replaced.append(relative)
        verify_published(root, manifest)
    except BaseException:
        for relative in reversed(replaced):
            destination = destinations[relative]
            if previous[relative] is None:
                destination.unlink()
            else:
                destination.write_bytes(previous[relative])
        raise
    finally:
        # Pending files are C-local scratch; keep the backup and any failed pending files.
        if not any(pending.iterdir()):
            pending.rmdir()
    return record


def main(argv=None):
    """Run publication only when explicitly requested with a complete reviewed bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--publish-bundle', type=Path)
    action.add_argument('--verify', action='store_true')
    parser.add_argument('--repo', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify:
            result = verify_published(args.repo)
        else:
            result = publish_bundle(args.publish_bundle, args.repo)
        print(f'Verified {len(result["publication"]["paths"])} published previews')
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Publication error: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
