"""Output destinations for runnable examples, independent of private settings.yaml paths."""
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Union


def get_output_dir(output_path: Optional[Union[str, Path]] = None) -> str:
    """Create an explicit destination or a fresh directory in the system temporary area.

    Args:
        output_path: Destination supplied by the reader; missing parents are created.

    Returns:
        Absolute destination, also printed so readers can find generated reports.
    """
    path = (Path(output_path).expanduser().resolve() if output_path is not None
            else Path(tempfile.mkdtemp(prefix='qis-examples-')))
    path.mkdir(parents=True, exist_ok=True)
    print(f'Example output: {path}')
    return str(path)


def parse_output_dir() -> Optional[str]:
    """Read the optional --output-dir argument of a standalone example."""
    parser = argparse.ArgumentParser(description='Generate a QIS example report.')
    parser.add_argument(
        '--output-dir', help='Create reports here (default: a temporary directory).'
    )
    return parser.parse_args().output_dir
