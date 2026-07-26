"""
local_path uses setting.yaml to return absolute path for file/folder paths
use:
git update-index --skip-worktree optimalportfolios/settings.yaml

example usage:
import local_path as lp
lp.get_resource_path()

for linux use
import os
import sys
sys.path.append(os.getcwd())
"""

import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict

_SETTINGS_PATH = Path(__file__).parent.joinpath('settings.yaml')


@lru_cache(maxsize=1)
def get_paths() -> Dict[str, str]:
    """
    read path specs in settings.yaml; cached after first call.
    Call get_paths.cache_clear() to force a re-read.
    """
    with open(_SETTINGS_PATH) as settings:
        settings_data = yaml.safe_load(settings)
    return settings_data


def get_resource_path() -> str:
    """
    directory for bundled data resources, read from ``settings.yaml``.

    Note:
        ``settings.yaml`` ships with a placeholder path, so this returns a directory that does not
        exist until the file is edited for the machine. Pass an explicit ``local_path`` to the
        file_utils readers and writers rather than relying on it.

    Returns:
        the configured resource path

    Raises:
        KeyError: if ``settings.yaml`` has no ``RESOURCE_PATH`` entry
    """
    return get_paths()['RESOURCE_PATH']


def get_output_path() -> str:
    """
    output path from settings.yaml
    """
    return get_paths()['OUTPUT_PATH']