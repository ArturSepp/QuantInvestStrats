"""Resolve resource/output directories from portable defaults or caller overrides.

Defaults are ``~/.qis/resources`` and ``~/.qis/outputs``; private YAML is optional.
Set ``QIS_SETTINGS_PATH`` to a YAML file outside the checkout/package to override
individual defaults. Never put private paths or credentials in the distributed YAML.
Relative YAML paths are relative to the YAML file, while explicit
directory arguments are relative to the working directory. Both expand ``~`` and
environment variables. See ``qis/README.md`` for configuration and runner examples.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

_SETTINGS_PATH = Path(__file__).parent.joinpath('settings.yaml')
PathLike = Union[str, Path]


def _settings_path(settings_path: Optional[PathLike]) -> Path:
    """Select an explicit configuration, the environment override, or the template."""
    value = settings_path if settings_path is not None else os.environ.get('QIS_SETTINGS_PATH')
    if value is None:
        return _SETTINGS_PATH
    if not str(value).strip():
        raise ValueError('QIS_SETTINGS_PATH must name a YAML file, not an empty string')
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


@lru_cache(maxsize=8)
def _read_settings(path: Path) -> dict:
    """Cache YAML by file, without exposing private values in parser diagnostics."""
    try:
        with path.open(encoding='utf-8-sig') as stream:
            values = yaml.safe_load(stream)
    except FileNotFoundError:
        raise FileNotFoundError(
            'QIS_SETTINGS_PATH/settings_path YAML file does not exist') from None
    except yaml.YAMLError:
        raise ValueError('Invalid YAML in QIS_SETTINGS_PATH/settings_path') from None
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise ValueError('QIS_SETTINGS_PATH/settings_path YAML must contain a mapping of keys')
    resolved = dict(values)
    for key, value in values.items():
        if key.endswith('_PATH') and isinstance(value, str) and value.strip():
            if '...' not in value.replace('\\', '/').split('/'):
                folder = Path(os.path.expandvars(value)).expanduser()
                resolved[key] = str((folder if folder.is_absolute() else path.parent / folder)
                                    .resolve())
    return resolved


def get_paths(settings_path: Optional[PathLike] = None) -> Dict[str, str]:
    """Read YAML configuration, with private entries overriding packaged defaults.

    Args:
        settings_path: Explicit YAML file; otherwise use ``QIS_SETTINGS_PATH`` or the
            distributed template. Cached by file; call ``get_paths.cache_clear()``
            after editing a file. Changing the selected file takes effect immediately.

    Returns:
        A fresh dictionary. Directory entries are absolute, except unresolved placeholders.
        Non-path settings (for example database URLs) are left unchanged.

    Raises:
        FileNotFoundError: The selected YAML file is missing; no silent fallback occurs.
        ValueError: YAML is malformed or does not contain a mapping.
    """
    selected = _settings_path(settings_path)
    values = dict(_read_settings(_SETTINGS_PATH))
    if selected != _SETTINGS_PATH:
        values.update(_read_settings(selected))
    return values


# Preserve the established cache-clear hook without caching the environment selection.
get_paths.cache_clear = _read_settings.cache_clear


def _directory(key: str, local_path: Optional[PathLike], settings_path: Optional[PathLike],
               create: bool, must_exist: bool) -> str:
    """Apply common validation and creation after explicit-argument precedence."""
    value = local_path if local_path is not None else get_paths(settings_path).get(key)
    if (not isinstance(value, (str, Path)) or not str(value).strip()
            or '...' in str(value).replace('\\', '/').split('/')):
        raise ValueError(
            f'{key} is missing, empty or a placeholder; configure QIS_SETTINGS_PATH '
            'with private YAML or pass an explicit local_path directory')
    folder = Path(os.path.expandvars(str(value))).expanduser().resolve()
    if folder.exists() and not folder.is_dir():
        raise NotADirectoryError(f'{key} must refer to a directory')
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    elif must_exist and not folder.is_dir():
        raise FileNotFoundError(f'{key} input directory does not exist: {folder}')
    return str(folder)


def get_resource_path(local_path: Optional[PathLike] = None, *,
                      settings_path: Optional[PathLike] = None, create: bool = False) -> str:
    """Resolve the input/resource directory, without creating missing inputs by default.

    Args:
        local_path: Explicit directory override; bypasses YAML resolution.
        settings_path: Explicit private YAML file; otherwise use ``QIS_SETTINGS_PATH``.
        create: Create missing parents for a deliberate data-download/cache destination.

    Returns:
        Absolute resource directory as a string.

    Raises:
        ValueError: RESOURCE_PATH is unconfigured, empty or a placeholder.
        FileNotFoundError: The configuration or input directory is missing.
        NotADirectoryError: The resolved path is an existing file.
    """
    return _directory('RESOURCE_PATH', local_path, settings_path, create, must_exist=True)


def get_output_path(local_path: Optional[PathLike] = None, *,
                    settings_path: Optional[PathLike] = None, create: bool = True) -> str:
    """Resolve the output directory through YAML, creating missing parents by default.

    Args:
        local_path: Explicit directory override; bypasses YAML resolution.
        settings_path: Explicit private YAML file; otherwise use ``QIS_SETTINGS_PATH``.
        create: Create the directory. Set False for a report writer requiring a fresh path.

    Returns:
        Absolute output directory as a string; never a temporary-directory fallback.

    Raises:
        ValueError: OUTPUT_PATH is unconfigured, empty or a placeholder.
        FileNotFoundError: The selected configuration file is missing.
        NotADirectoryError: The resolved path is an existing file.
    """
    return _directory('OUTPUT_PATH', local_path, settings_path, create, must_exist=False)
