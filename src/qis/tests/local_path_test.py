"""One YAML-backed path contract for examples and library callers."""

from pathlib import Path

import pytest
import yaml

from qis import local_path


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Never read private configuration or create directories in the real user home."""
    user_home = tmp_path / 'home'
    user_home.mkdir()
    monkeypatch.setenv('HOME', str(user_home))
    monkeypatch.setenv('USERPROFILE', str(user_home))
    monkeypatch.delenv('QIS_SETTINGS_PATH', raising=False)
    local_path.get_paths.cache_clear()
    yield
    local_path.get_paths.cache_clear()


def write_settings(folder, **values):
    """Write only synthetic paths into a pytest temporary directory."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'private.yaml'
    path.write_text(yaml.safe_dump(values), encoding='utf-8')
    return path


def test_external_yaml_relative_paths_and_creation(tmp_path, monkeypatch):
    config = write_settings(tmp_path / 'config', OUTPUT_PATH='../reports', RESOURCE_PATH='data')
    (config.parent / 'data').mkdir()
    monkeypatch.setenv('QIS_SETTINGS_PATH', str(config))
    monkeypatch.chdir(tmp_path)
    assert local_path.get_output_path() == str(tmp_path / 'reports')
    assert (tmp_path / 'reports').is_dir()
    assert local_path.get_resource_path() == str(config.parent / 'data')
    # Partial private YAML retains unrelated packaged keys for legacy get_paths callers.
    assert 'UNIVERSE_PATH' in local_path.get_paths()


def test_explicit_directory_bypasses_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv('QIS_SETTINGS_PATH', str(tmp_path / 'missing.yaml'))
    monkeypatch.chdir(tmp_path)
    assert local_path.get_output_path('explicit/output') == str(tmp_path / 'explicit/output')
    assert local_path.get_resource_path(tmp_path) == str(tmp_path)


def test_explicit_config_beats_environment(tmp_path, monkeypatch):
    config = write_settings(tmp_path / 'config', OUTPUT_PATH='reports')
    monkeypatch.setenv('QIS_SETTINGS_PATH', str(tmp_path / 'missing.yaml'))
    assert local_path.get_output_path(settings_path=config) == str(config.parent / 'reports')


def test_environment_switch_and_explicit_cache_clear(tmp_path, monkeypatch):
    first = write_settings(tmp_path / 'first', OUTPUT_PATH='one')
    second = write_settings(tmp_path / 'second', OUTPUT_PATH='two')
    monkeypatch.setenv('QIS_SETTINGS_PATH', str(first))
    assert Path(local_path.get_output_path()).name == 'one'
    monkeypatch.setenv('QIS_SETTINGS_PATH', str(second))
    assert Path(local_path.get_output_path()).name == 'two'
    second.write_text('OUTPUT_PATH: three\n', encoding='utf-8')
    local_path.get_paths.cache_clear()
    assert Path(local_path.get_output_path()).name == 'three'


def test_get_paths_returns_independent_mapping(tmp_path):
    config = write_settings(tmp_path, OUTPUT_PATH='reports')
    values = local_path.get_paths(settings_path=config)
    values['OUTPUT_PATH'] = 'changed'
    assert Path(local_path.get_paths(settings_path=config)['OUTPUT_PATH']).name == 'reports'


def test_expands_environment_and_home(tmp_path, monkeypatch):
    monkeypatch.setenv('QIS_TEST_ROOT', str(tmp_path))
    config = write_settings(tmp_path, OUTPUT_PATH='$QIS_TEST_ROOT/reports', RESOURCE_PATH='~')
    assert local_path.get_output_path(settings_path=config) == str(tmp_path / 'reports')
    assert local_path.get_resource_path(settings_path=config) == str(Path.home())


@pytest.mark.parametrize('value', ['', '  ', 'C:\\Users\\...\\', None, 42, ['folder']])
def test_invalid_path_is_actionable_without_making_directory(tmp_path, value):
    config = write_settings(tmp_path, OUTPUT_PATH=value)
    with pytest.raises(ValueError, match='OUTPUT_PATH.*QIS_SETTINGS_PATH'):
        local_path.get_output_path(settings_path=config)


def test_packaged_defaults_resolve_without_creating_directories(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = Path.home() / '.qis'
    paths = local_path.get_paths()
    assert paths['RESOURCE_PATH'] == str(root / 'resources')
    assert paths['LOCAL_RESOURCE_PATH'] == str(root / 'resources')
    assert paths['UNIVERSE_PATH'] == str(root / 'resources/universes')
    assert paths['OUTPUT_PATH'] == str(root / 'outputs')
    assert not root.exists()


def test_default_output_is_created_on_use():
    output = Path.home() / '.qis/outputs'
    assert local_path.get_output_path() == str(output)
    assert output.is_dir()


def test_default_resource_directory_is_only_created_explicitly():
    resources = Path.home() / '.qis/resources'
    with pytest.raises(FileNotFoundError, match='RESOURCE_PATH'):
        local_path.get_resource_path()
    assert not resources.exists()
    assert local_path.get_resource_path(create=True) == str(resources)
    assert list(resources.iterdir()) == []


def test_partial_private_yaml_retains_portable_defaults(tmp_path):
    config = write_settings(tmp_path / 'config', OUTPUT_PATH='reports')
    paths = local_path.get_paths(settings_path=config)
    assert paths['OUTPUT_PATH'] == str(config.parent / 'reports')
    assert paths['RESOURCE_PATH'] == str(Path.home() / '.qis/resources')
    assert not (Path.home() / '.qis').exists()


def test_missing_input_is_not_created(tmp_path):
    folder = tmp_path / 'missing-input'
    with pytest.raises(FileNotFoundError, match='RESOURCE_PATH'):
        local_path.get_resource_path(folder)
    assert not folder.exists()
    assert local_path.get_resource_path(folder, create=True) == str(folder)


def test_output_can_be_resolved_without_creation(tmp_path):
    folder = tmp_path / 'fresh-report'
    assert local_path.get_output_path(folder, create=False) == str(folder)
    assert not folder.exists()


def test_existing_file_is_not_a_directory(tmp_path):
    file = tmp_path / 'file.txt'
    file.write_text('keep me', encoding='utf-8')
    for getter in (local_path.get_output_path, local_path.get_resource_path):
        with pytest.raises(NotADirectoryError):
            getter(file)
    assert file.read_text(encoding='utf-8') == 'keep me'


@pytest.mark.parametrize('content', ['- item\n', 'null\n', 'secret: [private-token\n'])
def test_invalid_yaml_does_not_echo_private_values(tmp_path, content):
    config = tmp_path / 'private.yaml'
    config.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match='YAML') as error:
        local_path.get_paths(settings_path=config)
    assert 'private-token' not in str(error.value)


def test_missing_explicit_config_does_not_fall_back(tmp_path):
    with pytest.raises(FileNotFoundError, match='QIS_SETTINGS_PATH'):
        local_path.get_output_path(settings_path=tmp_path / 'missing.yaml')
