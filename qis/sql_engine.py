import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from pathlib import Path


def get_engine(path: str = 'AWS_POSTGRES') -> Engine:
    full_file_path = Path(__file__).parent.joinpath('settings.yaml')
    with open(full_file_path) as settings:
        settings_data = yaml.load(settings, Loader=yaml.Loader)
    path = settings_data[path]
    engine = create_engine(path)
    return engine
