import platform
from enum import Enum
from qis.file_utils import (get_paths,
                            OUTPUT_PATH,
                            get_all_folder_files,
                            join_file_name_parts,
                            FileTypes,
                            get_local_file_path)



class Locals(Enum):
    LOCAL_PATHS = 1
    FOLDER_FILES = 2
    NAMES = 3
    DATA_FILE = 4
    UNIVERSE = 5


def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local == Locals.LOCAL_PATHS:
        print(get_paths())
        print(platform.system())
        print(OUTPUT_PATH)

    elif local == Locals.FOLDER_FILES:
        get_all_folder_files(folder_path="C://")

    elif local == Locals.NAMES:
        file_name = join_file_name_parts(['head', 'tails'])
        print(file_name)

    elif local == Locals.DATA_FILE:
        file_path = get_local_file_path(file_name='ETH', file_type=FileTypes.CSV)
        print(file_path)

    elif local == Locals.UNIVERSE:
        file_path = get_local_file_path(file_name='ETH',
                                        folder_name='bitmex',
                                        key='1d',
                                        file_type=FileTypes.CSV)
        print(file_path)


if __name__ == '__main__':

    run_local(local=Locals.NAMES)
