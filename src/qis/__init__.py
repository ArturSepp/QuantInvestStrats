
import qis.local_path

from qis.file_utils import (
    FileTypes,
    update_df_in_csv,
    append_df_to_feather,
    save_figs_to_pdf,
    get_all_folder_files,
    get_local_file_path,
    get_output_path,
    get_paths,
    get_pdf_path,
    get_resource_path,
    join_file_name_parts,
    load_df_dict_from_csv,
    load_df_dict_from_excel,
    load_df_dict_from_feather,
    load_df_dict_from_parquet,
    load_df_from_csv,
    load_df_from_excel,
    load_df_from_feather,
    load_df_from_parquet,
    save_df_dict_to_csv,
    save_df_dict_to_excel,
    save_df_dict_to_feather,
    save_df_dict_to_parquet,
    save_df_to_csv,
    save_df_to_excel,
    save_df_to_feather,
    save_df_to_parquet,
    save_fig,
    save_figs,
    timer
)

from qis.utils.__init__ import *

from qis.perfstats.__init__ import *

from qis.plots.__init__ import *

from qis.models.__init__ import *

from qis.portfolio.__init__ import *

from qis.market_data.__init__ import *


# the public surface, computed once here rather than inferred from ``dir(qis)`` later.
#
# ``dir(qis)`` is not stable: importing a submodule binds its name on the package, so
# ``import qis.api`` makes ``dir(qis)`` one name longer than it was. Anything that counts or
# checks the public surface has to be counting the same set every time, so the set is fixed at
# the end of this module, where the wildcard re-exports above have finished and nothing else has
# been imported yet. The nine subpackage names bound by those imports are included, because
# ``from qis import *`` has always bound them and removing them would be a silent break.
#
# ``qis.api.PUBLIC_API`` records the same names in a diffable literal, and
# ``src/qis/tests/test_core_api.py`` fails when the two disagree.
__all__ = [_name for _name in dir() if not _name.startswith('_')]
