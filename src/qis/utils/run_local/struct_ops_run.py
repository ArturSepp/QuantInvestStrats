
import itertools
from enum import Enum
from qis.utils.struct_ops import (to_flat_list,
                                  flatten,
                                  separate_number_from_string,
                                  list_intersection,
                                  merge_lists_unique,
                                  list_diff)

class Locals(Enum):
    FLATTEN = 1
    LIST = 2
    LIST_INTERSECTION = 3
    MERGE = 4
    LIST_DIFF = 5
    STRINGS = 6


def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local == Locals.FLATTEN:
        items = [[1, 2], [[3]], 4]
        flat_items = flatten(items)
        [print(item) for item in flat_items]
        print(to_flat_list(items))

    elif local == Locals.LIST:
        rows_edge_lines = list(itertools.accumulate(10 * [5]))
        print(rows_edge_lines)

    elif local == Locals.LIST_INTERSECTION:
        list2 = ['EQ', 'HUI', 'Metals']
        list1 = ['EQ', 'BD', 'STIR', 'FX', 'Energies', 'Metals', 'Ags']
        groups = list_intersection(list_check=list1, list_sample=list2)
        print('groups1')
        print(groups)
        groups = list_intersection(list_check=list2, list_sample=list1)
        print('groups2')
        print(groups)

    elif local == Locals.MERGE:
        list2 = ['EQ', 'HUI']
        list1 = ['EQ', 'BD', 'STIR', 'FX', 'Energies', 'Metals', 'Ags']
        groups = merge_lists_unique(list1=list1, list2=list2)
        print('groups1')
        print(groups)
        groups = merge_lists_unique(list1=list2, list2=list1)
        print('groups2')
        print(groups)

    elif local == Locals.LIST_DIFF:
        list2 = ['EQ', 'HUI']
        list1 = ['EQ', 'BD', 'STIR', 'FX', 'Energies', 'Metals', 'Ags']
        this = list_diff(list_check=list1, list_sample=list2)
        print(this)
        this = list_diff(list_check=list2, list_sample=list1)
        print(this)

    elif local == Locals.STRINGS:
        string = '123me45you0000me7+33.3'
        this = separate_number_from_string(string)
        print(this)


if __name__ == '__main__':

    run_local(local=Locals.LIST_INTERSECTION)
