"""
common python structures operations
"""
import itertools
import pandas as pd
from collections.abc import Iterable
from enum import Enum
from typing import Dict, Tuple, List, Any, NamedTuple, Union, Optional


def list_intersection(list_check: Union[List[Any], pd.Index],
                      list_sample: Union[List[Any], pd.Index]
                      ) -> List[Any]:
    list_out = list(filter(lambda x: x in list_check, list_sample))
    if list_out is None:
        list_out = []
    return list_out


def list_diff(list_check: List[Any],
              list_sample: List[Any]
              ) -> List[Any]:
    list_out = list(filter(lambda x: x not in list_check, list_sample))
    if list_out is None:
        list_out = []
    return list_out


def merge_lists_unique(list1: List[Any],
                       list2: List[Any]
                       ) -> List[Any]:
    list_out = list_intersection(list_check=list2, list_sample=list1)
    list_diffs1 = list_diff(list_check=list_out, list_sample=list1)
    list_diffs2 = list_diff(list_check=list_out, list_sample=list2)
    if len(list_diffs1) > 0:
        list_out.extend(list_diffs1)
    if len(list_diffs2) > 0:
        list_out.extend(list_diffs2)
    return list_out


def assert_list_subset(large_list: List[Any],  # larger list
                       list_sample: List[Any],  # smaller list,
                       is_stop: bool = True,
                       message: str = ''
                       ) -> bool:
    outlyers = list_diff(list_check=large_list, list_sample=list_sample)
    output = True
    if len(outlyers) > 0:
        if is_stop:
            raise ValueError(f"{message}\nnot found {outlyers}")
        else:
            print(f"in assert_list_subset:\n{message}\n{outlyers}")
            output = False
    return output


def list_to_unique_and_dub(lsdata: List) -> Tuple[List, List]:
    """
    find unique and dublicates in list
    """
    unique = set()
    dublicated = []  # can count occurencies if needed
    for x in lsdata:
        if x not in unique:
            unique.add(x)
        else:
            dublicated.append(x)
    unique = list(unique)
    return unique, dublicated


def assert_list_unique(lsdata: List[str]) -> None:
    unique, duplicated = list_to_unique_and_dub(lsdata=lsdata)
    if len(duplicated) > 0:
        raise ValueError(f"list has duplicated elements = {duplicated}")


def move_item_to_first(lsdata: List[Any], item: Any) -> List[Any]:
    out_list = lsdata.copy()
    out_list.remove(item)
    out_list = [item]+out_list
    return out_list


def flatten_dict_tuples(dict_tuples: Dict[str, Union[Any, NamedTuple]]) -> Dict[str, Any]:
    data = {}
    for k, v in dict_tuples.items():
        if isinstance(v, tuple):
            data.update(v._asdict())  # this will create a dict from named tuple
        else:
            data[k] = v
    return data


def split_dict(d: Dict) -> Tuple[Dict, Dict]:
    """
    split dictionary into 2 parts
    """
    n = len(d) // 2
    i = iter(d.items())

    d1 = dict(itertools.islice(i, n))   # grab first n items
    d2 = dict(i)                        # grab the rest

    return d1, d2


def flatten(items: Iterable) -> Any:
    """
    flatten list/items from any nested iterable
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def to_flat_list(items: Iterable) -> List[Any]:
    return [item for item in flatten(items)]


def update_kwargs(kwargs: Dict[Any, Any],
                  new_kwargs: Optional[Dict[Any, Any]]
                  ) -> Dict[Any, Any]:
    """
    update kwargs with optional kwargs dicts
    """
    local_kwargs = kwargs.copy()
    if new_kwargs is not None and not len(new_kwargs) == 0:
        local_kwargs.update(new_kwargs)
    return local_kwargs


def separate_number_from_string(string: str) -> List[str]:
    """
    given 'A3' get 3
    """
    previous_character = string[0]
    groups = []
    newword = string[0]
    for x, i in enumerate(string[1:]):
        if i.isalpha() and previous_character.isalpha():
            newword += i
        elif i.isnumeric() and previous_character.isnumeric():
            newword += i
        else:
            groups.append(newword)
            newword = i
        previous_character = i
        if x == len(string) - 2:
            groups.append(newword)
            newword = ''

    return groups


class UnitTests(Enum):
    FLATTEN = 1
    LIST = 2
    MERGE = 3
    STRINGS = 4


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.FLATTEN:
        items = [[1, 2], [[3]], 4]
        flat_items = flatten(items)
        [print(item) for item in flat_items]
        print(to_flat_list(items))

    elif unit_test == UnitTests.LIST:
        rows_edge_lines = list(itertools.accumulate(10 * [5]))
        print(rows_edge_lines)

    elif unit_test == UnitTests.MERGE:
        list2 = ['EQ', 'HUI']
        list1 = ['EQ', 'BD', 'STIR', 'FX', 'Energies', 'Metals', 'Ags']
        groups = merge_lists_unique(list1=list1, list2=list2)
        print('groups1')
        print(groups)
        groups = merge_lists_unique(list1=list2, list2=list1)
        print('groups2')
        print(groups)

    elif unit_test == UnitTests.STRINGS:
        string = '123me45you0000me7+33.3'
        this = separate_number_from_string(string)
        print(this)


if __name__ == '__main__':

    unit_test = UnitTests.STRINGS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)


