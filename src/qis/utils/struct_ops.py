"""
list, dict and kwargs helpers used across qis: membership, ordering and merging.

Order is why these exist rather than ``set`` operations: ``list_intersection`` and ``list_diff``
return the elements of ``list_sample`` in the order they appear there, and
``merge_lists_unique`` puts the intersection first, then what is unique to ``list1``, then what
is unique to ``list2``. That ordering is what fixes group and column order downstream.

``assert_list_subset`` raises ``ValueError`` naming the missing elements when ``list_sample`` is
not contained in ``large_list``, or prints them and returns False when ``is_stop=False``.
``update_kwargs`` merges two keyword dictionaries with the second winning and neither input
mutated - the copy is the point, since it is how one defaults dict is passed down to several
panels without one panel's override leaking into the next.
"""
import itertools
import pandas as pd
from collections.abc import Iterable
from typing import Dict, Tuple, List, Any, NamedTuple, Union, Optional


def list_intersection(list_check: Union[List[Any], pd.Index],
                      list_sample: Union[List[Any], pd.Index]
                      ) -> List[Any]:
    """Find the intersection of elements between two lists or pandas Index objects.

    Returns elements from list_sample that are also present in list_check,
    maintaining the order from list_sample.

    Args:
        list_check (Union[List[Any], pd.Index]): The reference list/Index to check
            against for membership.
        list_sample (Union[List[Any], pd.Index]): The list/Index to filter based
            on elements present in list_check.

    Returns:
        List[Any]: A list containing elements from list_sample that exist in
            list_check. Returns empty list if no intersection found.

    Example:
        >>> list_intersection([1, 2, 3], [2, 3, 4, 5])
        [2, 3]
    """
    list_out = list(filter(lambda x: x in list_check, list_sample))
    if list_out is None:
        list_out = []
    return list_out


def list_diff(list_check: Union[List[Any], pd.Index],
              list_sample: Union[List[Any], pd.Index]
              ) -> List[Any]:
    """Find elements in list_sample that are not present in list_check.

    Returns the difference between two lists, maintaining the order from
    list_sample for elements not found in list_check.

    Args:
        list_check (List[Any]): The reference list to check against for exclusion.
        list_sample (List[Any]): The list to filter, excluding elements that
            exist in list_check.

    Returns:
        List[Any]: A list containing elements from list_sample that do not
            exist in list_check. Returns empty list if all elements are found
            in list_check.

    Example:
        >>> list_diff([1, 2, 3], [2, 3, 4, 5])
        [4, 5]
    """
    list_out = list(filter(lambda x: x not in list_check, list_sample))
    if list_out is None:
        list_out = []
    return list_out


def merge_lists_unique(list1: Union[List[Any], pd.Index],
                       list2: Union[List[Any], pd.Index]
                       ) -> List[Any]:
    """Merge two lists while preserving all unique elements from both lists.

    Combines two lists by first finding their intersection, then appending
    unique elements from each list. The result contains all unique elements
    from both input lists without duplicates.

    Args:
        list1 (List[Any]): The first list to merge.
        list2 (List[Any]): The second list to merge.

    Returns:
        List[Any]: A merged list containing all unique elements from both
            input lists. Order is: intersection elements first, followed by
            unique elements from list1, then unique elements from list2.

    Example:
        >>> merge_lists_unique([1, 2, 3], [2, 3, 4, 5])
        [2, 3, 1, 4, 5]
    """
    list_out = list_intersection(list_check=list2, list_sample=list1)
    list_diffs1 = list_diff(list_check=list_out, list_sample=list1)
    list_diffs2 = list_diff(list_check=list_out, list_sample=list2)
    if len(list_diffs1) > 0:
        list_out.extend(list_diffs1)
    if len(list_diffs2) > 0:
        list_out.extend(list_diffs2)
    return list_out


def assert_list_subset(large_list: Union[List[Any], pd.Index],  # larger list
                       list_sample: Union[List[Any], pd.Index],  # smaller list,
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


def list_to_unique_and_dub(lsdata: Union[List[Any], pd.Index]) -> Tuple[List, List]:
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


def assert_list_unique(lsdata: Union[List[str], pd.Index], name: str = '') -> None:
    unique, duplicated = list_to_unique_and_dub(lsdata=lsdata)
    if len(duplicated) > 0:
        raise ValueError(f"list {name} has duplicated elements = {duplicated}")


def move_item_to_first(lsdata: Union[List[Any], pd.Index], item: Any) -> List[Any]:
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
    if isinstance(items, Iterable):
        flat_list = [item for item in flatten(items)]
    else:
        flat_list = [items]
    return flat_list


def update_kwargs(kwargs: Dict[Any, Any],
                  new_kwargs: Optional[Dict[Any, Any]]
                  ) -> Dict[Any, Any]:
    """
    merge two keyword dictionaries, with the second winning on collisions.

    Neither argument is mutated: the copy is the point. This is how the factsheet layer passes one
    defaults dict down to every panel while letting each panel override a few entries, without a
    panel's override leaking into its siblings.

    Args:
        kwargs: base keyword dictionary
        new_kwargs: overrides. None or empty leaves the base unchanged

    Returns:
        a new dictionary; neither input is modified
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

