"""
implementation for infinite lower and upper bounds for pd.cut with wrapper
https://stackoverflow.com/questions/30127427/pandas-cut-with-infinite-upper-lower-bounds
"""
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from pandas import Categorical
from enum import Enum


def x_bins_cut(a: np.ndarray,
               bins: np.ndarray,
               lower_infinite: bool = True,
               upper_infinite: bool = True,
               xvar_format: str = '{:.2f}',
               is_value_labels: bool = True,
               value_label: str = 'regime',
               bucket_prefix: str = None,
               **kwargs
               ) -> Tuple[Categorical, List[str]]:
    r"""Wrapper around pandas cut() to create infinite lower/upper bounds with proper labeling.

    Takes all the same arguments as pandas cut(), plus two more.

    Args :
        lower_infinite (bool, optional) : set whether the lower bound is infinite
            Default is True. If true, and your first bin element is something like 20, the
            first bin label will be '<= 20' (depending on other cut() parameters)
        upper_infinite (bool, optional) : set whether the upper bound is infinite
            Default is True. If true, and your last bin element is something like 20, the
            first bin label will be '> 20' (depending on other cut() parameters)
        **kwargs : any standard pandas cut() labeled parameters

    Returns :
        out : same as pandas cut() return value
        bins : same as pandas cut() return value
    """

    # Quick passthru if no infinite bounds
    if not lower_infinite and not upper_infinite:
        return pd.cut(a, bins, **kwargs)

    # Setup
    num_labels = len(bins) - 1
    include_lowest = kwargs.get("include_lowest", False)
    right = kwargs.get("right", True)

    # Prepend/Append infinities where indiciated
    bins_final = bins.copy()
    if upper_infinite:
        bins_final = np.append(bins_final, float("inf"))
        num_labels += 1
    if lower_infinite:
        bins_final = np.append(float("-inf"), bins_final)
        num_labels += 1

    # Decide all boundary symbols based on traditional cut() parameters
    symbol_lower = "<=" if include_lowest and right else "<"
    left_bracket = u"=" + "(" if right else "["
    right_bracket = "]" if right else ")"
    symbol_upper = ">" if right else ">="

    # Inner function reused in multiple clauses for labeling
    def make_label(i: int, lb: str = left_bracket, rb: str = right_bracket):
        if is_value_labels:
            label = f"{lb}{xvar_format.format(bins_final[i])}, {xvar_format.format(bins_final[i+1])}{rb}"
        else:
            label = f"{value_label}-{i+1}"
        return label

    # Create custom labels
    labels = []
    for i in range(0, num_labels):
        if i == 0 and is_value_labels:
            if lower_infinite:
                new_label = f"{symbol_lower}{xvar_format.format(bins_final[i+1])}"
            elif include_lowest:
                new_label = make_label(i, lb="[")
            else:
                new_label = make_label(i)
        elif upper_infinite and i == (num_labels - 1) and is_value_labels:
            new_label = f"{symbol_upper}{xvar_format.format(bins_final[i])}"
        else:
            new_label = make_label(i)
        if bucket_prefix is not None:
            new_label = f"{bucket_prefix}{new_label}"
        labels.append(new_label)
    try:
        out = pd.cut(a, bins_final, labels=labels)
    except ValueError:   # labels must be unique if ordered=True
        labels = [f"bin_{n+1}" for n, _ in enumerate(bins_final[1:])]
        out = pd.cut(a, bins_final, labels=labels)
    return out, labels


def add_classification(df: pd.DataFrame,
                       class_var_col: str,
                       bins: np.ndarray = np.array([-3.0, -1.5, 0.0, 1.5, 3.0]),
                       hue_name: str = 'hue',
                       xvar_format: str = '{:.2f}',
                       is_value_labels: bool = True,
                       is_to_str: bool = True,
                       bucket_prefix: str = None,
                       **kwargs
                       ) -> Tuple[pd.DataFrame, List[str]]:

    df = df.sort_values(by=class_var_col)
    class_var, labels = x_bins_cut(a=df[class_var_col],
                                   bins=bins,
                                   is_value_labels=is_value_labels,
                                   xvar_format=xvar_format,
                                   bucket_prefix=bucket_prefix,
                                   **kwargs)
    df = df.copy()
    df[hue_name] = class_var
    if is_to_str:
        df[hue_name] = df[hue_name].astype(str)
    return df, labels


def add_quantile_classification(df: pd.DataFrame,
                                x_column: str,
                                hue_name: str = 'hue',
                                num_buckets: Optional[Union[np.ndarray, int]] = None,
                                bins: Optional[np.ndarray] = np.array([-3.0, -1.5, 0.0, 1.5, 3.0]),
                                xvar_format: str = '{:.2f}',
                                is_value_labels: bool = True,
                                bucket_prefix: str = None,
                                **kwargs
                                ) -> Tuple[pd.DataFrame, List[str]]:
    if num_buckets is not None:  # create bins
        if isinstance(num_buckets, int):
            bins = np.nanquantile(df[x_column], q=[(1.0 / num_buckets) * (n + 1) for n in np.arange(num_buckets - 1)])
        else:
            bins = np.nanquantile(df[x_column], q=num_buckets)
    df, labels = add_classification(df=df,
                                    class_var_col=x_column,
                                    bins=bins,
                                    hue_name=hue_name,
                                    is_value_labels=is_value_labels,
                                    xvar_format=xvar_format,
                                    bucket_prefix=bucket_prefix,
                                    **kwargs)
    return df, labels


def sort_index_by_hue(df: pd.DataFrame, hue_order: List[str]) -> pd.DataFrame:
    sort_column = 'sort_column'
    name_sort = {key: idx for idx, key in enumerate(hue_order)}
    df[sort_column] = df.index.map(name_sort)
    df = df.sort_values(by=sort_column).drop(sort_column, axis=1)
    return df


def add_hue_fixed_years(df: pd.DataFrame,
                        hue: str,
                        fixed_years: List[int] = (2001, 2006, 2010, 2018, 2021, 2022)
                        ) -> pd.DataFrame:
    """
    map sequence of years into hue = (fixed_years[i-1], fixed_years[i]]
    """
    pd_idx = pd.IntervalIndex.from_breaks(fixed_years, closed='right')
    mapper = {idx: f"{p.left+1}-{p.right}" for idx, p in enumerate(pd_idx)}
    hue_vals = pd.Series([pd_idx.get_loc(x.year) for x in df.index]).map(mapper)
    df[hue] = hue_vals.to_list()
    return df


def add_hue_years(df: pd.DataFrame, hue: str) -> pd.DataFrame:
    """
    add hue in years
    """
    df[hue] = [x.year for x in df.index]
    return df


class UnitTests(Enum):
    CUT = 1
    CLASS = 2


def run_unit_test(unit_test: UnitTests):

    np.random.seed(2)  # freeze seed

    if unit_test == UnitTests.CUT:

        n = 1000000
        x = np.random.normal(0.0, 1.0, n)
        bins = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])

        pd1 = pd.cut(x, bins)
        print('pd.cut')
        print(pd1)

        pd2, labels = x_bins_cut(x, bins)
        print('x_bins_cut')
        print(pd2)

    elif unit_test == UnitTests.CLASS:

        n = 10000
        x = np.random.normal(0.0, 1.0, n)
        eps = np.random.normal(0.0, 1.0, n)
        y = x + eps
        bins = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        df = pd.concat([pd.Series(x, name='x'), pd.Series(y, name='y')], axis=1)

        df1, labels = add_classification(df=df, class_var_col='x', bins=bins)
        print(df1)


if __name__ == '__main__':

    unit_test = UnitTests.CUT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
