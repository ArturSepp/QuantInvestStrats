"""
scatter plot core
"""
# built in
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels import api as sm
from typing import Union, List, Tuple, Optional

# qis
import qis.plots.utils as put
from qis.utils.ols import get_ols_x, reg_model_params_to_str


def plot_scatter(df: pd.DataFrame,
                 x_column: str = None,
                 y_column: str = None,
                 hue: str = None,
                 xlabel: Union[str, bool, None] = True,
                 ylabel: Union[str, bool, None] = True,
                 title: Optional[str] = None,
                 use_regplot: bool = False,
                 annotation_labels: List[str] = None,
                 annotation_colors: List[str] = None,
                 annotation_color: Optional[str] = 'red',
                 add_universe_model_label: bool = True,
                 add_universe_model_prediction: bool = False,
                 add_universe_model_ci: bool = False,
                 add_hue_model_label: bool = False,  # add hue eqs for data with hue
                 ci: Optional[int] = None,
                 order: int = 1,  # regression order
                 fit_intercept: bool = True,
                 full_sample_order: Optional[int] = None,  # full sample order can be different
                 color0: str = 'darkblue',
                 colors: List[str] = None,
                 xvar_format: str = '{:.0%}',
                 yvar_format: str = '{:.0%}',
                 x_limits: Tuple[Optional[float], Optional[float]] = None,
                 y_limits: Tuple[Optional[float], Optional[float]] = None,
                 xticks: List[str] = None,
                 fontsize: int = 10,
                 linewidth: float = 1.5,
                 markersize: int = 4,
                 first_color_fixed: bool = False,
                 full_sample_label: str = 'Full sample: ',
                 is_add_45line: bool = False,
                 is_r2_only: bool = False,
                 legend_loc: str = 'upper left',
                 value_name: str = 'value_name',
                 ax: plt.Subplot = None,
                 **kwargs
                 ) -> plt.Figure:
    df = df.copy().dropna()

    if x_column is None:
        if len(df.columns) == 2:
            x_column = df.columns[0]
        else:
            raise ValueError(f"x_column is not defined for more than on columns")
    if y_column is None:
        if len(df.columns) == 2:  # x and y
            y_column = df.columns[1]
        else:  # melting to column value_name with hue = all columns ba t x
            hue = 'hue'
            add_hue_model_label = True
            y_column = value_name
            df = pd.melt(df, id_vars=[x_column], value_vars=df.columns.drop(x_column), var_name=hue,
                         value_name=value_name)

    if full_sample_order is None:
        full_sample_order = order

    if add_universe_model_label:
        hue_offset = 1
    else:
        hue_offset = 0

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    estimated_reg_models = {}

    if hue is not None:
        if colors is None:
            colors = put.get_n_colors(n=len(df[hue].unique()), first_color_fixed=first_color_fixed)

        if hue_offset == 1:
            palette = colors #[color0] +
        else:
            palette = colors

        hue_ids = df[hue].unique()
        for idx, hue_id in enumerate(hue_ids):
            # estimate model equation
            data_hue = df[df[hue] == hue_id].sort_values(by=x_column)
            x = data_hue[x_column].to_numpy(dtype=np.float)
            y = data_hue[y_column].to_numpy(dtype=np.float)
            x1 = get_ols_x(x=x, order=order, fit_intercept=fit_intercept)
            reg_model = sm.OLS(y, x1).fit()
            estimated_reg_models[hue_id] = reg_model

            # plot data points
            #try:
            sns.scatterplot(x=x_column, y=y_column, data=data_hue, color=palette[idx], s=markersize, ax=ax)
            #except:
            #    print(f"scatter plot problem with {data_hue}")

            if order > 0:  # plot prediction
                if use_regplot:
                    # not possible to control reg equation in regplot
                    sns.regplot(x=x_column, y=y_column, data=data_hue,
                                ci=ci,
                                order=order, truncate=True,
                                color=palette[idx],
                                scatter_kws={'s': markersize},
                                line_kws={'linewidth': linewidth},
                                ax=ax)
                else:
                    prediction = reg_model.predict(x1)
                    ax.plot(x, prediction, color=palette[idx], lw=linewidth, linestyle='-')

    else:
        if full_sample_order == 0:  # just scatter plot
            sns.scatterplot(x=x_column, y=y_column, data=df,
                            # ci=ci,
                            s=markersize, color=color0, ax=ax)
        else:  # regplot add scatter and ml lines even if order is == 0
            sns.regplot(x=x_column, y=y_column, data=df, ci=ci, order=full_sample_order, color=color0,
                        scatter_kws={'s': markersize},
                        line_kws={'linewidth': linewidth}, ax=ax)

    # add ml equations to labels
    legend_labels = []
    legend_colors = []
    if (add_universe_model_prediction or add_universe_model_label or add_universe_model_ci) and order > 0:
        xy = df[[x_column, y_column]].sort_values(by=x_column)
        x = xy[x_column].to_numpy(dtype=np.float)
        y = xy[y_column].to_numpy(dtype=np.float)
        x1 = get_ols_x(x=x, order=full_sample_order, fit_intercept=fit_intercept)
        reg_model = sm.OLS(y, x1).fit()

        if add_universe_model_prediction:
            prediction = reg_model.predict(x1)
            ax.plot(x, prediction, color=color0, lw=linewidth, linestyle='--')

        if add_universe_model_ci:
            y_model = reg_model.predict(x1)
            ci = calc_ci(x=x, y=y, y_model=y_model)
            # ax.fill_between(x, y + ci, y - ci, color="None", linestyle="--")
            ax.plot(x, y_model - ci, "--", color="0.5")
            ax.plot(x, y_model + ci, "--", color="0.5")

        if add_universe_model_label:
            text_str = f"{full_sample_label} " \
                       f"{reg_model_params_to_str(reg_model=reg_model, order=full_sample_order, is_r2_only=False, fit_intercept=fit_intercept, **kwargs)}"
            legend_labels.append(text_str)
            legend_colors.append(color0)

    # add colors for annotation labels
    df['color'] = color0
    if hue is not None and add_hue_model_label:
        hue_ids = df[hue].unique()
        for color, hue_id in zip(colors, hue_ids):
            df.loc[df[hue] == hue_id, 'color'] = 'red'  # ad color for hue
            if order > 0:
                reg_model = estimated_reg_models[hue_id]
                text_str = (f"{hue_id}: " 
                            f"{reg_model_params_to_str(reg_model=reg_model, order=order, is_r2_only=is_r2_only, fit_intercept=fit_intercept, **kwargs)}")
                legend_labels.append(text_str)
            else:
                legend_labels.append(hue_id)
            legend_colors.append(color)

    elif hue is not None and order == 0:
        legend_labels = df[hue].unique()
        legend_colors = colors

    # add labels
    if annotation_labels is not None:
        if annotation_colors is not None:
            colors = annotation_colors
        elif annotation_color is not None:
            colors = len(df.index) * [annotation_color]
        else:
            colors = df['color']
        for label, x, y, color in zip(annotation_labels, df[x_column], df[y_column], colors):
            ax.annotate(label,
                        xy=(x, y), xytext=(1, 1),
                        textcoords='offset points', ha='left', va='bottom',
                        color=color,
                        fontsize=fontsize)
            if label != '':
                ax.scatter(x=x, y=y, c=color, s=20)

    if is_add_45line:  # make equal:
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        min = ymin if ymin < xmin else xmin
        max = ymax if ymax > xmax else xmax
        ax.set_xlim([min, max])
        ax.set_ylim([min, max])
        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x, color='black', lw=1, linestyle='--')

    if x_limits is not None:
        put.set_x_limits(ax=ax, x_limits=x_limits)
    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)

    if xticks is not None:
        put.set_ax_tick_labels(ax=ax, x_rotation=0, xticks=df[x_column].to_numpy(), x_labels=xticks,
                               fontsize=fontsize)
    else:
        put.set_ax_tick_labels(ax=ax, x_rotation=0, fontsize=fontsize)

    if isinstance(xlabel, bool) and xlabel is True:
        xlabel = 'x = ' + x_column
    if isinstance(ylabel, bool) and ylabel is True:
        ylabel = 'y = ' + y_column
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)

    put.set_ax_ticks_format(ax=ax, xvar_format=xvar_format, yvar_format=yvar_format, fontsize=fontsize, **kwargs)

    put.set_legend(ax=ax,
                   labels=legend_labels,
                   colors=legend_colors,
                   legend_loc=legend_loc,
                   fontsize=fontsize,
                   **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    put.set_spines(ax=ax, **kwargs)

    return fig


def calc_ci(x: np.ndarray, y: np.ndarray, y_model: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = 2
    dof = n - m
    t = stats.t.ppf(0.95, dof)
    #x2 = np.linspace(np.min(x), np.max(x), 100)

    # Estimates of Error in Data/Model
    resid = y - y_model
    # chi2 = np.sum((resid / y_model) ** 2)  # chi-squared; estimates error in data
    # chi2_red = chi2 / dof  # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid ** 2) / dof)  # standard deviation of the error

    ci = t * s_err * np.sqrt(1 / n + (x - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    return ci
