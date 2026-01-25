"""
scatter plot core
"""
# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels import api as sm
from typing import Union, List, Tuple, Optional

# qis
import qis.plots as qp
import qis.utils as qu


def plot_scatter(df: pd.DataFrame,
                 x: str = None,
                 y: str = None,
                 hue: str = None,
                 xlabel: Union[str, bool, None] = True,
                 ylabel: Union[str, bool, None] = True,
                 title: Optional[str] = None,
                 annotation_labels: List[str] = None,
                 annotation_colors: List[str] = None,
                 annotation_markers: List[str] = None,
                 annotation_color: Optional[str] = 'red',
                 add_universe_model_label: bool = True,
                 add_universe_model_prediction: bool = False,
                 add_universe_model_ci: bool = False,
                 add_hue_model_label: Optional[bool] = None,  # add hue eqs for data with hue
                 ci: Optional[int] = None,
                 order: int = 2,  # regression order
                 full_sample_order: Optional[int] = 2,  # full sample order can be different
                 fit_intercept: bool = True,
                 full_sample_color: str = 'blue',
                 colors: List[str] = None,
                 xvar_format: str = '{:.0%}',
                 yvar_format: str = '{:.0%}',
                 x_limits: Tuple[Optional[float], Optional[float]] = None,
                 y_limits: Tuple[Optional[float], Optional[float]] = None,
                 xticks: List[str] = None,
                 fontsize: int = 10,
                 linewidth: int = 1,
                 markersize: int = 4,
                 full_sample_label: str = 'Full sample: ',
                 add_45line: bool = False,
                 align_axis: bool = False,
                 legend_loc: Optional[str] = 'upper left',
                 ax: plt.Subplot = None,
                 **kwargs
                 ) -> plt.Figure:
    """
    x-y scatter of df
    """
    if x is None:
        if len(df.columns) == 2 or (len(df.columns) == 3 and hue is not None):
            x = df.columns[0]
        else:
            raise ValueError(f"x_column is not defined for more than on columns")
    if y is None:
        if len(df.columns) == 2 or (len(df.columns) == 3 and hue is not None):  # x and y
            y = df.columns[1]
        else:  # melting to column value_name with hue = all columns ba t x
            hue = 'hue'
            df = pd.melt(df, id_vars=[x], value_vars=df.columns.drop(x), var_name=hue,
                         value_name=y)

    if isinstance(xlabel, bool):
        if xlabel is True:
            xlabel = f"x={x}"
        else:
            xlabel = 'x'
    if isinstance(ylabel, bool):
        if ylabel is True:
            ylabel = f"y={y}"
        else:
            ylabel = 'y'

    # drop nans
    if hue is not None:
        df = df[[x, y, hue]].dropna()
    else:
        df = df[[x, y]].dropna()

    if hue is not None and add_hue_model_label is None:  # override to true unless false
        add_hue_model_label = True

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    def plot_regression_with_model(x_col, y_col, data, reg_model, x_vals, color, ax,
                                   ci=None, order=1, markersize=20, linewidth=2):
        """Helper to plot scatter + regression line consistently"""
        # Plot scatter
        sns.scatterplot(x=x_col, y=y_col, data=data, color=color, s=markersize, ax=ax)

        if order > 0:
            if ci is not None:
                # Use regplot for CI band only (hide its line)
                sns.regplot(x=x_col, y=y_col, data=data,
                            ci=ci, order=order, truncate=True,
                            color=color,
                            scatter=False,  # Don't plot scatter again
                            line_kws={'linewidth': 0},  # Hide regplot's line
                            ax=ax)

            # Plot custom regression line from our model
            x1_pred = qu.get_ols_x(x=x_vals, order=order, fit_intercept=reg_model.params.shape[0] > order)
            prediction = reg_model.predict(x1_pred)
            ax.plot(x_vals, prediction, color=color, lw=linewidth, linestyle='-')

    estimated_reg_models = {}
    if hue is not None:
        if colors is None:
            colors = qp.get_n_sns_colors(n=len(df[hue].unique()), **kwargs)
        palette = colors

        hue_ids = df[hue].unique()
        for idx, hue_id in enumerate(hue_ids):
            # Estimate model
            data_hue = df[df[hue] == hue_id].replace([np.inf, -np.inf], np.nan).dropna().sort_values(by=x)
            x_ = data_hue[x].to_numpy()
            y_ = data_hue[y].to_numpy()
            x1 = qu.get_ols_x(x=x_, order=order, fit_intercept=fit_intercept)
            reg_model = sm.OLS(y_, x1).fit()
            estimated_reg_models[hue_id] = reg_model

            # Plot
            plot_regression_with_model(x, y, data_hue, reg_model, x_, palette[idx], ax,
                                       ci=ci, order=order, markersize=markersize, linewidth=linewidth)

    else:
        if full_sample_order is None:
            pass
        elif full_sample_order == 0:
            sns.scatterplot(x=x, y=y, data=df, s=markersize, color=full_sample_color, ax=ax)
        else:
            # Estimate model for consistency
            data_clean = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(by=x)
            x_ = data_clean[x].to_numpy()
            y_ = data_clean[y].to_numpy()
            x1 = qu.get_ols_x(x=x_, order=full_sample_order, fit_intercept=fit_intercept)
            reg_model = sm.OLS(y_, x1).fit()

            # Plot
            plot_regression_with_model(x, y, data_clean, reg_model, x_, full_sample_color, ax,
                                       ci=ci, order=full_sample_order, markersize=markersize, linewidth=linewidth)

    # add ml equations to labels
    legend_labels = []
    legend_colors = []
    if full_sample_order is not None:
        if (add_universe_model_prediction or add_universe_model_label or add_universe_model_ci) and full_sample_order > 0:
            xy = df[[x, y]].sort_values(by=x)
            x_ = xy[x].to_numpy()
            y_ = xy[y].to_numpy()
            x1 = qu.get_ols_x(x=x_, order=full_sample_order, fit_intercept=fit_intercept)
            reg_model = sm.OLS(y_, x1).fit()

            if add_universe_model_prediction:
                prediction = reg_model.predict(x1)
                ax.plot(x, prediction, color=full_sample_color, lw=linewidth, linestyle='--')

            if add_universe_model_ci:
                y_model = reg_model.predict(x1)
                ci = calc_ci(x=x, y=y, y_model=y_model)
                # ax.fill_between(x, y + ci, y - ci, color="None", linestyle="--")
                ax.plot(x, y_model - ci, "--", color="0.5")
                ax.plot(x, y_model + ci, "--", color="0.5")

            if add_universe_model_label:
                text_str = f"{full_sample_label} " \
                           f"{qu.reg_model_params_to_str(reg_model=reg_model, order=full_sample_order, fit_intercept=fit_intercept, **kwargs)}"
                legend_labels.append(text_str)
                legend_colors.append(full_sample_color)

    # add colors for annotation labels
    df['color'] = full_sample_color
    if hue is not None :
        hue_ids = df[hue].unique()
        for color, hue_id in zip(colors, hue_ids):
            df.loc[df[hue] == hue_id, 'color'] = 'red'  # ad color for hue
            if order > 0:
                if add_hue_model_label:
                    reg_model = estimated_reg_models[hue_id]
                    text_str = (f"{hue_id}: " 
                                f"{qu.reg_model_params_to_str(reg_model=reg_model, order=order, fit_intercept=fit_intercept, **kwargs)}")
                else:
                    text_str = hue_id
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
        if annotation_markers is None:
            annotation_markers = len(df.index) * ['o']

        for label, x_, y_, color, marker in zip(annotation_labels, df[x], df[y], colors, annotation_markers):
            ax.annotate(label,
                        xy=(x_, y_), xytext=(1, 1),
                        textcoords='offset points', ha='left', va='bottom',
                        color=color,
                        fontsize=fontsize)
            if label != '':
                ax.scatter(x=x_, y=y_, c=color, s=20, marker=marker)

    if align_axis or add_45line:
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        min = ymin if ymin < xmin else xmin
        max = ymax if ymax > xmax else xmax
        ax.set_xlim(min, max)
        ax.set_ylim(min, max)

        if add_45line:
            x = np.linspace(*ax.get_xlim())
            ax.plot(x, x, color='black', lw=1, linestyle='--')

    if x_limits is not None:
        qp.set_x_limits(ax=ax, x_limits=x_limits)
    if y_limits is not None:
        qp.set_y_limits(ax=ax, y_limits=y_limits)

    if xticks is not None:
        qp.set_ax_tick_labels(ax=ax, x_rotation=0, xticks=df[x].to_numpy(), x_labels=xticks,
                              fontsize=fontsize)
    else:
        qp.set_ax_tick_labels(ax=ax, x_rotation=0, fontsize=fontsize)

    qp.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)

    qp.set_ax_ticks_format(ax=ax, xvar_format=xvar_format, yvar_format=yvar_format, fontsize=fontsize, **kwargs)

    qp.set_legend(ax=ax,
                  labels=legend_labels,
                  colors=legend_colors,
                  legend_loc=legend_loc,
                  fontsize=fontsize,
                  **kwargs)

    if title is not None:
        qp.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    qp.set_spines(ax=ax, **kwargs)

    return fig


def plot_classification_scatter(df: pd.DataFrame,
                                x: Optional[str] = None,
                                y: Optional[str] = None,
                                hue_name: str = 'hue',
                                num_buckets: Optional[int] = None,
                                bins: np.ndarray = np.array([-3.0, -1.5, 0.0, 1.5, 3.0]),
                                order: int = 1,
                                title: str = None,
                                full_sample_order: Optional[int] = 3,
                                markersize: int = 10,
                                xvar_format: str = '{:.2f}',
                                yvar_format: str = '{:.2f}',
                                fit_intercept: bool = False,
                                ax: plt.Subplot = None,
                                **kwargs
                                ) -> Optional[plt.Figure]:
    """
    add bin classification using x_column
    """
    if x is None:
        if len(df.columns) == 2:
            x = df.columns[0]
        else:
            raise ValueError(f"x_column is not defined for more than on columns")
    if y is None:
        if len(df.columns) == 2:  # x and y
            y = df.columns[1]
        else:
            raise ValueError(f"y_column is not defined for more than on columns")

    df, _ = qu.add_quantile_classification(df=df, x_column=x, hue_name=hue_name, num_buckets=num_buckets, bins=bins)

    fig = plot_scatter(df=df,
                       x=x,
                       y=y,
                       hue=hue_name,
                       fit_intercept=fit_intercept,
                       title=title,
                       order=order,
                       full_sample_order=full_sample_order,
                       markersize=markersize,
                       xvar_format=xvar_format,
                       yvar_format=yvar_format,
                       add_universe_model_ci=False,
                       add_hue_model_label=True,
                       ax=ax,
                       **kwargs)
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


def estimate_classification_scatter(df: pd.DataFrame,
                                    x: Optional[str] = None,
                                    y: Optional[str] = None,
                                    hue_name: str = 'hue',
                                    num_buckets: Optional[int] = None,
                                    bins: np.ndarray = np.array([-3.0, -1.5, 0.0, 1.5, 3.0]),
                                    order: int = 1,
                                    fit_intercept: bool = False,
                                    ) -> pd.DataFrame:
    """
    add bin classification using x_column
    """
    if x is None:
        if len(df.columns) == 2:
            x = df.columns[0]
        else:
            raise ValueError(f"x_column is not defined for more than on columns")
    if y is None:
        if len(df.columns) == 2:  # x and y
            y = df.columns[1]
        else:
            raise ValueError(f"y_column is not defined for more than on columns")

    df, _ = qu.add_quantile_classification(df=df, x_column=x, hue_name=hue_name, num_buckets=num_buckets, bins=bins)

    x_np = df[x].to_numpy()
    x_range = np.linspace(np.min(x_np), np.max(x_np), 200)

    dfs = df.groupby(hue_name)
    y_rpeds = []
    for hue, df in dfs:
        # estimate model equation
        data_hue = df.sort_values(by=x)
        x_ = data_hue[x].to_numpy()
        y_ = data_hue[y].to_numpy()
        x1 = qu.get_ols_x(x=x_, order=order, fit_intercept=fit_intercept)
        reg_model = sm.OLS(y_, x1).fit()
        x_hue = np.extract(np.logical_and(x_range>=np.min(x_), x_range<np.max(x_)), x_range)
        y_hue = reg_model.predict(x_hue)
        y_rpeds.append(pd.Series(y_hue, index=x_hue))
    y_rpeds = pd.concat(y_rpeds).sort_index()
    return y_rpeds


def plot_multivariate_scatter_with_prediction(df: pd.DataFrame,
                                              x: List[str],
                                              y: str,
                                              x_axis_column: str,
                                              hue: str = None,
                                              xlabel: Union[str, bool, None] = True,
                                              ylabel: Union[str, bool, None] = True,
                                              full_sample_color: str = 'crimson',
                                              linewidth: float = 1.5,
                                              fit_intercept: bool = True,
                                              verbose: bool = True,
                                              xvar_format: str = '{:.0%}',
                                              yvar_format: str = '{:.0%}',
                                              title: str = None,
                                              fontsize: int = 10,
                                              legend_loc: str = 'upper left',
                                              ax: plt.Subplot = None,
                                              **kwargs
                                              ) -> Optional[plt.Figure]:

    prediction, params, reg_label = qu.fit_multivariate_ols(x=df[x], y=df[y], fit_intercept=fit_intercept, verbose=verbose)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    sns.scatterplot(x=x_axis_column, y=y, data=df, hue=hue, ax=ax)

    ds_pred = pd.Series(prediction.to_numpy(), index=df.loc[prediction.index, x_axis_column]).sort_index()
    ax.plot(ds_pred.index, ds_pred.to_numpy(), color=full_sample_color, lw=linewidth, linestyle='-')

    if isinstance(xlabel, bool):
        if xlabel is True:
            xlabel = f"x={x_axis_column}"
        else:
            xlabel = ''
    if isinstance(ylabel, bool):
        if ylabel is True:
            ylabel = f"y={y}"
        else:
            ylabel = ''
    qp.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)

    qp.set_ax_ticks_format(ax=ax, xvar_format=xvar_format, yvar_format=yvar_format, fontsize=fontsize, **kwargs)

    # add prediction and axis labels
    labels = [f"prediction {reg_label}"]
    colors = [full_sample_color]
    leg = ax.get_legend()
    for label, line in zip(leg.get_texts(), leg.get_lines()):
        labels.append(label.get_text())
        colors.append(line.get_color())

    qp.set_legend(ax=ax,
                  labels=labels,
                  colors=colors,
                  legend_loc=legend_loc,
                  fontsize=fontsize,
                  **kwargs)

    if title is not None:
        qp.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    qp.set_spines(ax=ax, **kwargs)

    return fig
