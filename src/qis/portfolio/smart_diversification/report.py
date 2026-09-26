"""
the smart-diversification report: what adding an overlay to a principal portfolio buys.

``SmartDiversificationReport`` holds ``principal_nav`` and a panel of ``overlay_navs`` and draws
each panel onto a supplied axis or into a new ``plt.Figure``. The mixes come from
``create_overlay_portfolio_curve`` in ``overlay_curve.py``;
``compute_smart_diversification_curve`` reads two statistics off them, the regime-conditional
``PerfStat.BEAR_SHARPE`` against the full-sample ``PerfStat.SHARPE_RF0`` by default.
``plot_smart_diversification_curve`` draws one curve per overlay,
``plot_smart_diversification_scatter`` one point per overlay and, unless ``is_drop_principal``,
the principal too, with a cross-sectional fit.

A candidate-overlay exhibit, not a track record: unlike ``strategy_factsheet.py`` it carries no
realised weights, turnover or costs. Until qis 5.31.0 this module was
``qis/portfolio/reports/overlays_smart_diversification.py``; that import path remains valid.
"""
# packages
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, Union, List, Optional, Tuple
import qis as qis
from qis import PerfStat, RegimeData, BenchmarkReturnsQuantilesRegime, PerfParams
from qis.plots.utils import calc_table_height, get_n_markers
from qis.utils.df_str import series_to_str
from qis.portfolio.smart_diversification.overlay_curve import create_overlay_portfolio_curve

PERF_PARAMS = PerfParams(freq='ME')


PERF_COLUMNS = (
    # PerfStat.START_DATE,
    # PerfStat.END_DATE,
    PerfStat.PA_RETURN,
    PerfStat.VOL,
    PerfStat.SHARPE_RF0,
    PerfStat.BEAR_SHARPE,
    PerfStat.NORMAL_SHARPE,
    PerfStat.BULL_SHARPE,
    PerfStat.MAX_DD,
    PerfStat.MAX_DD_VOL,
    PerfStat.SKEWNESS)


@dataclass
class SmartDiversificationReport:
    """Diversification frontiers of candidate overlays stacked on a principal portfolio.

    Attributes:
        overlay_navs: navs of the candidate overlays, one column per overlay
        principal_nav: nav of the principal portfolio, also the regime benchmark
        regime_classifier: classifier of the benchmark regimes; None uses
            ``BenchmarkReturnsQuantilesRegime()``, quarterly at the 16%/84% quantiles
        perf_params: performance parameters of the statistics; None uses ``PERF_PARAMS``,
            ``PerfParams(freq='ME')``: monthly volatility and regression sampling
        benchmark_description: optional label of the principal portfolio
    """
    overlay_navs: pd.DataFrame
    principal_nav: pd.Series = None
    regime_classifier: BenchmarkReturnsQuantilesRegime = None
    perf_params: PerfParams = None
    benchmark_description: str = None

    def __post_init__(self):
        """Fill the defaults of the regime classifier and the performance parameters."""
        if self.regime_classifier is None:
            self.regime_classifier = BenchmarkReturnsQuantilesRegime()
        if self.perf_params is None:
            self.perf_params = PERF_PARAMS

    def compute_smart_diversification_curve(self,
                                            principal_nav: pd.Series,
                                            overlay_nav: pd.Series,
                                            principal_weight: float = 1.0,
                                            max_overlay_weight: float = 1.0,
                                            rebalancing_freq: str = 'QE',
                                            is_principal_weight_fixed: bool = True,
                                            x_var: PerfStat = PerfStat.BEAR_SHARPE,
                                            y_var: PerfStat = PerfStat.SHARPE_RF0
                                            ) -> pd.DataFrame:
        """Two statistics of the eleven principal-overlay mixes of one overlay.

        The first mix holds no overlay, so the first row is the principal portfolio; each mix
        is classified against that first mix.

        Args:
            principal_nav: nav of the principal portfolio
            overlay_nav: nav of the overlay
            principal_weight: weight of the principal portfolio in every mix
            max_overlay_weight: overlay weight of the last mix
            rebalancing_freq: rebalancing frequency of the mixes
            is_principal_weight_fixed: keep ``principal_weight`` fixed and add the overlay on top;
                False funds the overlay from the principal, with weights ``1 - w`` and ``w``
            x_var: statistic of the first column
            y_var: statistic of the second column

        Returns:
            one row per mix and the columns of ``x_var`` and ``y_var``
        """
        portfolio_navs = create_overlay_portfolio_curve(principal_nav=principal_nav,
                                                        overlay_nav=overlay_nav,
                                                        principal_weight=principal_weight,
                                                        max_overlay_weight=max_overlay_weight,
                                                        rebalancing_freq=rebalancing_freq,
                                                        is_principal_weight_fixed=is_principal_weight_fixed)
        portfolio_navs = portfolio_navs.dropna(how='any')  # drop union-index padding
        cvar_table, _ = self.regime_classifier.compute_regimes_pa_perf_table(prices=portfolio_navs,
                                                                             benchmark=portfolio_navs.columns[0],
                                                                             perf_params=self.perf_params)
        return cvar_table[[x_var.to_str(), y_var.to_str()]]

    def get_overlay_points(self,
                           principal_nav: pd.Series,
                           x_var: PerfStat = PerfStat.BEAR_SHARPE,
                           y_var: PerfStat = PerfStat.SHARPE_RF0
                           ) -> pd.DataFrame:
        """Two statistics of the principal portfolio and of each standalone overlay.

        Args:
            principal_nav: nav of the principal portfolio, the regime benchmark
            x_var: statistic of the first column
            y_var: statistic of the second column

        Returns:
            one row for the principal and one per overlay, with the columns of ``x_var`` and
            ``y_var``
        """
        portfolio_navs = pd.concat([principal_nav, self.overlay_navs], axis=1, sort=True)

        # the first 100% is the benchmark
        cvar_table, _ = self.regime_classifier.compute_regimes_pa_perf_table(prices=portfolio_navs,
                                                                             benchmark=portfolio_navs.columns[0],
                                                                             perf_params=self.perf_params)
        return cvar_table[[x_var.to_str(), y_var.to_str()]]

    def plot_nav(self,
                 var_format: str = '{:.0%}',
                 sharpe_format: str = '{:.2f}',
                 principal_nav: pd.Series = None,
                 is_include_principal: bool = False,
                 title: str = None,
                 is_log: bool = True,
                 ax: plt.Subplot = None,
                 **kwargs
                 ) -> plt.Figure:
        """Navs of the overlays, rebased to one, optionally with the principal portfolio.

        Args:
            var_format: format of the p.a. returns in the legend
            sharpe_format: format of the Sharpe ratios in the legend
            principal_nav: nav of the principal portfolio; None uses the report's
            is_include_principal: also draw the principal portfolio
            title: figure title
            is_log: logarithmic y axis
            ax: axis to draw on; None creates a figure
            **kwargs: passed to ``qis.plot_prices``

        Returns:
            the figure, the one of ``ax`` when given
        """
        if principal_nav is None:
            principal_nav = self.principal_nav

        if is_include_principal:
            prices = pd.concat([principal_nav, self.overlay_navs], axis=1, sort=True)
        else:
            prices = self.overlay_navs.copy()

        fig = qis.plot_prices(prices=prices,
                              perf_params=self.perf_params,
                              start_to_one=True,
                              is_log=is_log,
                              var_format=var_format,
                              sharpe_format=sharpe_format,
                              title=title,
                              ax=ax,
                              **kwargs)

        if ax is None:
            ax = fig.axes[0]
        """
        classification_data = sdi.create_regime_classification(
            regime_classifier=self.regime_classifier, pivot_prices=principal_nav)

        qis.add_regime_shadows_to_ax(ax=ax,
                                     regime_classifier=self.regime_classifier,
                                     classification_data=classification_data,
                                     pivot_prices=principal_nav,
                                     price_data_index=principal_nav.index)
        """
        return fig

    def plot_ra_table(self,
                      time_period: qis.TimePeriod = None,
                      desc_map: Dict[str, str] = None,
                      ticker_map: Dict[str, str] = None,
                      perf_columns: List[PerfStat] = PERF_COLUMNS,
                      columns_title: str = 'Assets',
                      ax: plt.Subplot = None,
                      **kwargs
                      ) -> plt.Figure:
        """Table of the performance statistics of the principal portfolio and the overlays.

        Args:
            time_period: sample of the statistics; None uses the full history
            desc_map: optional names by column, shown in a ``Name`` column
            ticker_map: optional tickers by column, shown in a ticker column
            perf_columns: statistics of the table, in order
            columns_title: header of the first column
            ax: axis to draw on; None creates a figure
            **kwargs: passed to the statistic formats and to ``qis.plot_df_table``

        Returns:
            the figure, the one of ``ax`` when given
        """
        prices = pd.concat([self.principal_nav, self.overlay_navs], axis=1, sort=True)

        if time_period is not None:
            prices = time_period.locate(prices)

        benchmark = (self.principal_nav.name if isinstance(self.principal_nav, pd.Series)
                     else self.principal_nav.columns[0])
        cvar_table, _ = self.regime_classifier.compute_regimes_pa_perf_table(prices=prices,
                                                                             benchmark=benchmark,
                                                                             perf_params=self.perf_params)
        table_data = pd.DataFrame(data=prices.columns, index=cvar_table.index,
                                  columns=[columns_title])

        col_widths = [15]
        if desc_map is not None:
            table_data['Name'] = table_data.index.map(desc_map)
            col_widths.append(15)
        if ticker_map is not None:
            table_data['BBG\nTicker'] = table_data.index.map(ticker_map)
            col_widths.append(15)

        for perf_column in perf_columns:
            table_data[perf_column.to_str()] = series_to_str(ds=cvar_table[perf_column.to_str()],
                                                             var_format=perf_column.to_format(**kwargs))
            col_widths.append(7)

        special_columns_colors = [(0, 'steelblue')]
        fig = qis.plot_df_table(df=table_data,
                                col_widths=col_widths,
                                column_width=7,
                                add_index_as_column=False,
                                index_column_name='Strategies',
                                special_columns_colors=special_columns_colors,
                                row_height=0.5,
                                ax=ax,
                                **kwargs)
        return fig

    def plot_conditional_sharpes(self,
                                 principal_nav: pd.Series = None,
                                 overlay_navs: pd.DataFrame = None,
                                 regime_data_to_plot: RegimeData = RegimeData.REGIME_SHARPE,
                                 var_format: str = '{:.2f}',
                                 is_names_to_2lines: bool = False,
                                 title: str = None,
                                 ax: plt.Subplot = None,
                                 **kwargs
                                 ) -> plt.Figure:
        """Regime-conditional bars of the principal and overlays, by ``qis.plot_regime_data``.

        Args:
            principal_nav: nav of the principal portfolio, the regime benchmark; None uses the
                report's
            overlay_navs: navs of the overlays; None uses the report's
            regime_data_to_plot: statistic of the bars
            var_format: format of the bar values
            is_names_to_2lines: break the names at spaces and before ``+``
            title: figure title
            ax: axis to draw on; None creates a figure
            **kwargs: passed to ``qis.plot_regime_data``

        Returns:
            the figure, the one of ``ax`` when given
        """
        if principal_nav is None:
            principal_nav = self.principal_nav

        if overlay_navs is None:
            overlay_navs = self.overlay_navs

        prices = pd.concat([principal_nav, overlay_navs], axis=1, sort=True)
        benchmark = (principal_nav.name if isinstance(principal_nav, pd.Series)
                     else principal_nav.columns[0])

        if is_names_to_2lines:
            prices.columns = [x.replace(' ', '\n') for x in prices.columns]
            prices.columns = [x.replace('+', '\n+') for x in prices.columns]
            benchmark = benchmark.replace(' ', '\n')

        fig = qis.plot_regime_data(regime_classifier=self.regime_classifier,
                                   prices=prices,
                                   benchmark=benchmark,
                                   is_conditional_sharpe=True,
                                   regime_data_to_plot=regime_data_to_plot,
                                   var_format=var_format or '{:.2f}',
                                   perf_params=self.perf_params,
                                   title=title,
                                   ax=ax,
                                   **kwargs)
        return fig

    def plot_corr_matrix(self,
                         var_format: str = '{:.0%}',
                         freq: Optional[str] = None,
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> plt.Figure:
        """Correlation table of the overlay returns.

        Args:
            var_format: format of the correlations
            freq: sampling frequency of the returns
            ax: axis to draw on; None creates a figure
            **kwargs: passed to ``qis.plot_returns_corr_table``

        Returns:
            the new figure, or None when ``ax`` is given
        """
        if ax is None:  # create new axis
            height = calc_table_height(num_rows=len(self.overlay_navs.columns),
                                       first_row_height=2.0)
            fig, ax = plt.subplots(1, 1, figsize=(height, height))
        else:
            fig = None
        qis.plot_returns_corr_table(prices=self.overlay_navs,
                                    freq=freq,
                                    var_format=var_format,
                                    ax=ax,
                                    **kwargs)
        return fig

    def plot_smart_diversification_curve(self,
                                         principal_nav: pd.Series = None,
                                         principal_weight: float = 1.0,
                                         is_principal_weight_fixed: bool = True,
                                         rebalancing_freq: str = 'QE',
                                         x_var: PerfStat = PerfStat.BEAR_SHARPE,
                                         y_var: PerfStat = PerfStat.SHARPE_RF0,
                                         xlabel: str = None,
                                         ylabel: str = None,
                                         title: str = None,
                                         constraints: Dict[str, float] = None,
                                         shifts: Dict[str, bool] = None,
                                         shifts_n: Dict[str, float] = None,
                                         with_filter: List[str] = None,
                                         x_limits: Tuple[Optional[float], Optional[float]] = None,
                                         y_limits: Tuple[Optional[float], Optional[float]] = None,
                                         ax: plt.Subplot = None,
                                         **kwargs
                                         ) -> plt.Figure:
        """One curve per overlay through its eleven principal-overlay mixes.

        Args:
            principal_nav: nav of the principal portfolio; None uses the report's
            principal_weight: weight of the principal portfolio in every mix
            is_principal_weight_fixed: see ``compute_smart_diversification_curve``
            rebalancing_freq: rebalancing frequency of the mixes
            x_var: statistic of the x axis
            y_var: statistic of the y axis, different from ``x_var``
            xlabel: x-axis label; None uses the statistic name
            ylabel: y-axis label; None uses the statistic name
            title: figure title
            constraints: optional maximum overlay weight by overlay, 1.0 otherwise
            shifts: optional label placement by name, True above and False below
            shifts_n: overlays whose end label gets a line break
            with_filter: overlays whose curve is smoothed by a cubic fit
            x_limits: x-axis limits
            y_limits: y-axis limits
            ax: axis to draw on; None creates a figure
            **kwargs: passed to the statistic formats and to ``qis.plot_lines_list``

        Returns:
            the figure, the one of ``ax`` when given

        Raises:
            ValueError: if ``x_var`` and ``y_var`` are the same statistic
        """
        if y_var.to_str() == x_var.to_str():
            raise ValueError('x_var and y_var cannot be the same')

        if principal_nav is None:
            principal_nav = self.principal_nav

        xy_datas = {}
        data_labels = []
        y_0 = None
        for idx, asset in enumerate(self.overlay_navs.columns):
            max_overlay_weight = 1.0
            if constraints is not None:
                if asset in constraints.keys():
                    max_overlay_weight = constraints[asset]
            name = asset
            if shifts_n is not None:
                if asset in shifts_n.keys():
                    name = asset + '\n'

            xy = self.compute_smart_diversification_curve(principal_nav=principal_nav,
                                                          overlay_nav=self.overlay_navs[asset],
                                                          principal_weight=principal_weight,
                                                          is_principal_weight_fixed=is_principal_weight_fixed,
                                                          max_overlay_weight=max_overlay_weight,
                                                          rebalancing_freq=rebalancing_freq,
                                                          x_var=x_var,
                                                          y_var=y_var)
            if with_filter is not None and asset in with_filter:
                y = xy[y_var.to_str()].to_numpy()
                x = xy[x_var.to_str()].to_numpy()
                # replace x with smooth linear grid
                x_lin = np.linspace(start=x[0], stop=x[-1], num=len(x))
                poly_coeffs = safe_polyfit(x_lin, y, 3)
                y_smooth = np.poly1d(poly_coeffs)(x_lin)
                # Adjust to match endpoint
                if y_0 is None:
                    y_0 = y[0]
                y_smooth += np.linspace(0, y_0 - y_smooth[0], len(x_lin))
                y_smooth[0] = y_0

                xy[y_var.to_str()] = y_smooth
                xy[x_var.to_str()] = x_lin

            portfolio_labels = ['' for _ in xy.index]
            portfolio_labels[0] = f"{principal_nav.name}\n{'{:.0%}'.format(principal_weight)}"
            portfolio_labels[-1] = f"{name}  {'{:.0%}'.format(max_overlay_weight)}"
            if shifts is not None:
                if asset in shifts.keys():
                    if shifts[asset]:  # up
                        portfolio_labels[-1] = (name + ' ' + '{:.0%}'.format(max_overlay_weight)
                                                + '\n')
                    else: #down
                        portfolio_labels[-1] = (' \n' + name + ' '
                                                + '{:.0%}'.format(max_overlay_weight))

                if principal_nav.name in shifts.keys():
                    principal_share = '{:.0%}'.format(principal_weight)
                    portfolio_labels[0] = f"{principal_nav.name} {principal_share} \n"

            xy_datas[asset] = xy
            data_labels.append(portfolio_labels)

        markers = get_n_markers(n=len(xy_datas.keys()))
        xlabel = xlabel or x_var.to_str()
        ylabel = ylabel or y_var.to_str()

        fig = qis.plot_lines_list(xy_datas=xy_datas,
                                  data_labels=data_labels,
                                  title=title,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  xvar_format=x_var.to_format(**kwargs),
                                  yvar_format=y_var.to_format(**kwargs),
                                  markers=markers,
                                  x_limits=x_limits,
                                  y_limits=y_limits,
                                  ax=ax,
                                  **kwargs)
        return fig

    def plot_smart_diversification_scatter(self,
                                           principal_nav: pd.Series = None,
                                           is_drop_principal: bool = False,
                                           x_var: PerfStat = PerfStat.BEAR_SHARPE,
                                           y_var: PerfStat = PerfStat.SHARPE_RF0,
                                           xvar_format: str = '{:.1f}',
                                           yvar_format: str = '{:.1f}',
                                           shifts: Dict[str, bool] = None,
                                           xlabel: Union[str, bool, None] = True,
                                           ylabel: Union[str, bool, None] = True,
                                           is_add_model_equation: bool = True,
                                           annotation_labels: List[str] = None,
                                           portfolio_curve: List[str] = None,
                                           ci: int = 95,
                                           ax: plt.Subplot = None,
                                           **kwargs
                                           ) -> plt.Figure:
        """One point per standalone overlay and the principal, with a cross-sectional fit.

        Args:
            principal_nav: nav of the principal portfolio; None uses the report's
            is_drop_principal: leave the principal portfolio out
            x_var: statistic of the x axis
            y_var: statistic of the y axis
            xvar_format: format of the x values
            yvar_format: format of the y values
            shifts: optional label placement by name, True above and False below
            xlabel: x-axis label; True uses the statistic name
            ylabel: y-axis label; True uses the statistic name
            is_add_model_equation: show the equation of the fit
            annotation_labels: point labels; None uses the names
            portfolio_curve: names of points joined by a quadratic fit
            ci: confidence level of the fit band, in percent
            ax: axis to draw on; None creates a figure
            **kwargs: passed to ``qis.plot_scatter``

        Returns:
            the figure, the one of ``ax`` when given
        """
        if principal_nav is None:
            principal_nav = self.principal_nav

        overlay_points = self.get_overlay_points(principal_nav=principal_nav,
                                                 x_var=x_var, y_var=y_var)
        if is_drop_principal:
            overlay_points = overlay_points.drop(index=principal_nav.name)

        if annotation_labels is None:
            if shifts is not None:
                annotation_labels = []
                for asset in overlay_points.index.to_list():
                    annotation_label = asset
                    if asset in shifts.keys():
                        if shifts[asset]:
                            annotation_label = ' \n' + asset + '\n '
                        else:
                            annotation_label = asset + ' \n'
                    annotation_labels.append(annotation_label)
            else:
                annotation_labels = overlay_points.index.to_list()

        fig = qis.plot_scatter(df=overlay_points,
                               x=x_var.to_str(),
                               y=y_var.to_str(),
                               xvar_format=xvar_format,
                               yvar_format=yvar_format,
                               ax=ax,
                               annotation_labels=annotation_labels,
                               add_universe_model_label=is_add_model_equation,
                               xlabel=xlabel,
                               ylabel=ylabel,
                               colors=['#00284A'],
                               full_sample_label='Cross-sectional fit: ',
                               ci=ci,
                               **kwargs)

        if portfolio_curve is not None:
            is_on_curve = np.isin(overlay_points.index, portfolio_curve, assume_unique=True)
            curve = overlay_points.loc[is_on_curve, :]
            y = curve[y_var.to_str()].to_numpy()
            x = curve[x_var.to_str()].to_numpy()
            try:
                curve_poly = np.polyfit(x, y, 2)  # fit 2-order polynomial to x and y curve
                # make linear space between x_min and x_max
                x_lin = np.linspace(np.min(x), np.max(x), 50)
                y_curve = np.poly1d(curve_poly)(x_lin)  # fill in
                sns.lineplot(x=x_lin, y=y_curve, marker='None', color='blue', ax=ax)
            except np.linalg.LinAlgError:
                warnings.warn("SVD did not converge in Linear Least Squares")

        return fig

    def plot_returns_scatter(self,
                             principal_nav: pd.Series = None,
                             overlay_navs: pd.DataFrame = None,
                             add_45line: bool = False,
                             title: Union[str, None] = None,
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> plt.Figure:
        """Quarterly returns of each overlay against the principal portfolio.

        Args:
            principal_nav: nav of the principal portfolio; None uses the report's
            overlay_navs: navs of the overlays; None uses the report's
            add_45line: draw the 45-degree line
            title: figure title
            ax: axis to draw on; None creates a figure
            **kwargs: passed to ``qis.plot_returns_scatter``

        Returns:
            the figure, the one of ``ax`` when given
        """
        if principal_nav is None:
            principal_nav = self.principal_nav
        if overlay_navs is None:
            overlay_navs = self.overlay_navs
        prices = pd.concat([principal_nav, overlay_navs], axis=1, sort=True)
        fig = qis.plot_returns_scatter(prices=prices,
                                       benchmark=str(principal_nav.name),
                                       add_45line=add_45line,
                                       #ylabel=f"{self.overlay_nav.name}",
                                       title=title,
                                       freq='QE',
                                       order=2,
                                       ci=95,
                                       ax=ax,
                                       **kwargs)
        return fig


def safe_polyfit(x, y, degree=3) -> np.ndarray:
    """Polynomial coefficients of y on x, lowering the degree when the fit fails.

    Args:
        x: abscissas; pairs with a missing x or y are dropped
        y: ordinates
        degree: requested degree, capped at the number of clean points less one

    Returns:
        coefficients from the highest power, or the mean of y when every degree fails
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < degree + 1:
        # Not enough points for requested degree
        degree = max(1, len(x_clean) - 1)
    """
    # Normalize x for numerical stability
    x_mean, x_std = np.mean(x_clean), np.std(x_clean)
    if x_std > 0:
        x_norm = (x_clean - x_mean) / x_std
    else:
        x_norm = x_clean - x_mean
    """
    # Try fitting with decreasing polynomial degrees
    for deg in range(degree, 0, -1):
        try:
            poly = np.polyfit(x_clean, y_clean, deg)
            return poly
        except np.linalg.LinAlgError:
            warnings.warn("SVD did not converge in Linear Least Squares")
            continue

    # Ultimate fallback: return mean
    return np.array([np.mean(y_clean)])
