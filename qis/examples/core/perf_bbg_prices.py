# packages
import matplotlib.pyplot as plt
import qis as qis
from enum import Enum
from bbg_fetch import fetch_field_timeseries_per_tickers


def run_report():
    # tickers = {'TY1 Comdty': '10y', 'UXY1 Comdty': '10y Ultra'}
    tickers = {
        'SPTR Index': 'SPTR Index',
        'CIEQVEHG Index': 'Citi SPX 0D Vol Carry',
        'CIEQVRUG Index': 'Citi SX5E 1W Vol Carry',
        'CICXCOSE  Index': 'Citi Brent Vol Carry',
        'GSISXC07 Index': 'GS Multi Asset Carry',
        'GSISXC11 Index': 'GS Macro Carry',
        'XUBSPGRA Index': 'UBS Gold Strangles',
        'XUBSU1D1 Index': 'UBS Short Vol Daily',
        #'BCKTARU2 Index': 'BNP Call on Short-vol Carry'
        }

    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    print(prices)

    time_period = qis.TimePeriod('31Dec2019', '07Nov2024')
    kwargs = qis.fetch_default_report_kwargs(time_period=time_period)

    fig = qis.generate_multi_asset_factsheet(prices=prices,
                                             benchmark='SPTR Index',
                                             time_period=time_period,
                                             **kwargs)
    qis.save_figs_to_pdf(figs=[fig],
                         file_name=f"bbg_multiasset_report", orientation='landscape',
                         # local_path='C://Users//uarts//outputs//',
                         local_path=qis.get_output_path()
                         )


def run_price():
    tickers = {'CL1 Comdty': 'WTI'}
    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    print(prices)

    time_period = qis.TimePeriod('31Dec1989', '08Nov2024')
    prices = time_period.locate(prices)

    qis.plot_prices_with_dd(prices,
                            start_to_one=False)


class UnitTests(Enum):
    REPORT = 1
    PRICE = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.REPORT:
        run_report()

    elif unit_test == UnitTests.PRICE:
        run_price()

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PRICE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)



plt.show()
