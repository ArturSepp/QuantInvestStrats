# packages
import pandas as pd
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
        'BCKTARU2 Index': 'BNP Call on Short-vol Carry',
        'BNPXAUUS Index': 'BNP Intraday SPX Vol Carry',
        'BNPXAUTS Index': 'BNP Intraday NDX Vol Carry',
        'BNPXOV3U Index': 'BNP 3M Long DHhedged Puts'
        }

    benchmark = 'HYG US Equity'
    tickers = {
        benchmark: benchmark,
        'NMVVR1EL Index': 'IRVING1 EUR',
        'NMVVR1UL Index': 'IRVING1 USD',
        'NMVVR1L Index': 'IRVING1',
        'BNPXLVRE Index': 'BNP Long Rates Vol EUR',
        'BNPXLVRU Index': 'BNP Long Rates Vol USD',
        'BXIIULSV Index': 'Barclays Long Rates Vol',
        'BXIIUGNT Index': 'Barclays Gamma Neutral Vol',
        'BXIIUENT Index': 'Barclays Triangle Vol'
    }
    
    benchmark = 'SPTR Index'
    tickers = {
        benchmark: benchmark,
        'BNPIV1EE Index': 'BNP Europe 1Y Volatility',
        'BNPIV1UE Index': 'BNP US 1Y Volatility',
        'BNPXVO3A Index': 'BNP VOLA 3 Index',
        'AIJPVT1U Index': 'JPM Volatility Trend Following',
        'JPOSLVUS Index': 'JPM US Long Variance',
        'JPOSPRU2 Index': 'JPM US Put Ratio',
        'JPOSTUDN Index': 'JPM US Equity Tail Hedge',
        'JPRC85BE Index': 'JPM Dynamic 85% Rolling Collar EU',
        'JPRC85BU Index': 'JPM Dynamic 85% Rolling Collar US',
        'JPUSVXCR Index': 'JPM US Volatility Call Ratio'
    }

    benchmark = 'XNDX Index'
    tickers = {
        benchmark: benchmark,
        'BNPXTHUE Index': 'Thalia',
        'BNPXTHUN Index': 'Thalia Neutral',
        'BNPXTDUE Index': 'Thalia Dynamic',
        'BNPXTDUN Index': 'Thalia Neutral Dynamic',
        'BNPXLVRU Index': 'BNP Long Rates Vol USD'
    }

    benchmark = 'SPTR Index'
    tickers = {
        benchmark: benchmark,
        'AIJPMT1U Index': 'JPM Macro Trend',
        'AIJPLT3U Index': 'JPM Cross Trend',
        'AIJPXSK1 Index': 'JPM XA Skeweness',
        'NEIXCTAT Index': 'SG Trend'
    }

    benchmark = 'SPTR Index'
    tickers = {
        benchmark: benchmark,
        'JPQGM4W1 Index': 'JPM Factor1',
        'JPQTR4W1 Index': 'JPM Factor2'
    }

    benchmark = 'SPTR Index'
    tickers = {
        benchmark: benchmark,
        'DBBNE05Y Index': 'DBBNE05Y',
        'DBBNE10Y Index': 'DBBNE10Y',
        'DBBNE15Y Index': 'DBBNE15Y',
        'DBBNU05Y Index': 'DBBNU05Y',
        'DBCUU10Y Index': 'DBCUU10Y',
        'DBBNU15Y Index': 'DBBNU15Y'
    }

    benchmark = 'SPTR Index'
    tickers = {
        benchmark: benchmark,
        'CICMCI5B Index': 'CDX IG Citi',
        'UISYMI5S Index': 'CDX IG UBS shortable',
        'DBCDIG5F Index': 'CDX IG DB long fixed',
        'DBCDIG5L Index': 'CDX IG DB long variable',
        'DBCDIG5S Index': 'CDX IG DB short',
        'CICMCH5B Index': 'CDX HY Citi',
        'UISYMH5S Index': 'CDX HY UBS shortable',
        'DBCDHYLG Index': 'CDX HY DB long fixed',
        'DBCDHY5A Index': 'CDX HY DB long variable',
        # 'DBCDHY5S Index': 'CDX HY DB short'
    }

    benchmark = 'AOR US Equity'
    tickers = {
        benchmark: benchmark,
        'HFRXGL Index': 'HFRXGL Index',
        'HFRIFWI Index': 'HFRIFWI Index',
        'GMSGRMU ID Equity': 'Graham',
        'WINTFUI ID Equity': 'Winton',
        'BHMG LN Equity': 'Brevan',
        'OMEIUSA ID Equity': 'Jupiter'
    }

    benchmark = 'SPTGGUT Index'
    tickers = {
        benchmark: benchmark,
        'NEIXCTAT Index': 'SG Trend',
        'NEIXCTA Index': 'SG CTA',
        'HFRIMTF Index': 'HFRI TF',
        'BHCTA Index': 'BBG CTA',
        'HFRXM Index': 'HFRX Macro/CTA'
    }

    benchmark = 'AOR US Equity'
    tickers = {
        benchmark: benchmark,
        'EHFI804 Index': 'ILS EHFI804 Index',
        'FUCBIAU ID Equity': 'Fermat Cat ID Fund',
        'FECABIA KY Equity': 'Fermat Cat KY Fund',
    }

    benchmark = 'AOR US Equity'
    tickers = {
        benchmark: benchmark,
        'NMVVR1EL Index': 'IRVING1 EUR',
        'NMVVR1UL Index': 'IRVING1 USD',
        'NMVVR1L Index': 'IRVING1',
    }


    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
    print(prices)
    # qis.save_df_to_csv(df=prices, file_name='qis_vol_indices', local_path=qis.get_output_path())

    time_period = qis.TimePeriod('07Nov2006', '31Dec2025')
    # kwargs = qis.fetch_default_report_kwargs(time_period=time_period, add_rates_data=False)
    # kwargs = qis.fetch_factsheet_config_kwargs(factsheet_config=qis.FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD, add_rates_data=False)
    kwargs = qis.fetch_factsheet_config_kwargs(factsheet_config=qis.FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD, add_rates_data=True)

    fig = qis.generate_multi_asset_factsheet(prices=prices,
                                             benchmark=benchmark,
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


class LocalTests(Enum):
    REPORT = 1
    PRICE = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


    if local_test == LocalTests.REPORT:
        run_report()

    elif local_test == LocalTests.PRICE:
        run_price()

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.REPORT)
