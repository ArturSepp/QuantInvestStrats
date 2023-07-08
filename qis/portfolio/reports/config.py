from qis import PerfParams, BenchmarkReturnsQuantileRegimeSpecs
import yfinance as yf

RISK_FREE_RATE = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0

PERF_PARAMS = PerfParams(freq='W-WED', freq_reg='W-WED', rates_data=RISK_FREE_RATE)
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='Q')


# use for number years > 5
KWARG_LONG = dict(perf_params=PerfParams(freq='W-WED', freq_reg='Q', rates_data=RISK_FREE_RATE),
                  regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='Q'),
                  x_date_freq='A',
                  date_format='%b-%y')

# use for number years < 3
KWARG_SHORT = dict(perf_params=PerfParams(freq='W-WED', freq_reg='M', rates_data=RISK_FREE_RATE),
                   regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='M'),
                   x_date_freq='Q',
                   date_format='%b-%y')

