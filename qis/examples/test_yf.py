import yfinance as yf

tickers = ['SPY']
prices = yf.download(tickers, start=None, end=None, period='max', interval='1d')
print(prices)
