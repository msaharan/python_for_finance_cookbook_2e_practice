#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")
import os

import warnings
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

sns.set_theme(context="talk", style="whitegrid", 
              palette="colorblind", color_codes=True,
              rc = {"figure.figsize": (12, 8)})
#%%

# Chapter 1: Acquiring Financial Data
# 1.1 Getting data from Yahoo Finance
import yfinance as yf

# Download method 1
df = yf.download("AAPL", start="2010-01-01", end="2023-10-01", progress=False)
df.head()

# Download method 2
aapl_data = yf.Ticker("AAPL")
df = aapl_data.history(start="2010-01-01", end="2023-10-01")

df.head()
print(aapl_data.info)
print(aapl_data.actions)
print(aapl_data.financials)
print(aapl_data.quarterly_financials)
print(aapl_data.earnings)
print(aapl_data.quarterly_earnings)
print(aapl_data.calendar)
# %%
# 1.2 Getting data from Nasdaq Data Link
# import nasdaqdatalink as ndl

# ndl.ApiConfig.api_key = os.environ.get("NASDAQ_API_KEY")

# df = ndl.get(dataset="WIKI/AAPL",
#                         start_date="2011-01-01", 
#                         end_date="2021-12-31")

# search_results = ndl.Dataset.find('AAPL')
# print(search_results)
# does not work anymore, as WIKI dataset is deprecated
# %%
# 1.3 Getting data from Intrinio
# import intrinio_sdk as intrinio

# intrinio.ApiClient().set_api_key(os.environ.get("INTRINIO_API_KEY"))
# security_api = intrinio.SecurityApi() 

# r = security_api.get_security_stock_prices(
#     identifier="AAPL", 
#     start_date="2011-01-01",
#     end_date="2021-12-31", 
#     frequency="daily",
#     page_size=10000
# )
# Error: ApiException: (401), Reason: Unauthorized
# %%
# 1.4 Getting data from Alpha Vantage
from alpha_vantage.cryptocurrencies import CryptoCurrencies
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

crypto_api = CryptoCurrencies(key=ALPHA_VANTAGE_API_KEY, 
                              output_format="pandas")

data, meta_data = crypto_api.get_digital_currency_daily(
    symbol="BTC", 
    market="EUR"
)
#%%
meta_data
# %%
data
# %%
crypto_api.get_digital_currency_exchange_rate(
    from_currency="BTC", 
    to_currency="USD"
)[0].transpose()
# %%
import requests
from io import BytesIO
#%%
AV_API_URL = "https://www.alphavantage.co/query"
parameters = {
    "function": "CRYPTO_INTRADAY",
    "symbol": "ETH",
    "market": "USD",
    "interval": "30min",
    "outputsize": "full",
    "apikey": ALPHA_VANTAGE_API_KEY
}
r = requests.get(AV_API_URL, params=parameters)
data = r.json()
print(data)
# {'Information': 'Thank you for using Alpha Vantage! This is a premium endpoint. You may subscribe to any of the premium plans at https://www.alphavantage.co/premium/ to instantly unlock all premium endpoints'}
#%%
# Get Tesla's earnings
AV_API_URL = "https://www.alphavantage.co/query" 
parameters = {
    "function": "EARNINGS",
    "symbol": "TSLA", 
    "apikey": ALPHA_VANTAGE_API_KEY
}
r = requests.get(AV_API_URL,  params=parameters)
data = r.json()
print(data.keys())
# %%

# %%
df = pd.DataFrame(data["quarterlyEarnings"])
df.head()
# %%
# download the upcoming IPOs 
import csv

AV_API_URL = "https://www.alphavantage.co/query" 
parameters = {
    "function": "IPO_CALENDAR",
    "apikey": ALPHA_VANTAGE_API_KEY
}

with requests.Session() as s:
    download = s.get(AV_API_URL, params=parameters)
    decoded_content = download.content.decode("utf-8")
    ipos_list = list(csv.reader(decoded_content.splitlines(), delimiter=","))

df = pd.DataFrame(ipos_list[1:], columns=ipos_list[0])
df
# %%
# Download Google's stock prices using the `TimeSeries` module
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
data, _ = ts.get_daily("GOOGL", outputsize="full")
print(f"Downloaded {len(data)} observations")
data
# %%
# Download intraday prices (works)
data, _ = ts.get_intraday("GOOGL", interval="5min", outputsize="full")
data
# %%
# 1.5 Geting data from CoinGecko

from pycoingecko import CoinGeckoAPI
from datetime import datetime
# %%
cg = CoinGeckoAPI()

# %%
# Get Bitcoin's OHLC prices from the last 14 days
ohlc = cg.get_coin_ohlc_by_id(
    id="bitcoin", vs_currency="usd", days="14"
)
ohlc_df = pd.DataFrame(ohlc)
ohlc_df.columns = ["date", "open", "high", "low", "close"]
ohlc_df["date"] = pd.to_datetime(ohlc_df["date"], unit="ms")
ohlc_df
# %%
# Get the top 7 trending coins (based on the # of searches in the last 24h) from CoinGecko:
trending_coins = cg.get_search_trending()
(
    pd.DataFrame([coin["item"] for coin in trending_coins["coins"]])
    .drop(columns=["thumb", "small", "large"])
)
# %%
# Get Bitcoin's current price in USD:
cg.get_price(ids="bitcoin", vs_currencies="usd")

# %%
# Get current prices of ETH, BTC in USD and EUR:
cg.get_price(ids=["ethereum", "bitcoin"], vs_currencies=["usd", "eur"])
# %%
# Get the current BTC/USD eschange rate, market capitalization, 24h volumne and change and the last-updated timestamp:
cg.get_price(ids="bitcoin", vs_currencies="usd", 
             include_market_cap=True, 
             include_24hr_vol=True, 
             include_24hr_change=True, 
             include_last_updated_at=True)
# %%
# Get the list of all supported coin ids, together with their name and symbol:
pd.DataFrame(cg.get_coins_list())

# %%
# Get all the coins market data:
pd.DataFrame(cg.get_coins_markets(vs_currency="eur"))

# %%
# Get all the supported crypto exchanges:
exchanges_df = pd.DataFrame(cg.get_exchanges_list(per_page=250))
exchanges_df.head()

# %%
# Get a summary of DEFI:
cg.get_global_decentralized_finance_defi()

# %%