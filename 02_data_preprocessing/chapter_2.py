# %%
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")
import nasdaqdatalink as ndl
import yfinance as yf
from forex_python.converter import CurrencyRates
from binance.spot import Spot as Client
import cpi

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# feel free to modify, for example, change the context to "notebook"
sns.set_theme(context="talk", style="whitegrid", 
              palette="colorblind", color_codes=True, 
              rc={"figure.figsize": [12, 8]})

# %% [markdown]
# 

# %% [markdown]
# # Chapter 2 - Data Preprocessing
# ## 2.1 Converting prices to returns

# %% [markdown]
# Download the data and keep the adjusted close prices only:

# %%
df = yf.download("AAPL", 
                 start="2011-01-01", 
                 end="2021-12-31",
                 progress=False)
# Book
# df = df.loc[:, ["Adj Close"]]
# error: "Adj Close" is not available in the downloaded data
# AI: In the past, Yahoo Finance provided both Close and Adj Close columns. However, yfinance now simplifies this by including only the adjusted close price under the Close column.
df = df.loc[:, ["Close"]]
df.columns = df.columns.get_level_values(0)
df

# %% [markdown]
# Convert adjusted close prices to simple and log returns:

# %%
df["simple_rtn"] = df["Close"].pct_change()
df["log_rtn"] = np.log(df["Close"]/df["Close"].shift(1))

# %% [markdown]
# Inspect the output:
# 

# %%
df.head()

# %% [markdown]
# ## 2.2 Adjusting the returns for inflation

# %%
df = yf.download("AAPL", 
                 start="2011-01-01", 
                 end="2021-12-31",
                 progress=False)
# Book
# df = df.loc[:, ["Adj Close"]]
# error: "Adj Close" is not available in the downloaded data
# AI: In the past, Yahoo Finance provided both Close and Adj Close columns. However, yfinance now simplifies this by including only the adjusted close price under the Close column.
df = df.loc[:, ["Close"]]
df.columns = df.columns.get_level_values(0)
df

# %%
df.head()

# %%
ndl.ApiConfig.api_key = os.environ.get("NASDAQ_API_KEY")

# %%
# Resample daily prices to monthly:
df = df.resample("M").last()
df

# %%
# Download inflation data from Nasdaq Data Link:
# df_cpi = (
#     ndl.get(dataset="RATEINF/CPI_USA", 
#                        start_date="2009-12-01", 
#                        end_date="2020-12-31")
#     .rename(columns={"Value": "cpi"})
# )

# Alternatives for US CPI Data (No Nasdaq API Needed)
# 1. Use pandas_datareader for FRED CPI
# import pandas_datareader.data as web

# df_cpi = web.DataReader("CPIAUCSL", "fred", "2009-12-01", "2020-12-31")
# df_cpi = df_cpi.rename(columns={"CPIAUCSL": "cpi"})

# 2. Use the cpi Python package
# import cpi

cpi_series = cpi.series.get()
df_cpi = cpi_series.to_dataframe()
df_cpi = df_cpi.query("period_type == 'monthly' and year > 2009 and year < 2022")
df_cpi = df_cpi.set_index("date")[["value"]].rename(columns={"value": "cpi"})

# Make sure the index is datetime
df_cpi.index = pd.to_datetime(df_cpi.index)

# Convert CPI index to month end
df_cpi.index = df_cpi.index.to_period('M').to_timestamp('M')

df_cpi

# %%
# Join inflation data to prices:
df = df.join(df_cpi, how="left") 

# %%
# Calculate simple returns and inflation rate:
df["simple_rtn"] = df["Close"].pct_change()
df["inflation_rate"] = df["cpi"].pct_change()

# %%
# Adjust the returns for inflation:
df["real_rtn"] = (
    (df["simple_rtn"] + 1) / (df["inflation_rate"] + 1) - 1
)
df.head()

# %%
# Obtain the default CPI series:
cpi_series = cpi.series.get()
print(cpi_series)

# %%
# Convert the object into a pandas DataFrame:
df_cpi_2 = cpi_series.to_dataframe()


# %%
# Filter the DataFrame and view the top 12 observations:
df_cpi_2.query("period_type == 'monthly' and year >= 2010") \
        .loc[:, ["date", "value"]] \
        .set_index("date") \
        .head(12)

# %% [markdown]
# ## 2.3 Changing the frequency of time series data

# %%
# Obtain the log returns in case of starting in this recipe:
# download data 
df = yf.download("AAPL", 
                 start="2000-01-01", 
                 end="2010-12-31", 
                 auto_adjust=False,
                 progress=False)

# keep only the adjusted close price
df = df.loc[:, ["Adj Close"]] \
       .rename(columns={"Adj Close": "adj_close"})

# calculate simple returns
df["log_rtn"] = np.log(df["adj_close"]/df["adj_close"].shift(1))

# remove redundant data
df = df.drop("adj_close", axis=1) \
       .dropna(axis=0)

df.head()

# %%
# Define the function for calculating the realized volatility:
def realized_volatility(x):
    return np.sqrt(np.sum(x**2))

# %%
# Calculate monthly realized volatility:
df_rv = (
    df.groupby(pd.Grouper(freq="M"))
    .apply(realized_volatility)
    .rename(columns={"log_rtn": "rv"})
)

# %%
# Annualize the values:
df_rv.rv = df_rv["rv"] * np.sqrt(12)


# %%
# Plot the results:
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df)
ax[0].set_title("Apple's log returns (2000-2012)")
ax[1].plot(df_rv)
ax[1].set_title("Annualized realized volatility")

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_2_5', dpi=200)

# %% [markdown]
# ## 2.4 Different ways of imputing missing data

# %%
# # Download the inflation data from Nasdaq Data Link:
# nasdaqdatalink.ApiConfig.api_key = "YOUR_KEY_HERE" 

# df = (
#     nasdaqdatalink.get(dataset="RATEINF/CPI_USA", 
#                        start_date="2015-01-01", 
#                        end_date="2020-12-31")
#     .rename(columns={"Value": "cpi"})
# )

cpi_series = cpi.series.get()
df = cpi_series.to_dataframe()
df = df.query("period_type == 'monthly' and year > 2015 and year < 2021")
df = df.set_index("date")[["value"]].rename(columns={"value": "cpi"})

# Make sure the index is datetime
df.index = pd.to_datetime(df.index)

# Convert CPI index to month end
df.index = df.index.to_period('M').to_timestamp('M')

df

# %%
# Introduce 5 missing values at random:
np.random.seed(42)
rand_indices = np.random.choice(df.index, 5, replace=False)

df["cpi_missing"] = df.loc[:, "cpi"]
df.loc[rand_indices, "cpi_missing"] = np.nan
df.head()

# %%
# Fill the missing values using different methods:
for method in ["bfill", "ffill"]:
    df[f"method_{method}"] = (
        df[["cpi_missing"]].fillna(method=method)
    )

# %%
# Inspect the results by displaying the rows in which we created the missing values:
df.loc[rand_indices].sort_index()

# %%
# Plot the results for years 2015-2016:
df.loc[:"2017-01-01"] \
  .drop(columns=["cpi_missing"]) \
  .plot(title="Different ways of filling missing values");

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_2_8', dpi=200)

# %%
# Use linear interpolation to fill the missing values:
df["method_interpolate"] = df[["cpi_missing"]].interpolate()


# %%
# Inspect the results:
df.loc[rand_indices].sort_index()


# %%
# Plot the results:
df.loc[:"2017-01-01"] \
  .drop(columns=["cpi_missing"]) \
  .plot(title="Different ways of filling missing values");

sns.despine()
plt.tight_layout()
# plt.savefig('images/figure_2_10', dpi=200)

# %% [markdown]
# ## 2.5 Converting currencies
# 

# %%
# Download Apple's OHLC prices from January 2020:
df = yf.download("AAPL", 
                 start="2020-01-01", 
                 end="2020-01-31",
                 progress=False)
df = df.drop(columns=["Close", "Volume"])

# %%
# Instantiate the CurrencyRates object:
c = CurrencyRates()

# %%
# Download the USD/EUR rate for each required date:
df["usd_eur"] = [c.get_rate("USD", "EUR", date) for date in df.index]

# %%
# Convert the prices in USD to EUR:
for column in df.columns[:-1]:
    df[f"{column}_EUR"] = df[column] * df["usd_eur"]
df.head().round(3)

# %%
# Get the USD exchange rates to 31 available currencies:
usd_rates = c.get_rates("USD")
usd_rates

# %%
len(usd_rates)

# %%
# Download the USD/EUR exchange rate from Yahoo Finance:
df = yf.download("USDEUR=X", 
                 start="2000-01-01", 
                 end="2010-12-31",
                 progress=False)
df.head()

# %% [markdown]
# ## 2.6 Different ways of aggregating trade data

# %%
# Instantiate the Binance client and download the last 500 BTCEUR trades:
spot_client = Client(base_url="https://api3.binance.com")
r = spot_client.trades("BTCEUR")

# %%
# Process the downloaded trades into a pandas DataFrame:
df = (
    pd.DataFrame(r)
    .drop(columns=["isBuyerMaker", "isBestMatch"])
)
df["time"] = pd.to_datetime(df["time"], unit="ms")

for column in ["price", "qty", "quoteQty"]:
    df[column] = pd.to_numeric(df[column])
df

# %%
# Define a function aggregating the raw trades information:
def get_bars(df, add_time=False):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    ohlc = df["price"].ohlc()
    vwap = (
        df.apply(lambda x: np.average(x["price"], weights=x["qty"]))
        .to_frame("vwap")
    )
    vol = df["qty"].sum().to_frame("vol")
    cnt = df["qty"].size().to_frame("cnt")
    
    if add_time:
        time = df["time"].last().to_frame("time")
        res = pd.concat([time, ohlc, vwap, vol, cnt], axis=1)
    else:
        res = pd.concat([ohlc, vwap, vol, cnt], axis=1)
    return res

# %%
# Get time bars:
df_grouped_time = df.groupby(pd.Grouper(key="time", freq="1Min"))
time_bars = get_bars(df_grouped_time)
time_bars

# %%
# Get tick bars:
bar_size = 50 
df["tick_group"] = (
    pd.Series(list(range(len(df))))
    .div(bar_size)
    .apply(np.floor)
    .astype(int)
    .values
)
df_grouped_ticks = df.groupby("tick_group")
tick_bars = get_bars(df_grouped_ticks, add_time=True)
tick_bars

# %%
# Get volume bars:
bar_size = 1 
df["cum_qty"] = df["qty"].cumsum()
df["vol_group"] = (
    df["cum_qty"]
    .div(bar_size)
    .apply(np.floor)
    .astype(int)
    .values
)
df_grouped_ticks = df.groupby("vol_group")
volume_bars = get_bars(df_grouped_ticks, add_time=True)
volume_bars

# %%
# Get dollar bars:
bar_size = 50000 
df["cum_value"] = df["quoteQty"].cumsum()
df["value_group"] = (
    df["cum_value"]
    .div(bar_size)
    .apply(np.floor)
    .astype(int)
    .values
)
df_grouped_ticks = df.groupby("value_group")
dollar_bars = get_bars(df_grouped_ticks, add_time=True)
dollar_bars

# %%
