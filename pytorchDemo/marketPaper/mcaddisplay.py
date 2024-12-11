import tushare as ts
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ochl
from matplotlib.pylab import date2num
import pandas as pd
import numpy as np

# 获取贵州茅台从2018年元旦至2023年3月20日的行情数据
df = ts.get_k_data('600519', start='2018-01-01', end='2023-03-20')

# 计算MACD指标
def MACD(df, fastperiod=12, slowperiod=26, signalperiod=9):
    ewma12 = pd.Series.ewm(df['close'], span=fastperiod).mean()
    ewma26 = pd.Series.ewm(df['close'], span=slowperiod).mean()
    dif = ewma12 - ewma26
    dea = pd.Series.ewm(dif, span=signalperiod).mean()
    macd = (dif - dea) * 2
    return dif, dea, macd

dif, dea, macd = MACD(df)

# 画图展示
df['time'] = pd.to_datetime(df['date'])
df['time'] = df['time'].apply(date2num)
df = df[['time', 'open', 'close', 'high', 'low']]
fig, ax = plt.subplots(figsize=(20, 10))
candlestick_ochl(ax, df.values, width=0.6, colorup='red', colordown='green', alpha=0.8)
plt.plot(dif, label='DIF', color='blue')
plt.plot(dea, label='DEA', color='orange')
plt.bar(macd.index, macd, label='MACD', color='purple')
plt.legend()
plt.grid()
plt.show()
