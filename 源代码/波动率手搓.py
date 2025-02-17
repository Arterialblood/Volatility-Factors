import pandas as pd
import numpy as np
import os  # 添加os模块用于创建文件夹

# 读取数据
index_returns = pd.read_csv('raw_data/hs300_index_returns.csv')
index_returns['日期'] = pd.to_datetime(index_returns['日期'])
index_returns = index_returns.set_index('日期')

stock_returns = pd.read_csv('raw_data/hs300_stock_returns.csv') 
stock_returns['日期'] = pd.to_datetime(stock_returns['日期'])
stock_returns = stock_returns.set_index('日期')

# 删除空值
index_returns = index_returns.dropna()
stock_returns = stock_returns.dropna(how='all')
stock_returns = stock_returns.dropna(axis=1, how='all')

# 计算VOL_3M因子(60天滚动窗口标准差)
def calculate_vol_3m(returns):
    return returns.rolling(window=60).std()

# 使用更高效的方式计算VOL_3M
vol_3m = stock_returns.apply(calculate_vol_3m)

# 创建factors文件夹(如果不存在)
if not os.path.exists('factors'):
    os.makedirs('factors')

# 打印结果信息
print("VOL_3M因子计算完成")
print("因子数据形状:", vol_3m.shape)
print("\n前5行数据预览:")
print(vol_3m.head())

# 保存结果
vol_3m.to_csv('factors/vol_3m.csv')
print("\n因子数据已保存至 'factors/vol_3m.csv'")

# VOL_3M因子数据预处理
vol_3m = pd.read_csv('factors/vol_3m.csv')
vol_3m['日期'] = pd.to_datetime(vol_3m['日期'])
vol_3m = vol_3m.set_index('日期')

print("处理前:")
print("数据形状:", vol_3m.shape)
print("空值数量:", vol_3m.isna().sum().sum())

vol_3m = vol_3m.dropna(how='all', axis=0)
vol_3m = vol_3m.dropna(how='all', axis=1)

print("\n处理后:")
print("数据形状:", vol_3m.shape)
print("空值数量:", vol_3m.isna().sum().sum())

vol_3m.to_csv('factors/vol_3m_processed.csv')
print("\n处理后的数据已保存至 factors/vol_3m_processed.csv")

# 计算RANKVOL因子
# 对每日收益率计算横截面分位数
daily_ranks = stock_returns.rank(axis=1, pct=True)

# 计算60天滚动窗口的分位数标准差
rankvol = daily_ranks.rolling(window=60).std()

# 打印RANKVOL因子信息
print("\nRANKVOL因子计算完成")
print("因子数据形状:", rankvol.shape)
print("\n前5行数据预览:")
print(rankvol.head())

# 保存RANKVOL因子
rankvol.to_csv('factors/rankvol.csv')
print("\nRANKVOL因子数据已保存至 'factors/rankvol.csv'")

# RANKVOL因子数据预处理
rankvol = pd.read_csv('factors/rankvol.csv')
rankvol['日期'] = pd.to_datetime(rankvol['日期'])
rankvol = rankvol.set_index('日期')

print("\nRANKVOL处理前:")
print("数据形状:", rankvol.shape)
print("空值数量:", rankvol.isna().sum().sum())

rankvol = rankvol.dropna(how='all', axis=0)
rankvol = rankvol.dropna(how='all', axis=1)

print("\nRANKVOL处理后:")
print("数据形状:", rankvol.shape)
print("空值数量:", rankvol.isna().sum().sum())

rankvol.to_csv('factors/rankvol_processed.csv')
print("\n处理后的数据已保存至 factors/rankvol_processed.csv")
