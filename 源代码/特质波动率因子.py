import pandas as pd
import numpy as np  
# # 读取数据
# returns = pd.read_csv('raw_data/hs300_stock_returns.csv')
# fama = pd.read_csv('factors/Fama因子_filtered.csv')

# # 将日期列转换为datetime格式
# returns['日期'] = pd.to_datetime(returns['日期'])
# fama['日期'] = pd.to_datetime(fama['日期'])

# # 按日期合并数据
# merged_data = pd.merge(returns, fama, on='日期', how='inner')
# merged_data = merged_data.sort_values('日期').reset_index(drop=True)

# # 创建一个与returns相同结构的DataFrame来存储特质波动率
# stock_columns = [col for col in returns.columns if col != '日期']
# idio_vol = pd.DataFrame(index=merged_data.index, columns=['日期'] + stock_columns)
# idio_vol['日期'] = merged_data['日期']

# # 对每只股票计算特质波动率
# for stock in stock_columns:
#     residuals = []
    
#     # 对每个时间点计算过去60天的特质波动率
#     for i in range(60, len(merged_data)):
#         # 获取过去60天的数据
#         window_data = merged_data.iloc[i-60:i]
#         stock_returns = window_data[stock]
#         factors = window_data[['MKT', 'SMB', 'HML']]
        
#         try:
#             # 去除任何包含NaN的行
#             valid_mask = ~(stock_returns.isna() | factors.isna().any(axis=1))
#             stock_returns = stock_returns[valid_mask]
#             factors = factors[valid_mask]
            
#             if len(stock_returns) < 30:  # 如果有效数据少于30天，跳过
#                 residuals.append(np.nan)
#                 continue
            
#             # 进行回归
#             X = np.c_[np.ones(len(factors)), factors]
#             y = stock_returns
#             beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
#             # 计算残差
#             y_pred = X @ beta
#             residual = y - y_pred
            
#             # 计算残差的标准差作为特质波动率
#             idio_volatility = np.std(residual)
#             residuals.append(idio_volatility)
#         except:
#             residuals.append(np.nan)
    
#     # 添加前60天的NaN值
#     residuals = [np.nan] * 60 + residuals
#     idio_vol[stock] = residuals


# # 保存结果
# idio_vol.to_csv('factors/RVOL.csv', index=False)
# 读取RVOL.csv
rvol = pd.read_csv('factors/RVOL.csv')

# 删除前60行数据
rvol = rvol.iloc[60:].reset_index(drop=True)

# 打印数据形状
print("数据形状:", rvol.shape)

# 检查是否存在缺失值
print("\n各列缺失值数量:")
print(rvol.isna().sum())

# 打印总的缺失值数量
print("\n总缺失值数量:", rvol.isna().sum().sum())