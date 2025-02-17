import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd

# 1. 处理Fama三因子数据
# 读取原始CSV文件
df = pd.read_csv('raw_data/STK_MKT_THRFACDAY.csv')

# 提取MarkettypeID为'P9707'的行
fama_factors = df[df['MarkettypeID'] == 'P9707']

# 重命名列以符合要求
fama_factors = fama_factors.rename(columns={
    'TradingDate': '日期',
    'RiskPremium2': 'MKT',
    'SMB2': 'SMB',
    'HML2': 'HML'
})

# 只保留需要的列
fama_factors = fama_factors[['日期', 'MKT', 'SMB', 'HML']]

# 保存为新的CSV文件
fama_factors.to_csv('Fama因子.csv', index=False)

# 2. 处理数据对齐
# 读取hs300成分股收益率数据
returns = pd.read_csv('raw_data/hs300_stock_returns.csv')

# 删除第一行空值
returns = returns.dropna(how='all')  # 删除全为空的行

# 读取Fama因子数据
fama = pd.read_csv('Fama因子.csv')

# 确保日期格式一致
returns['日期'] = pd.to_datetime(returns['日期'])
fama['日期'] = pd.to_datetime(fama['日期'])

# 根据returns的日期筛选fama数据
fama_filtered = fama[fama['日期'].isin(returns['日期'])]

# 保存筛选后的Fama因子数据
fama_filtered.to_csv('factors/Fama因子_filtered.csv', index=False)

print(f"原始Fama因子数据行数: {len(fama)}")
print(f"hs300成分股收益率数据行数: {len(returns)}")
print(f"筛选后Fama因子数据行数: {len(fama_filtered)}")

# 保存为新的CSV文件
fama_factors.to_csv('Fama因子.csv', index=False)

