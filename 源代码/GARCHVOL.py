import pandas as pd

# 读取GARCHVOL(1).csv
garch_vol = pd.read_csv('factors/GARCHVOL(1).csv')

# 将第一列设为索引
garch_vol.set_index('日期', inplace=True)

# 打印原始数据形状
print("原始数据形状:", garch_vol.shape)

# 删除前60行全部缺失的数据
garch_vol = garch_vol.iloc[60:]

# 计算每只股票的缺失值数量
null_counts = garch_vol.isnull().sum()

# 找出存在缺失值的股票
stocks_with_missing = null_counts[null_counts > 0].index.tolist()

print("\n将被删除的股票列表:")
print(stocks_with_missing)
print(f"需要删除的股票数量: {len(stocks_with_missing)}")

# 删除存在缺失值的股票
garch_vol = garch_vol.drop(columns=stocks_with_missing)

# 打印处理后的数据形状
print("\n处理后的数据形状:", garch_vol.shape)

# 确认是否还有缺失值
remaining_nulls = garch_vol.isnull().sum().sum()
print("剩余缺失值数量:", remaining_nulls)

# 保存处理后的数据
garch_vol.to_csv('factors/GARCHVOL.csv')