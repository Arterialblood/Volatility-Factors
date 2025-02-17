import pandas as pd

# 导入数据
returns_df = pd.read_csv('raw_data/hs300_stock_returns.csv')
prices_df = pd.read_csv('raw_data/hs300_close_prices.csv')

# 要删除的股票列表
stocks_to_remove = ['000792', '001289', '300919', '300957', '300979', '301269', 
                    '600188', '600905', '600938', '600941', '601059', '601728',
                    '601868', '603296', '605117', '605499', '688041', '688082', 
                    '688187', '688223', '688271', '688303']

# 打印处理前的形状
print("处理前:")
print(f"收益率数据形状: {returns_df.shape}")
print(f"价格数据形状: {prices_df.shape}")

# 删除指定的股票列
returns_df = returns_df.drop(columns=stocks_to_remove)
prices_df = prices_df.drop(columns=stocks_to_remove)

# 直接覆盖原文件
returns_df.to_csv('raw_data/hs300_stock_returns.csv', index=False)
prices_df.to_csv('raw_data/hs300_close_prices.csv', index=False)

# 打印处理后的形状
print("\n处理后:")
print(f"收益率数据形状: {returns_df.shape}")
print(f"价格数据形状: {prices_df.shape}")
