import pandas as pd

stocks_to_remove = ['000792', '001289', '300919', '300957', '300979', '301269', 
                    '600188', '600905', '600938', '600941', '601059', '601728',
                    '601868', '603296', '605117', '605499', '688041', '688082',
                    '688187', '688223', '688271', '688303']

# 处理 RVOL.csv
rvol_df = pd.read_csv('factors/RVOL.csv')
rvol_df = rvol_df.drop(columns=stocks_to_remove, errors='ignore')
rvol_df.to_csv('factors/RVOL.csv', index=False)

# 处理 vol_3m.csv
vol_3m_df = pd.read_csv('factors/vol_3m.csv')
vol_3m_df = vol_3m_df.drop(columns=stocks_to_remove, errors='ignore')
vol_3m_df.to_csv('factors/vol_3m.csv', index=False)

# 处理 rankvol.csv
rankvol_df = pd.read_csv('factors/rankvol.csv')
rankvol_df = rankvol_df.drop(columns=stocks_to_remove, errors='ignore')
rankvol_df.to_csv('factors/rankvol.csv', index=False)

# 处理 rankvol_processed.csv
rankvol_processed_df = pd.read_csv('factors/rankvol_processed.csv')
rankvol_processed_df = rankvol_processed_df.drop(columns=stocks_to_remove, errors='ignore')
rankvol_processed_df.to_csv('factors/rankvol_processed.csv', index=False)

# 处理 vol_3m_processed.csv
vol_3m_processed_df = pd.read_csv('factors/vol_3m_processed.csv')
vol_3m_processed_df = vol_3m_processed_df.drop(columns=stocks_to_remove, errors='ignore')
vol_3m_processed_df.to_csv('factors/vol_3m_processed.csv', index=False)
# 读取并打印每个文件的形状
print("RVOL.csv shape:", pd.read_csv('factors/RVOL.csv').shape)
print("vol_3m.csv shape:", pd.read_csv('factors/vol_3m.csv').shape)
print("rankvol.csv shape:", pd.read_csv('factors/rankvol.csv').shape)
print("rankvol_processed.csv shape:", pd.read_csv('factors/rankvol_processed.csv').shape)
print("vol_3m_processed.csv shape:", pd.read_csv('factors/vol_3m_processed.csv').shape)