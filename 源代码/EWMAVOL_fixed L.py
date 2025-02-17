import pandas as pd
import numpy as np

# 读取数据
returns_df = pd.read_csv('raw_data/hs300_stock_returns.csv', index_col='日期')

def calculate_ewmavol(returns, window=60, lambda_param=0.9):
    """
    计算EWMA波动率
    
    参数:
    returns: 收益率序列
    window: 滚动窗口大小
    lambda_param: 衰减因子
    """
    # 将numpy array中的nan替换为0
    returns = np.nan_to_num(returns, 0)
    
    # 初始化结果数组
    volatility = np.zeros(len(returns))
    
    for t in range(window, len(returns)):
        # 获取窗口内的数据
        window_returns = returns[t-window:t]
        
        # 计算均值
        mean_return = np.mean(window_returns)
        
        # 初始化第一个方差值(使用简单方差)
        var_t = np.var(window_returns[0:window])
        
        # 对窗口内的每个数据点使用EWMA更新方差
        for r in window_returns[1:]:
            var_t = (1 - lambda_param) * ((r - mean_return) ** 2) + lambda_param * var_t
            
        # 计算波动率(标准差)
        volatility[t] = np.sqrt(var_t)
    
    return volatility

# 对每只股票计算EWMAVOL
ewmavol_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)

for column in returns_df.columns:
    returns = returns_df[column].values
    ewmavol = calculate_ewmavol(returns)
    ewmavol_df[column] = ewmavol

# 删除前60天的数据
ewmavol_df = ewmavol_df.iloc[60:]

# 重置索引，保留日期列
ewmavol_df = ewmavol_df.reset_index()

# 保存结果
ewmavol_df.to_csv('factors/EWMAVOL.csv', index=False)

# 打印结果形状
print(f"EWMAVOL因子数据形状: {ewmavol_df.shape}")
