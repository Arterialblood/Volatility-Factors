import pandas as pd
import numpy as np
from arch import arch_model
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

def calculate_garch_vol(stock, data):
    predictions = []
    for i in range(60, len(data)):
        window_data = data.iloc[i-60:i]
        stock_returns = window_data[stock]
        factors = window_data[['MKT', 'SMB', 'HML']]
        
        try:
            valid_mask = ~(stock_returns.isna() | factors.isna().any(axis=1))
            stock_returns = stock_returns[valid_mask]
            factors = factors[valid_mask]
            
            if len(stock_returns) < 30:
                predictions.append(np.nan)
                continue
            
            X = np.c_[np.ones(len(factors)), factors]
            y = stock_returns
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            y_pred = X @ beta
            residuals = y - y_pred
            
            # Rescale residuals to avoid DataScaleWarning
            residuals *= 100
            
            garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            
            forecast = garch_fit.forecast(horizon=20)
            pred_vol_value = np.sqrt(forecast.variance.values[-1, 0])
            predictions.append(pred_vol_value)
            
        except:
            predictions.append(np.nan)
    
    predictions = [np.nan] * 60 + predictions
    return stock, predictions

if __name__ == '__main__':
    # 读取数据
    returns = pd.read_csv(r"C:\Users\86187\Desktop\hs300_stock_returns.csv")
    fama = pd.read_csv(r"C:\Users\86187\Desktop\Fama因子_filtered.csv")

    # 将日期列转换为datetime格式
    returns['日期'] = pd.to_datetime(returns['日期'])
    fama['日期'] = pd.to_datetime(fama['日期'])

    # 按日期合并数据
    merged_data = pd.merge(returns, fama, on='日期', how='inner')
    merged_data = merged_data.sort_values('日期').reset_index(drop=True)

    # 限制处理的股票数量
    # num_stocks_to_process = 5  # 只处理前5个股票
    stock_columns = [col for col in returns.columns if col != '日期']  # 使用所有股票

    # 创建一个DataFrame来存储预测的特质波动率
    pred_vol = pd.DataFrame(index=merged_data.index, columns=['日期'] + stock_columns)
    pred_vol['日期'] = merged_data['日期']

    # 使用并行处理
    with ProcessPoolExecutor() as executor:
        # 使用partial固定merged_data参数
        func = partial(calculate_garch_vol, data=merged_data)
        results = list(tqdm(executor.map(func, stock_columns), 
                          total=len(stock_columns), 
                          desc="Processing stocks", 
                          position=0, 
                          leave=True))

    # 将结果填入DataFrame
    for stock, predictions in results:
        pred_vol[stock] = predictions

    # 保存结果
    pred_vol.to_csv(r"C:\Users\86187\Desktop\GARCHVOL.csv", index=False)