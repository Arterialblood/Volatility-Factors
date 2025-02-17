import akshare as ak
import pandas as pd
import numpy as np
import os

def get_hs300_stocks():
    """获取沪深300成分股数据"""
    try:
        hs300_stocks = ak.index_stock_cons_weight_csindex(symbol="000300")
        stock_codes = hs300_stocks['成分券代码'].tolist()
        return stock_codes, hs300_stocks
    except Exception as e:
        print(f"获取沪深300成分股数据时出错: {str(e)}")
        return None, None

def get_stock_data(stock_codes, start_date="20200922", end_date="20241101"):
    """获取多个股票的历史数据"""
    try:
        close_prices = pd.DataFrame()
        
        total_stocks = len(stock_codes)
        print(f"开始获取{total_stocks}只股票的历史数据...")
        
        for i, code in enumerate(stock_codes, 1):
            try:
                stock_data = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                
                stock_data['日期'] = pd.to_datetime(stock_data['日期'])
                stock_data.set_index('日期', inplace=True)
                
                close_prices[code] = stock_data['收盘']
                
                if i % 10 == 0:
                    print(f"已完成 {i}/{total_stocks} 只股票的数据获取")
                
            except Exception as e:
                print(f"获取股票 {code} 数据时出错: {str(e)}")
                continue
        
        print("数据获取完成！")
        print(f"收盘价数据维度: {close_prices.shape}")
        
        return close_prices
        
    except Exception as e:
        print(f"获取股票数据时出错: {str(e)}")
        return None

def get_market_returns():
    """获取沪深300指数的历史数据"""
    try:
        hs300_data = ak.index_zh_a_hist(
            symbol="000300", 
            period="daily", 
            start_date="20200922", 
            end_date="20241101"
        )
        
        hs300_data['日期'] = pd.to_datetime(hs300_data['日期'])
        hs300_data.set_index('日期', inplace=True)
        
        return hs300_data['收盘']
        
    except Exception as e:
        print(f"获取市场收益率数据时出错: {str(e)}")
        return None

def get_risk_free_rate(start_date="20200922", end_date="20241101"):
    """获取无风险利率数据"""
    try:
        print("正在获取国债收益率数据...")
        
        all_rates = []
        current_start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        while current_start < end:
            current_end = min(current_start + pd.DateOffset(years=1), end)
            current_start_str = current_start.strftime('%Y%m%d')
            current_end_str = current_end.strftime('%Y%m%d')
            
            print(f"获取期间: {current_start_str} - {current_end_str}")
            
            bond_data = ak.bond_china_yield(
                start_date=current_start_str,
                end_date=current_end_str
            )
            
            treasury_data = bond_data[bond_data['曲线名称'] == '中债国债收益率曲线']
            
            if not treasury_data.empty:
                all_rates.append(treasury_data)
            
            current_start = current_end + pd.DateOffset(days=1)
        
        if not all_rates:
            raise Exception("未获取到任何数据")
        
        combined_data = pd.concat(all_rates)
        combined_data['日期'] = pd.to_datetime(combined_data['日期'])
        combined_data.set_index('日期', inplace=True)
        
        return combined_data['1年']
        
    except Exception as e:
        print(f"获取无风险利率数据时出错: {str(e)}")
        return None

def calculate_returns(prices_df):
    """计算收益率"""
    return prices_df.pct_change()

# 创建保存数据的文件夹
if not os.path.exists('raw_data'):
    os.makedirs('raw_data')

# 获取数据
stock_codes, _ = get_hs300_stocks()
if stock_codes:
    # 获取成分股收盘价
    close_prices = get_stock_data(stock_codes)
    if close_prices is not None:
        # 计算成分股收益率
        stock_returns = calculate_returns(close_prices)
        # 保存收盘价和收益率
        close_prices.to_csv('raw_data/hs300_close_prices.csv')
        stock_returns.to_csv('raw_data/hs300_stock_returns.csv')
        print("沪深300成分股收盘价和收益率数据已保存")

    # 获取沪深300指数收盘价
    hs300_price = get_market_returns()
    if hs300_price is not None:
        # 计算指数收益率
        hs300_returns = calculate_returns(hs300_price)
        # 保存指数收盘价和收益率
        hs300_price.to_csv('raw_data/hs300_index.csv')
        hs300_returns.to_csv('raw_data/hs300_index_returns.csv')
        print("沪深300指数收盘价和收益率数据已保存")

    # 获取无风险利率
    risk_free = get_risk_free_rate()
    if risk_free is not None:
        risk_free.to_csv('raw_data/risk_free_rate.csv')
        print("无风险利率数据已保存")

print("\n=== 数据获取和保存完成 ===")
print("数据已保存至 'raw_data' 文件夹")
print("\n保存的文件包括:")
print("- hs300_close_prices.csv (沪深300成分股收盘价)")
print("- hs300_stock_returns.csv (沪深300成分股收益率)")
print("- hs300_index.csv (沪深300指数收盘价)")
print("- hs300_index_returns.csv (沪深300指数收益率)")
print("- risk_free_rate.csv (无风险利率)")