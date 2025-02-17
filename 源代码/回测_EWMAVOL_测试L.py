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
import pandas as pd
import numpy as np
import time

# 设置日期范围
START_DATE = '2021-01-01'
END_DATE = '2024-11-01'

# 读取数据
market_ret = pd.read_csv('raw_data/hs300_index_returns.csv', index_col='日期')
stock_returns = pd.read_csv('raw_data/hs300_stock_returns.csv', index_col='日期')
stock_returns = stock_returns.iloc[1:]
stock_returns = stock_returns.astype(float)

# 统一日期索引格式
market_ret.index = pd.to_datetime(market_ret.index)
stock_returns.index = pd.to_datetime(stock_returns.index)

# 读取因子数据
factors_data = {
    'EWMAVOL': pd.read_csv('factors/EWMAVOL.csv', index_col='日期'),
    'GARCHVOL': pd.read_csv('factors/GARCHVOL.csv', index_col='日期'),
    'RANKVOL': pd.read_csv('factors/rankvol_processed.csv', index_col='日期'),
    'RVOL': pd.read_csv('factors/RVOL.csv', index_col='日期'),
    'VOL_3M': pd.read_csv('factors/vol_3m_processed.csv', index_col='日期')
}

# 统一因子数据日期索引格式
for name in factors_data:
    factors_data[name].index = pd.to_datetime(factors_data[name].index)

# 筛选日期范围
date_mask = (market_ret.index >= START_DATE) & (market_ret.index <= END_DATE)
market_ret = market_ret[date_mask]

date_mask = (stock_returns.index >= START_DATE) & (stock_returns.index <= END_DATE)
stock_returns = stock_returns[date_mask]

for name in factors_data:
    date_mask = (factors_data[name].index >= START_DATE) & (factors_data[name].index <= END_DATE)
    factors_data[name] = factors_data[name][date_mask]

# 添加调试信息
print("\n数据读取调试信息:")
print("市场收益率数据:")
print(market_ret.head())
print("\n因子数据 (EWMAVOL):")
print(factors_data['EWMAVOL'].head())
print("\n个股收益率数据:")
print(stock_returns.head())

# 获取所有数据集的共同日期
common_dates = market_ret.index
for factor in factors_data.values():
    common_dates = common_dates.intersection(factor.index)

print("\n共同日期数量:", len(common_dates))
if len(common_dates) > 0:
    print("第一个日期:", common_dates[0])
    print("最后一个日期:", common_dates[-1])

# ��齐所有数据到共同日期
market_ret = market_ret.loc[common_dates]
for name in factors_data:
    factors_data[name] = factors_data[name].loc[common_dates]

class IntegratedFactorData:
    def __init__(self, factors, market_returns, stock_returns):
        self.factors = factors
        self.market_returns = market_returns
        self.stock_returns = stock_returns
        self.dates = common_dates
        
    def get_factor(self, factor_name, date=None):
        if date is None:
            return self.factors[factor_name]
        return self.factors[factor_name].loc[date]
    
    def get_market_return(self, date=None):
        if date is None:
            return self.market_returns
        return self.market_returns.loc[date]
    
    def get_stock_returns(self):
        return self.stock_returns
    
    def get_available_factors(self):
        return list(self.factors.keys())
    
    def get_date_range(self):
        return [self.dates.min(), self.dates.max()]

# 创建整合后的数据对象
integrated_data = IntegratedFactorData(factors_data, market_ret, stock_returns)

class FactorPreprocessor:
    def __init__(self, factor_data):
        self.factor_data = factor_data
    
    def winsorize(self, data, lower_percentile=0.025, upper_percentile=0.975):
        """
        缩尾处理：将极端��限制在指定分位数范围内
        """
        if isinstance(data, pd.Series):
            lower_bound = data.quantile(lower_percentile)
            upper_bound = data.quantile(upper_percentile)
            return data.clip(lower=lower_bound, upper=upper_bound)
        else:  # DataFrame
            result = data.copy()
            for date in result.index:
                daily_data = result.loc[date]
                lower_bound = daily_data.quantile(lower_percentile)
                upper_bound = daily_data.quantile(upper_percentile)
                result.loc[date] = daily_data.clip(lower=lower_bound, upper=upper_bound)
            return result
    
    def standardize(self, data):
        """
        标准化处理：转换为均值为0，标准差为1的分布
        """
        if isinstance(data, pd.Series):
            mean = data.mean()
            std = data.std()
            return (data - mean) / std if std != 0 else data * 0
        else:  # DataFrame
            result = pd.DataFrame(index=data.index, columns=data.columns)
            for date in data.index:
                daily_data = data.loc[date]
                mean = daily_data.mean()
                std = daily_data.std()
                result.loc[date] = (daily_data - mean) / std if std != 0 else 0
            return result
    
    def process_factor(self, factor_name):
        """
        完整的因子预处理流程
        1. 初始缩尾处理
        2. 标准化
        3. 再次缩尾处理确保没有极端值
        """
        # 1. 初始缩尾处理
        factor = self.winsorize(self.factor_data[factor_name])
        
        # 2. 标准化
        factor = self.standardize(factor)
        
        # 3. 再次缩尾处理，使用更严格的界限
        factor = self.winsorize(factor, lower_percentile=0.01, upper_percentile=0.99)
        
        # 4. 最后再标准化一次
        factor = self.standardize(factor)
        
        return factor

class BacktestSystem:
    def __init__(self, integrated_data):
        self.data = integrated_data
        self.preprocessor = FactorPreprocessor(integrated_data.factors)
        # 初始化时就计算好月度调仓日期
        self.monthly_dates = self._get_monthly_dates()
    
    def _get_monthly_dates(self):
        """获取每月第一个交易日"""
        dates = pd.DatetimeIndex(self.data.dates)
        # 将日期转换为年月字符串
        year_month = dates.strftime('%Y-%m')
        # 找出每个月的第一个交易日
        is_month_start = year_month != pd.Series(year_month).shift(1)
        monthly_dates = dates[is_month_start]
        return monthly_dates.sort_values()
    
    def form_portfolios(self, factor_name, date, n_groups=5):
        """根据因子值将股票分组"""
        factor_values = self.preprocessor.process_factor(factor_name).loc[date]
        sorted_stocks = factor_values.sort_values()  # 因子值从小到大排序
        group_size = len(sorted_stocks) // n_groups
        
        print(f"\n分组信息 - 日期: {date}")
        print(f"总股票数: {len(sorted_stocks)}")
        print(f"每组大约股票数: {group_size}")
        
        portfolios = []
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < n_groups - 1 else len(sorted_stocks)
            portfolio = sorted_stocks.index[start_idx:end_idx]
            
            # 打印每组的因子值统计
            group_factors = sorted_stocks[start_idx:end_idx]
            print(f"\n第{i+1}组:")
            print(f"股票数量: {len(portfolio)}")
            print(f"因子值范围: {group_factors.min():.4f} 到 {group_factors.max():.4f}")
            print(f"因子值均值: {group_factors.mean():.4f}")
            print(f"因子值标准差: {group_factors.std():.4f}")
            
            portfolios.append(list(portfolio))
        
        return portfolios
    
    def calculate_portfolio_returns(self, portfolio_stocks, start_date, end_date):
        """计算组合在给定时间段的收益率"""
        # 获取个股收益率数据
        stock_rets = self.data.get_stock_returns()
        
        # 获取从start_date到end_date的所有交易日收益率
        mask = (stock_rets.index > start_date) & (stock_rets.index <= end_date)
        period_returns = stock_rets.loc[mask]
        
        # 只选择组合内的股票
        portfolio_returns = period_returns[portfolio_stocks]
        
        # 打印调试信息
        print(f"\n组合收益率计算信息:")
        print(f"期间: {start_date} 到 {end_date}")
        print(f"组合股票数: {len(portfolio_stocks)}")
        print(f"有效收益率数据的股票数: {portfolio_returns.notna().sum().mean():.0f}")
        
        # 等权重分配
        weight = 1.0 / len(portfolio_stocks)
        
        # 计算组合收益率（处理可能的缺失值）
        weighted_returns = portfolio_returns.mean(axis=1)  # 等权重平均，自动忽略NA值
        
        return weighted_returns
    
    def calculate_ic(self, factor_name, forward_returns):
        """计算IC值"""
        factor_values = self.preprocessor.process_factor(factor_name)
        ic_series = pd.Series(index=self.monthly_dates[:-1])
        
        print("\nIC计算详细信息:")
        for i in range(len(self.monthly_dates)-1):
            current_date = self.monthly_dates[i]
            next_date = self.monthly_dates[i+1]
            
            # 获取当前月的因子值
            if current_date in factor_values.index:
                current_factors = factor_values.loc[current_date]
                
                # 获取下一个月的个股收益率
                mask = (self.data.get_stock_returns().index > current_date) & \
                      (self.data.get_stock_returns().index <= next_date)
                next_month_returns = self.data.get_stock_returns()[mask].mean()
                
                # 确保因子值和收益率的股票代码对应
                common_stocks = current_factors.index.intersection(next_month_returns.index)
                if len(common_stocks) > 0:
                    # 计算相关系数
                    ic = current_factors[common_stocks].corr(next_month_returns[common_stocks])
                    ic_series[current_date] = ic
                    
                    print(f"\n日期: {current_date}")
                    print(f"因子值数量: {len(common_stocks)}")
                    print(f"因子值范围: {current_factors[common_stocks].min():.4f} 到 {current_factors[common_stocks].max():.4f}")
                    print(f"下月收益率范围: {next_month_returns[common_stocks].min():.4%} 到 {next_month_returns[common_stocks].max():.4%}")
                    print(f"IC值: {ic:.4f}")
        
        return ic_series.dropna()
    
    def calculate_rank_ic(self, factor_name, forward_returns):
        """计算RankIC值"""
        factor_values = self.preprocessor.process_factor(factor_name)
        rank_ic_series = pd.Series(index=self.monthly_dates[:-1])
        
        print("\nRankIC计算详细信息:")
        for i in range(len(self.monthly_dates)-1):
            current_date = self.monthly_dates[i]
            next_date = self.monthly_dates[i+1]
            
            # 获取当前月的因子值
            if current_date in factor_values.index:
                current_factors = factor_values.loc[current_date]
                
                # 获取下一个月的个股收益率
                mask = (self.data.get_stock_returns().index > current_date) & \
                      (self.data.get_stock_returns().index <= next_date)
                next_month_returns = self.data.get_stock_returns()[mask].mean()
                
                # 确保因子值和收益率的股票代码对应
                common_stocks = current_factors.index.intersection(next_month_returns.index)
                if len(common_stocks) > 0:
                    # 计算秩相关系数
                    rank_ic = current_factors[common_stocks].rank().corr(next_month_returns[common_stocks].rank())
                    rank_ic_series[current_date] = rank_ic
                    
                    print(f"\n日期: {current_date}")
                    print(f"股票数量: {len(common_stocks)}")
                    print(f"因子值排名范围: {current_factors[common_stocks].rank().min():.0f} 到 {current_factors[common_stocks].rank().max():.0f}")
                    print(f"收益排名范围: {next_month_returns[common_stocks].rank().min():.0f} 到 {next_month_returns[common_stocks].rank().max():.0f}")
                    print(f"RankIC值: {rank_ic:.4f}")
        
        return rank_ic_series.dropna()
    
    def calculate_performance_metrics(self, returns):
        """计算绩效指标"""
        # 计算累积收益
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # 计算年化收益率
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1/years) - 1
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def run_backtest(self, factor_name):
        """运行完整的回测"""
        self.current_factor = factor_name  # 保存当前因子名称
        monthly_dates = self.monthly_dates
        
        # 获取所有交易日期
        all_dates = self.data.get_market_return().index
        # 初始化返回DataFrame，使用所有交易日作为索引
        portfolio_returns = pd.DataFrame(index=all_dates)
        
        print("\n开始回测:")
        print(f"月度调仓日期数量: {len(monthly_dates)}")
        
        # 先算所有组合的收益
        for i in range(len(monthly_dates)-1):
            start_date = monthly_dates[i]
            end_date = monthly_dates[i+1]
            
            print(f"\n处理期间: {start_date} 到 {end_date}")
            
            # 获取分组
            portfolios = self.form_portfolios(factor_name, start_date)
            
            # 算各组合收益
            for group_id, portfolio in enumerate(portfolios, 1):
                print(f"\n计算第{group_id}组收益率")
                returns = self.calculate_portfolio_returns(portfolio, start_date, end_date)
                
                # 使用loc赋值前先查日期
                print(f"当期收益率日期范围: {returns.index.min()} 到 {returns.index.max()}")
                print(f"收益率天数: {len(returns)}")
                
                # 为该期间赋值收益率
                mask = (portfolio_returns.index > start_date) & (portfolio_returns.index <= end_date)
                portfolio_returns.loc[mask, f'Group_{group_id}'] = returns
        
        # 添加调试信息
        print("\nPortfolio returns summary:")
        print("Shape:", portfolio_returns.shape)
        print("Columns:", portfolio_returns.columns)
        print("First few rows:\n", portfolio_returns.head())
        
        # 计算IC和RankIC
        ic = self.calculate_ic(factor_name, portfolio_returns['Group_1'])
        rank_ic = self.calculate_rank_ic(factor_name, portfolio_returns['Group_1'])
        
        # 算多空组收益
        long_short_returns = portfolio_returns['Group_1'] - portfolio_returns['Group_5']
        
        # 计算绩效指标
        performance = {}
        for group in portfolio_returns.columns:
            performance[group] = self.calculate_performance_metrics(portfolio_returns[group])
        performance['long_short'] = self.calculate_performance_metrics(long_short_returns)
        
        return {
            'returns': portfolio_returns,
            'long_short_returns': long_short_returns,
            'ic': ic,
            'rank_ic': rank_ic,
            'performance': performance
        }

class ResultAnalyzer:
    def __init__(self, backtest_results, market_returns):
        self.results = backtest_results
        self.market_returns = market_returns
        
    def calculate_performance_metrics(self, returns):
        """计算绩效指标"""
        # 计算累积收益
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # 计算年化收益率
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1/years) - 1
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def calculate_excess_metrics(self, returns):
        """计算超额收益相关指标"""
        market_returns = self.market_returns['收盘']
        
        # 打印市场收益率的基本信息
        print("市场收益率统计:")
        print(f"数据起始日期: {market_returns.index[0]}")
        print(f"数据结束日期: {market_returns.index[-1]}")
        print(f"数据点数: {len(market_returns)}")
        print(f"日均收益率: {market_returns.mean():.6f}")
        print(f"收益率标准差: {market_returns.std():.6f}")
        
        # 计算市场累积收益
        market_cum_returns = (1 + market_returns).cumprod()
        market_total_return = market_cum_returns.iloc[-1] - 1
        years = len(market_returns) / 252
        market_annual_return = (1 + market_total_return) ** (1/years) - 1
        
        print(f"市场总收益: {market_total_return:.4%}")
        print(f"年化收益率: {market_annual_return:.4%}")
        
        # 计算策略的年化收益率
        strategy_cum_returns = (1 + returns).cumprod()
        strategy_total_return = strategy_cum_returns.iloc[-1] - 1
        years = len(returns) / 252
        strategy_annual_return = (1 + strategy_total_return) ** (1/years) - 1
        
        # 计算年化超额收益
        annual_excess_return = strategy_annual_return - market_annual_return
        
        # 计算超额收益的波动率
        excess_returns = returns - market_returns
        excess_volatility = excess_returns.std() * np.sqrt(252)
        
        # 信息比率
        ir = annual_excess_return / excess_volatility if excess_volatility != 0 else 0
        
        # 相对胜率
        relative_win_rate = (excess_returns > 0).mean()
        
        # 超额最大回撤
        cumulative_excess = (1 + excess_returns).cumprod()
        rolling_max = cumulative_excess.expanding().max()
        drawdowns = cumulative_excess / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'excess_return': annual_excess_return,
            'information_ratio': ir,
            'excess_drawdown': max_drawdown,
            'relative_win_rate': relative_win_rate
        }
    
    def calculate_yearly_metrics(self, returns, strategy_type='long'):
        """计算分年度的绩效指标"""
        # 将收益率数据按年份分组
        returns.index = pd.to_datetime(returns.index)
        yearly_groups = returns.groupby(returns.index.year)
        market_returns = self.market_returns['收盘']
        
        # 创建结果DataFrame
        results = []
        
        for year, year_returns in yearly_groups:
            # 获取对应年份的市场收益率
            year_market_returns = market_returns[market_returns.index.year == year]
            
            # 计算该年的绩效指标
            perf = self.calculate_performance_metrics(year_returns)
            excess = self.calculate_excess_metrics(year_returns)
            
            # 汇总该年的指标
            year_metrics = {
                '年份': year,
                '年化收益': perf['annual_return'],
                '超额收益': excess['excess_return'],
                '夏普比率': perf['sharpe_ratio'],
                '超额回撤': excess['excess_drawdown'],
                '相对胜率': excess['relative_win_rate'],
                '信息比率': excess['information_ratio']
            }
            results.append(year_metrics)
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        df_results.set_index('年份', inplace=True)
        
        # 格式化百分比列
        percent_columns = ['年化收益', '超额收益', '超额回撤', '相对胜率']
        for col in percent_columns:
            df_results[col] = df_results[col].apply(lambda x: f"{x:.2%}")
        
        # 格式化比率列
        ratio_columns = ['夏普比率', '信息比率']
        for col in ratio_columns:
            df_results[col] = df_results[col].apply(lambda x: f"{x:.2f}")
        
        return df_results

    def generate_yearly_performance_report(self, factor_name, output_path=None):
        """生成分年度表现报告"""
        if output_path is None:
            output_path = f'{factor_name}因子选股的分年表现.xlsx'
        
        # 计算多头策略的分年表现
        long_yearly_perf = self.calculate_yearly_metrics(
            self.results['returns']['Group_1'], 
            strategy_type='long'
        )
        
        # 计算多空策略的分年表现
        ls_yearly_perf = self.calculate_yearly_metrics(
            self.results['long_short_returns'], 
            strategy_type='long_short'
        )
        
        # 保存到Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            long_yearly_perf.to_excel(writer, sheet_name='多头策略分年表现')
            ls_yearly_perf.to_excel(writer, sheet_name='多空策略分年表现')
            
            # 调整列宽
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(worksheet.columns, 1):
                    max_length = max(len(str(cell.value)) for cell in col)
                    worksheet.column_dimensions[chr(64 + idx)].width = max_length + 2
        
        print(f"\n分年度表现报告已保存至: {output_path}")
        return long_yearly_perf, ls_yearly_perf

def analyze_all_factors():
    """
    对所有因子进行分析并生成报告
    """
    # 获取所有因子名称
    all_factor_names = list(factors_data.keys())
    print(f"开始分析以下因子：")
    for i, factor in enumerate(all_factor_names, 1):
        print(f"{i}. {factor}")
    
    # 创建回测系统
    backtest = BacktestSystem(integrated_data)
    
    # 只分析EWMAVOL因子的分年表现
    factor_name = 'EWMAVOL'
    print(f"\n正在分析因子: {factor_name}")
    
    # 运行回测
    results = backtest.run_backtest(factor_name)
    
    # 创建分析器
    analyzer = ResultAnalyzer(results, integrated_data.get_market_return())
    
    # 生成分年度表现报告
    long_yearly_perf, ls_yearly_perf = analyzer.generate_yearly_performance_report(factor_name)
    
    # 打印分年度表现
    print("\n多头策略分年表现:")
    print(long_yearly_perf)
    print("\n多空策略分年表现:")
    print(ls_yearly_perf)

def analyze_decay_factors():
    """
    分析不同衰减因子的表现
    """
    # 定义衰减因子列表
    decay_factors = [0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    
    # 创建结果存储列表
    results = []
    
    # 创建回测系统
    backtest = BacktestSystem(integrated_data)
    
    print("开始分析不同衰减因子的表现：")
    
    for decay in decay_factors:
        print(f"\n正在分析衰减因子: {decay}")
        
        # 重新计算EWMAVOL因子
        stock_returns = integrated_data.get_stock_returns()
        ewma_vol = stock_returns.ewm(alpha=1-decay).std()
        
        # 更新因子数据
        factors_data['EWMAVOL'] = ewma_vol
        
        # 运行回测
        results_decay = backtest.run_backtest('EWMAVOL')
        
        # 创建分析器
        analyzer = ResultAnalyzer(results_decay, integrated_data.get_market_return())
        
        # 获取多头策略表现
        long_perf = results_decay['performance']['Group_1']
        
        # 获取多空策略表现
        ls_returns = results_decay['long_short_returns']
        ls_perf = analyzer.calculate_performance_metrics(ls_returns)
        
        # 获取IC相关指标
        ic = results_decay['ic']
        rank_ic = results_decay['rank_ic']
        ic_ir = (ic.mean()/ic.std()) * np.sqrt(12) if ic.std() != 0 else 0
        rank_ic_ir = (rank_ic.mean()/rank_ic.std()) * np.sqrt(12) if rank_ic.std() != 0 else 0
        
        # 汇总该衰减因子的结果
        decay_result = {
            '衰减因子': decay,
            '多头年化收益': long_perf['annual_return'],
            '多头夏普比率': long_perf['sharpe_ratio'],
            '多空年化收益': ls_perf['annual_return'],
            '多空夏普比率': ls_perf['sharpe_ratio'],
            'IC均值': ic.mean(),
            '年化ICIR': ic_ir,
            'RankIC均值': rank_ic.mean(),
            '年化RankICIR': rank_ic_ir
        }
        
        results.append(decay_result)
        
        print(f"衰减因子 {decay} 分析完成")
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    df_results.set_index('衰减因子', inplace=True)
    
    # 格式化百分比列
    percent_columns = ['多头年化收益', '多空年化收益']
    for col in percent_columns:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.2%}")
    
    # 格式化比率列
    ratio_columns = ['多头夏普比率', '多空夏普比率', 'IC均值', '年化ICIR', 'RankIC均值', '年化RankICIR']
    for col in ratio_columns:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.4f}")
    
    # 保存到Excel
    output_path = '衰减因子敏感性分析.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='衰减因子敏感性')
        
        # 调整列宽
        worksheet = writer.sheets['衰减因子敏感性']
        for idx, col in enumerate(df_results.columns, 1):
            max_length = max(
                df_results[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(64 + idx)].width = max_length + 2
    
    print(f"\n衰减因子敏感性分析报告已保存至: {output_path}")
    return df_results

def analyze_window_length():
    """
    分析不同历史数据长度的表现
    """
    # 定义历史数据长度列表
    window_lengths = [10, 20, 40, 60, 120]
    
    # 创建结果存储列表
    results = []
    
    # 创建回测系统
    backtest = BacktestSystem(integrated_data)
    
    print("开始分析不同历史数据长度的表现：")
    
    for window in window_lengths:
        print(f"\n正在分析历史数据长度: {window}天")
        
        # 重新计算EWMAVOL因子，使用calculate_ewmavol函数
        stock_returns = integrated_data.get_stock_returns()
        ewma_vol = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns)
        
        for column in stock_returns.columns:
            returns = stock_returns[column].values
            vol = calculate_ewmavol(returns, window=window)  # 使用不同的window长度
            ewma_vol[column] = vol
        
        # 更新因子数据
        factors_data['EWMAVOL'] = ewma_vol
        
        # 运行回测
        results_window = backtest.run_backtest('EWMAVOL')
        
        # 创建分析器
        analyzer = ResultAnalyzer(results_window, integrated_data.get_market_return())
        
        # 获取多头策略表现
        long_perf = results_window['performance']['Group_1']
        
        # 获取多空策略表现
        ls_returns = results_window['long_short_returns']
        ls_perf = analyzer.calculate_performance_metrics(ls_returns)
        
        # 获取IC相关指标
        ic = results_window['ic']
        rank_ic = results_window['rank_ic']
        ic_ir = (ic.mean()/ic.std()) * np.sqrt(12) if ic.std() != 0 else 0
        rank_ic_ir = (rank_ic.mean()/rank_ic.std()) * np.sqrt(12) if rank_ic.std() != 0 else 0
        
        # 汇总该窗口长度的结果
        window_result = {
            '历史数据长度': window,
            '多头年化收益': long_perf['annual_return'],
            '多头夏普比率': long_perf['sharpe_ratio'],
            '多空年化收益': ls_perf['annual_return'],
            '多空夏普比率': ls_perf['sharpe_ratio'],
            'IC均值': ic.mean(),
            '年化ICIR': ic_ir,
            'RankIC均值': rank_ic.mean(),
            '年化RankICIR': rank_ic_ir
        }
        
        results.append(window_result)
        
        print(f"历史数据长度 {window} 分析完成")
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    df_results.set_index('历史数据长度', inplace=True)
    
    # 格式化百分比列
    percent_columns = ['多头年化收益', '多空年化收益']
    for col in percent_columns:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.2%}")
    
    # 格式化比率列
    ratio_columns = ['多头夏普比率', '多空夏普比率', 'IC均值', '年化ICIR', 'RankIC均值', '年化RankICIR']
    for col in ratio_columns:
        df_results[col] = df_results[col].apply(lambda x: f"{x:.4f}")
    
    # 保存到Excel
    output_path = '历史数据长度敏感性分析.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='历史数据长度敏感性')
        
        # 调整列宽
        worksheet = writer.sheets['历史数据长度敏感性']
        for idx, col in enumerate(df_results.columns, 1):
            max_length = max(
                df_results[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(64 + idx)].width = max_length + 2
    
    print(f"\n历史数据长度敏感性分析报告已保存至: {output_path}")
    return df_results

if __name__ == "__main__":
    # analyze_all_factors()  # 注释掉原来的函数调用
    # results = analyze_decay_factors()  # 注释掉原来的函数调用
    results = analyze_window_length()  # 运行历史数据长度敏感性分析
    print("\n历史数据长度敏感性分析结果:")
    print(results)