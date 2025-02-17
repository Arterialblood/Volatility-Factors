import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

# 筛选日期
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

# 对齐所有数据到共同日期
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
        缩尾处理：将极端值限制在指定分位数范围内
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
        
        # 获取从start_date到end_date的所有交易日收益
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
        """算绩效指标"""
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
                
                # 使用loc赋值前先检查日期
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
        print("市场���益率统计:")
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
        
        # 算年化超额收益
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
    
    def generate_performance_table(self):
        """生成因子收益表现表格"""
        # 多头表现
        long_perf = self.results['performance']['Group_1']
        long_excess = self.calculate_excess_metrics(self.results['returns']['Group_1'])
        
        # 多空组合表现
        ls_returns = self.results['returns']['Group_1'] - self.results['returns']['Group_5']
        ls_perf = self.calculate_performance_metrics(ls_returns)
        ls_excess = self.calculate_excess_metrics(ls_returns)
        
        performance_data = {
            '多头表现': {
                '年化收益': f"{long_perf['annual_return']:.2%}",
                '超额收益': f"{long_excess['excess_return']:.2%}",
                '夏普比率': f"{long_perf['sharpe_ratio']:.2f}",
                '超额回撤': f"{long_excess['excess_drawdown']:.2%}",
                '相对胜率': f"{long_excess['relative_win_rate']:.2%}",
                '信息比率': f"{long_excess['information_ratio']:.2f}"
            },
            '多空组合表现': {
                '年化收益': f"{ls_perf['annual_return']:.2%}",
                '超额收益': f"{ls_excess['excess_return']:.2%}",
                '夏普比率': f"{ls_perf['sharpe_ratio']:.2f}",
                '超额回撤': f"{ls_excess['excess_drawdown']:.2%}",
                '相对胜率': f"{ls_excess['relative_win_rate']:.2%}",
                '信息比率': f"{ls_excess['information_ratio']:.2f}"
            }
        }
        
        return pd.DataFrame(performance_data)
    
    def generate_ic_table(self):
        """生成因子IC表现表格"""
        ic = self.results['ic']
        rank_ic = self.results['rank_ic']
        
        # 计算IC相关指标
        ic_mean = ic.mean()
        ic_std = ic.std()
        ic_ir = ic_mean / ic_std if ic_std != 0 else 0
        
        # 计算RankIC相关指标
        rank_ic_mean = rank_ic.mean()
        rank_ic_std = rank_ic.std()
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 else 0
        rank_ic_positive_rate = (rank_ic > 0).mean()
        
        ic_data = {
            'IC表现': {
                'IC均值': f"{ic_mean:.4f}",
                'IC标准差': f"{ic_std:.4f}",
                '年化ICIR': f"{ic_ir * np.sqrt(12):.4f}"
            },
            'RankIC表现': {
                'RankIC均值': f"{rank_ic_mean:.4f}",
                'RankIC标准差': f"{rank_ic_std:.4f}",
                '年化RankICIR': f"{rank_ic_ir * np.sqrt(12):.4f}",
                'RankIC>0': f"{rank_ic_positive_rate:.2%}"
            }
        }
        
        return pd.DataFrame(ic_data)
    
    def print_factor_analysis_results(self):
        """打印简化的因子分析结果"""
        print("\n=== 因子表现分析 ===")
        
        # 1. 多头组合表现
        print("\n多头组合表现:")
        group1_returns = self.results['returns']['Group_1']
        group1_perf = self.calculate_performance_metrics(group1_returns)
        group1_excess = self.calculate_excess_metrics(group1_returns)
        
        print(f"年化收益: {group1_perf['annual_return']:.2%}")
        print(f"超额收益: {group1_excess['excess_return']:.2%}")
        print(f"信息比��: {group1_excess['information_ratio']:.2f}")
        print(f"最大回撤: {group1_perf['max_drawdown']:.2%}")
        
        # 2. 多空组合表现
        print("\n多空组合表现:")
        ls_returns = self.results['long_short_returns']
        ls_perf = self.calculate_performance_metrics(ls_returns)
        
        print(f"年化收益: {ls_perf['annual_return']:.2%}")
        print(f"夏普比率: {ls_perf['sharpe_ratio']:.2f}")
        print(f"最大回撤: {ls_perf['max_drawdown']:.2%}")
        
        # 3. 因子IC分析
        print("\nIC分析:")
        ic = self.results['ic']
        rank_ic = self.results['rank_ic']
        
        print(f"IC均值: {ic.mean():.4f}")
        print(f"RankIC均值: {rank_ic.mean():.4f}")
        print(f"IC IR: {(ic.mean()/ic.std()) * np.sqrt(12):.4f}")
        print(f"RankIC > 0占比: {(rank_ic > 0).mean():.2%}")
    
    def print_data_summary(self, returns):
        """打印数据概要"""
        print("\n数据概要:")
        print("收益数据:")
        print(f"数据点数量: {len(returns)}")
        print(f"开始日期: {returns.index[0]}")
        print(f"结束日期: {returns.index[-1]}")
        print(f"最大值: {returns.max():.6f}")
        print(f"最小值: {returns.min():.6f}")
        print(f"均值: {returns.mean():.6f}")
        print(f"中位数: {returns.median():.6f}")
        
        market_returns = self.market_returns['收盘']
        print("\n市场收益率数据:")
        print(f"数据点数量: {len(market_returns)}")
        print(f"开始日期: {market_returns.index[0]}")
        print(f"结束日期: {market_returns.index[-1]}")
        print(f"最大值: {market_returns.max():.6f}")
        print(f"最小值: {market_returns.min():.6f}")
        print(f"均值: {market_returns.mean():.6f}")
        print(f"中位数: {market_returns.median():.6f}")

    def generate_factor_analysis_report(self, factor_name, output_path=None):
        """生成因子分析报告并输出到Excel"""
        if output_path is None:
            output_path = f'factor_analysis_{factor_name}.xlsx'
        
        # 首先打印结果结构以便调试
        print("Results structure:", self.results.keys())
        
        # 获取多头组合收益
        group1_returns = self.results['raw_results']['returns']['Group_1']
        group1_perf = self.calculate_performance_metrics(group1_returns)
        group1_excess = self.calculate_excess_metrics(group1_returns)
        
        # 获空组合收益
        ls_returns = self.results['raw_results']['long_short_returns']
        ls_perf = self.calculate_performance_metrics(ls_returns)
        ls_excess = self.calculate_excess_metrics(ls_returns)
        
        # 创建结果字典
        results = {
            '多头组合表现': {
                '年化收益': f"{group1_perf['annual_return']:.2%}",
                '超额收益': f"{group1_excess['excess_return']:.2%}",
                '夏普比率': f"{group1_perf['sharpe_ratio']:.2f}",
                '超额回撤': f"{group1_excess['excess_drawdown']:.2%}",
                '相对胜率': f"{group1_excess['relative_win_rate']:.2%}",
                '信息比率': f"{group1_excess['information_ratio']:.2f}"
            },
            '多空组合表现': {
                '年化收益': f"{ls_perf['annual_return']:.2%}",
                '超额收益': f"{ls_excess['excess_return']:.2%}",
                '夏普比率': f"{ls_perf['sharpe_ratio']:.2f}",
                '最大回撤': f"{ls_perf['max_drawdown']:.2%}",
                '胜率': f"{ls_perf['win_rate']:.2%}"
            },
            '因子IC分析': {
                'IC均值': f"{self.results['raw_results']['ic'].mean():.4f}",
                'IC标准差': f"{self.results['raw_results']['ic'].std():.4f}",
                '年化ICIR': f"{(self.results['raw_results']['ic'].mean()/self.results['raw_results']['ic'].std()) * np.sqrt(12):.4f}",
                'RankIC均值': f"{self.results['raw_results']['rank_ic'].mean():.4f}",
                'RankIC标准差': f"{self.results['raw_results']['rank_ic'].std():.4f}",
                '年化RankICIR': f"{(self.results['raw_results']['rank_ic'].mean()/self.results['raw_results']['rank_ic'].std()) * np.sqrt(12):.4f}",
                'RankIC>0占比': f"{(self.results['raw_results']['rank_ic'] > 0).mean():.2%}"
            }
        }
        
        # 转换为DataFrame并保存到Excel
        df_results = pd.DataFrame.from_dict(results, orient='index')
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='因子分析结果')
            
            # 调整列宽
            worksheet = writer.sheets['因子分析结果']
            for idx, col in enumerate(df_results.columns):
                max_length = max(
                    df_results[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        print(f"\n因子分析报告已保存至: {output_path}")
        
        # 打印结果
        print("\n=== 因子表现分析 ===")
        for section, metrics in results.items():
            print(f"\n{section}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")

def calculate_factor_returns(factor_names=['EWMAVOL', 'VOL_3M']):
    """
    计算因子回测的净值并输出到Excel
    波动率因子：做多低波动（G1），做空高波动（G5）
    """
    results = {}
    daily_nav_dict = {}
    
    for factor_name in factor_names:
        # 获取因子数据
        factor_data = factors_data[factor_name]
        
        # 初始化收益率字典和日期列表
        daily_returns = {}
        dates_list = []
        
        # 对每个交易日进行计算
        for date in factor_data.index:
            # 获取当日因子值并升序排列
            current_factors = factor_data.loc[date].sort_values()  # 升序排列
            
            # 计算每组的股票数量
            n_stocks = len(current_factors)
            stocks_per_group = n_stocks // 5
            
            # 分组（G1为最低波动率组，G5为最高波动率组）
            groups = pd.Series(index=current_factors.index, dtype=str)
            for i in range(5):
                if i < 4:
                    start_idx = i * stocks_per_group
                    end_idx = (i + 1) * stocks_per_group
                else:
                    # 最后一组包含剩余所有股票
                    start_idx = i * stocks_per_group
                    end_idx = n_stocks
                
                group_stocks = current_factors.index[start_idx:end_idx]
                groups[group_stocks] = f'G{i+1}'
            
            # 打印分组信息（仅第一个交易日）
            if date == factor_data.index[0]:
                print(f"\n{factor_name} 因子分组情况（第一个交易日）:")
                for i in range(5):
                    group_stocks = groups[groups == f'G{i+1}'].index
                    group_values = current_factors[group_stocks]
                    print(f"G{i+1}组 (从低波动到高波动):")
                    print(f"股票数量: {len(group_stocks)}")
                    print(f"因子值范围: {group_values.min():.4f} 到 {group_values.max():.4f}")
                    print(f"因子值均值: {group_values.mean():.4f}")
                    print("------------------------")
            
            # 获取次日收益
            if date in stock_returns.index:
                next_day_returns = stock_returns.loc[date]
                dates_list.append(date)
                
                # 计算各组合收益
                for group_id in range(5):
                    group_stocks = groups[groups == f'G{group_id+1}'].index
                    if len(group_stocks) > 0:
                        group_return = next_day_returns[group_stocks].mean()
                        group_key = f'G{group_id+1}_{factor_name}'
                        if group_key not in daily_returns:
                            daily_returns[group_key] = []
                        daily_returns[group_key].append(float(group_return))
        
        # 创建净值DataFrame
        factor_navs = {}
        
        # 计算分组净值
        for group in [f'G{i+1}_{factor_name}' for i in range(5)]:
            returns = pd.Series(daily_returns[group], index=dates_list).fillna(0)
            nav = (1 + returns).cumprod()  # 计算净值
            factor_navs[group] = nav
        
        # 计算多空组合净值（做多G1低波动，做空G5高波动）
        g1_returns = pd.Series(daily_returns[f'G1_{factor_name}'], index=dates_list).fillna(0)
        g5_returns = pd.Series(daily_returns[f'G5_{factor_name}'], index=dates_list).fillna(0)
        long_short_returns = g1_returns - g5_returns  # 做多低波动，做空高波动
        long_short_nav = (1 + long_short_returns).cumprod()  # 计算多空组合净值
        factor_navs['Long-Short'] = long_short_nav
        
        daily_nav_dict[factor_name] = factor_navs
    
    # 创建Excel writer
    with pd.ExcelWriter('factor_daily_nav.xlsx') as writer:
        for factor_name, factor_data in daily_nav_dict.items():
            df = pd.DataFrame(factor_data)
            
            # 重命名列
            df.columns = [col.split('_')[0] if 'Long-Short' not in col else 'Long-Short' 
                         for col in df.columns]
            
            # 确保日期列正确显示
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            df.index.name = 'Date'
            
            # 写入Excel
            df.to_excel(writer, sheet_name=factor_name, float_format='%.4f')
            
            # 调整列宽
            worksheet = writer.sheets[factor_name]
            worksheet.column_dimensions['A'].width = 12
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(66 + idx)].width = max_length + 2
    
    print(f"日度净值已保存至 factor_daily_nav.xlsx")
    return daily_nav_dict

# 使用示例
if __name__ == "__main__":
    results = calculate_factor_returns(['EWMAVOL', 'VOL_3M'])
