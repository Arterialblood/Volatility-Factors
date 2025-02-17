import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import os
import matplotlib.font_manager as fm
import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side

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

# 筛选期范围
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
    
    def get_factor_data(self):
        return self.factors
    
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
        1. ��始缩尾处理
        2. 标准化
        3. 再次缩尾处理确保极端值
        """
        # 1. 始缩尾处理
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
        """初始化回测系统"""
        self.data = integrated_data
        self.returns = integrated_data.get_stock_returns()
        self.factor_data = integrated_data.get_factor_data()  # 获取因子数据
        self.market_returns = integrated_data.get_market_return()
        
        # 初始化因子预处理器
        self.preprocessor = FactorPreprocessor(self.factor_data)
        
        # 获取月度调仓日期
        self.monthly_dates = self.get_monthly_dates()
        
    def get_monthly_dates(self):
        """获取每月第一个交易日"""
        dates = pd.DatetimeIndex(self.data.dates)
        # 将日期转换为年月字符串
        year_month = dates.strftime('%Y-%m')
        # ���出每个月的第一个交易日
        is_month_start = year_month != pd.Series(year_month).shift(1)
        monthly_dates = dates[is_month_start]
        return monthly_dates.sort_values()
    
    def form_portfolios(self, factor_name, date):
        """构建投资组合，考虑U型特征"""
        # 获取当前日期的因子值
        factor_data = self.factor_data[factor_name].loc[date].dropna()
        
        # 计算因子分位数（10个分组）
        factor_quantiles = pd.qcut(factor_data, q=10, labels=False)
        
        # 考虑U型特征，重新组合分组
        portfolios = []
        
        # Group 1: 组合最高和最低分位数
        group1_stocks = factor_data[
            (factor_quantiles == 0) | (factor_quantiles == 9)
        ].index.tolist()
        portfolios.append(group1_stocks)
        
        # Group 2: 组合第2和第9分位
        group2_stocks = factor_data[
            (factor_quantiles == 1) | (factor_quantiles == 8)
        ].index.tolist()
        portfolios.append(group2_stocks)
        
        # Group 3: 组合第3和第8分位
        group3_stocks = factor_data[
            (factor_quantiles == 2) | (factor_quantiles == 7)
        ].index.tolist()
        portfolios.append(group3_stocks)
        
        # Group 4: 组合第4和第7分位
        group4_stocks = factor_data[
            (factor_quantiles == 3) | (factor_quantiles == 6)
        ].index.tolist()
        portfolios.append(group4_stocks)
        
        # Group 5: 组合中间分位数
        group5_stocks = factor_data[
            (factor_quantiles == 4) | (factor_quantiles == 5)
        ].index.tolist()
        portfolios.append(group5_stocks)
        
        # 打印每个组合的股票数量
        for i, portfolio in enumerate(portfolios):
            print(f"Group {i+1} size: {len(portfolio)}")
        
        return portfolios
    
    def calculate_portfolio_returns(self, portfolio, start_date, end_date):
        """计算组合收益，考虑U型特征"""
        # 获取日期范围内的收益率数据
        mask = (self.returns.index > start_date) & (self.returns.index <= end_date)
        period_returns = self.returns.loc[mask]
        
        # 计算组合收益
        if len(portfolio) > 0:
            # 对于U型策略，我们可能需要根据因子值调整权重
            portfolio_returns = period_returns[portfolio].mean(axis=1)
        else:
            portfolio_returns = pd.Series(0, index=period_returns.index)
        
        return portfolio_returns
    
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
                    print(f"因子数量: {len(common_stocks)}")
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
                
                # 确保因���值和收益率的股票代码对应
                common_stocks = current_factors.index.intersection(next_month_returns.index)
                if len(common_stocks) > 0:
                    # 计算秩相关系数
                    rank_ic = current_factors[common_stocks].rank().corr(next_month_returns[common_stocks].rank())
                    rank_ic_series[current_date] = rank_ic
                    
                    print(f"\n日期: {current_date}")
                    print(f"股票数: {len(common_stocks)}")
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
            
            # 获分组
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
        
        # 算多空收益
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
    
    def visualize_u_shape(self, factor_name):
        """可视化因子值与收益率的U型关系"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import rcParams
        import matplotlib.font_manager as fm
        
        # 查找系统中的中文字体
        def find_chinese_font():
            fonts = [
                '/System/Library/Fonts/PingFang.ttc',  # macOS
                '/System/Library/Fonts/STHeiti Light.ttc',  # macOS
                '/System/Library/Fonts/Hiragino Sans GB.ttc',  # macOS
                '/Library/Fonts/Arial Unicode.ttf',  # macOS
                '/Library/Fonts/Microsoft/SimHei.ttf'  # Windows
            ]
            
            for font_path in fonts:
                try:
                    if os.path.exists(font_path):
                        return font_path
                except:
                    continue
            return None
        
        # 设置中文字体
        chinese_font_path = find_chinese_font()
        if chinese_font_path:
            chinese_font = fm.FontProperties(fname=chinese_font_path)
        else:
            print("警告：未找到中文字体，可能会显示为方框")
            chinese_font = fm.FontProperties()
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 设置整体风格
        sns.set_style("whitegrid")
        
        # 获取数据并处理
        factor_values = self.preprocessor.process_factor(factor_name)
        returns_data = []
        factor_data = []
        
        for i in range(len(self.monthly_dates)-1):
            current_date = self.monthly_dates[i]
            next_date = self.monthly_dates[i+1]
            
            if current_date in factor_values.index:
                current_factors = factor_values.loc[current_date]
                mask = (self.returns.index > current_date) & (self.returns.index <= next_date)
                next_month_returns = self.returns[mask].mean()
                common_stocks = current_factors.index.intersection(next_month_returns.index)
                factor_data.extend(current_factors[common_stocks].values)
                returns_data.extend(next_month_returns[common_stocks].values)
        
        factor_data = np.array(factor_data)
        returns_data = np.array(returns_data)
        
        # 1. 散点图
        plt.subplot(2, 1, 1)
        plt.scatter(factor_data, returns_data, alpha=0.3, s=2, color='darkblue', label='Individual Returns')
        
        # 添加趋势线
        z = np.polyfit(factor_data, returns_data, 2)
        p = np.poly1d(z)
        x_sorted = np.sort(factor_data)
        plt.plot(x_sorted, p(x_sorted), 'r--', linewidth=2, label='Quadratic Fit')
        
        plt.title(f'{factor_name} Factor Value vs Future Returns', fontproperties=chinese_font, fontsize=14, pad=15)
        plt.xlabel('Standardized Factor Value', fontproperties=chinese_font, fontsize=12)
        plt.ylabel('Future Returns', fontproperties=chinese_font, fontsize=12)
        plt.legend(prop=chinese_font, fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 2. 分组平均收益图
        plt.subplot(2, 1, 2)
        factor_quantiles = pd.qcut(factor_data, q=10, labels=False)
        mean_returns = [returns_data[factor_quantiles == i].mean() for i in range(10)]
        
        # 使用渐变
        colors = plt.cm.coolwarm(np.linspace(0, 1, 10))
        bars = plt.bar(range(10), mean_returns, color=colors)
        
        plt.title('Average Returns by Factor Decile', fontproperties=chinese_font, fontsize=14, pad=15)
        plt.xlabel('Factor Decile', fontproperties=chinese_font, fontsize=12)
        plt.ylabel('Average Return', fontproperties=chinese_font, fontsize=12)
        
        # 添加��值标签
        for i, v in enumerate(mean_returns):
            plt.text(i, v, f'{v:.2%}', ha='center', va='bottom' if v > 0 else 'top', 
                    fontsize=10, color='black')
        
        # 设置x轴刻度标签
        plt.xticks(range(10), [f'P{i+1}' for i in range(10)], fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(f'{factor_name}_u_shape_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print("\nU-Shape Feature Analysis:")
        print(f"Quadratic Term: {z[0]:.6f}")
        print(f"Linear Term: {z[1]:.6f}")
        print(f"Constant Term: {z[2]:.6f}")
        print(f"Extreme Point: {-z[1]/(2*z[0]):.4f}")
        print(f"U-Shape Strength: {((mean_returns[0] + mean_returns[-1])/2 - mean_returns[4]):.4%}")

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
        
        # 超额��大回撤
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
                '相��胜率': f"{long_excess['relative_win_rate']:.2%}",
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
        print(f"信息比率: {group1_excess['information_ratio']:.2f}")
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
        print("收益率数据:")
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
        """生成因子分析报告输出到Excel"""
        if output_path is None:
            output_path = f'factor_analysis_{factor_name}.xlsx'
        
        # 首先打印结果结构以便调试
        print("Results structure:", self.results.keys())
        
        # 获取多头组合收益
        group1_returns = self.results['raw_results']['returns']['Group_1']
        group1_perf = self.calculate_performance_metrics(group1_returns)
        group1_excess = self.calculate_excess_metrics(group1_returns)
        
        # 获取多空组合收益
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
                'RankIC值': f"{self.results['raw_results']['rank_ic'].mean():.4f}",
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

    def export_detailed_results(self, factor_name, output_path=None):
        """导出详细的��测结果到Excel"""
        if output_path is None:
            output_path = f'backtest_details_{factor_name}.xlsx'
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 导出每日收益率数据
            returns_df = pd.DataFrame({
                'Group_1': self.results['raw_results']['returns']['Group_1'],
                'Group_2': self.results['raw_results']['returns']['Group_2'],
                'Group_3': self.results['raw_results']['returns']['Group_3'],
                'Group_4': self.results['raw_results']['returns']['Group_4'],
                'Group_5': self.results['raw_results']['returns']['Group_5'],
                'Long_Short': self.results['raw_results']['long_short_returns'],
                'Market': self.market_returns['收盘']
            })
            returns_df.to_excel(writer, sheet_name='每日收益率')
            
            # 2. 导出IC数据
            ic_df = pd.DataFrame({
                'IC': self.results['raw_results']['ic'],
                'RankIC': self.results['raw_results']['rank_ic']
            })
            ic_df.to_excel(writer, sheet_name='IC分析')
            
            # 3. 导出累积收益
            cumulative_returns = (1 + returns_df).cumprod()
            cumulative_returns.to_excel(writer, sheet_name='累积收益')
            
            # 4. 导出月度统计 (分开计算不同的统计量)
            monthly_returns = returns_df.resample('ME').sum()
            monthly_win_rates = returns_df.resample('ME').apply(lambda x: (x > 0).mean())
            monthly_vol = returns_df.resample('ME').std()
            
            # 合并月度统计结果
            monthly_stats = pd.DataFrame()
            for col in returns_df.columns:
                monthly_stats[f'{col}_收益率'] = monthly_returns[col]
                monthly_stats[f'{col}_胜率'] = monthly_win_rates[col]
                monthly_stats[f'{col}_波动率'] = monthly_vol[col]
            
            monthly_stats.to_excel(writer, sheet_name='月度统计')
            
            # 调整所有sheet的列宽
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    for cell in column:
                        try:
                            if hasattr(cell, 'column_letter'):
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                        except:
                            continue
                    if max_length > 0:
                        column_letter = column[0].column_letter
                        worksheet.column_dimensions[column_letter].width = max_length + 2
        
        print(f"\n详细回测结果已保存至: {output_path}")

def run_factor_analysis(factor_name, integrated_data):
    """运行完整的因子分析"""
    # 创建回测系统
    backtest = BacktestSystem(integrated_data)
    
    # 运行回测
    results = backtest.run_backtest(factor_name)
    
    # 分析结果
    analyzer = ResultAnalyzer(results, integrated_data.get_market_return())
    
    # 生成表格
    performance_table = analyzer.generate_performance_table()
    ic_table = analyzer.generate_ic_table()
    
    return {
        'performance': performance_table,
        'ic': ic_table,
        'raw_results': results
    }

def analyze_multiple_factors(integrated_data, factor_names):
    """分析多个因子并生成汇总表格"""
    # 创建回测系统
    backtest = BacktestSystem(integrated_data)
    
    # 初始化结果存储 - 确保键名完全匹配
    results_summary = {
        '因子': [],
        '多头策略年化收益': [],  # 删除下划线
        '多头策略超额收益': [],
        '多头策略夏普比率': [],
        '多头策略超额回撤': [],
        '多头策略相对胜率': [],
        '多头策略信息比率': [],
        'U型策略年化收益': [],
        'U型策略超额收益': [],
        'U型策略夏普比率': [],
        'U型策略超额回撤': [],
        'U型策略相对胜率': [],
        'U型策略信息比率': []
    }
    
    for factor_name in factor_names:
        print(f"\n分析因子: {factor_name}")
        
        # 运行回测
        results = backtest.run_backtest(factor_name)
        
        # 创建结果分析器
        analyzer = ResultAnalyzer(results, integrated_data.get_market_return())
        
        # 生成U型特征可视化
        backtest.visualize_u_shape(factor_name)
        
        # 获取多头策略（Group_1）的表现
        group1_returns = results['returns']['Group_1']
        group1_perf = analyzer.calculate_performance_metrics(group1_returns)
        group1_excess = analyzer.calculate_excess_metrics(group1_returns)
        
        # 获取U型策略的表现（Group_1 - Group_5）
        u_shape_returns = results['long_short_returns']
        u_shape_perf = analyzer.calculate_performance_metrics(u_shape_returns)
        u_shape_excess = analyzer.calculate_excess_metrics(u_shape_returns)
        
        # 添加结果到汇总
        results_summary['因子'].append(factor_name)
        
        # 多头策略结果
        results_summary['多头策略年化收益'].append(f"{group1_perf['annual_return']:.2%}")
        results_summary['多头策略超额收益'].append(f"{group1_excess['excess_return']:.2%}")
        results_summary['多头策略夏普比率'].append(f"{group1_perf['sharpe_ratio']:.2f}")
        results_summary['多头策略超额回撤'].append(f"{group1_excess['excess_drawdown']:.2%}")
        results_summary['多头策略相对胜率'].append(f"{group1_excess['relative_win_rate']:.2%}")
        results_summary['多头策略信息比率'].append(f"{group1_excess['information_ratio']:.2f}")
        
        # U型策略结果
        results_summary['U型策略年化收益'].append(f"{u_shape_perf['annual_return']:.2%}")
        results_summary['U型策略超额收益'].append(f"{u_shape_excess['excess_return']:.2%}")
        results_summary['U型策略夏普比率'].append(f"{u_shape_perf['sharpe_ratio']:.2f}")
        results_summary['U型策略超额回撤'].append(f"{u_shape_excess['excess_drawdown']:.2%}")
        results_summary['U型策略相对胜率'].append(f"{u_shape_excess['relative_win_rate']:.2%}")
        results_summary['U型策略信息比率'].append(f"{u_shape_excess['information_ratio']:.2f}")
    
    # 创建汇总DataFrame
    summary_df = pd.DataFrame(results_summary)
    
    # 保存到Excel，使用更好的格式
    output_path = 'factors_analysis_summary.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='因子分析汇总', index=False)
        
        # 获取工作表
        worksheet = writer.sheets['因子分析汇总']
        
        # 设置列宽
        for idx, col in enumerate(summary_df.columns):
            max_length = max(
                summary_df[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        # 设置标题行格式
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center')
        
        # 设置数据行格式
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(horizontal='center')
    
    print(f"\n分析结果已保存至: {output_path}")
    return summary_df

# 在主程序中添加以下��入
import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side

if __name__ == "__main__":
    # 检查点1：数据加载
    print("1. 数据加载检查:")
    print(f"市场收益率数据形状: {market_ret.shape}")
    print(f"市场收益率日期范围: {market_ret.index.min()} 到 {market_ret.index.max()}")
    for name, data in factors_data.items():
        print(f"{name}因子数据形状: {data.shape}")
        print(f"{name}因子日期范围: {data.index.min()} 到 {data.index.max()}")

    # 检查点2：共同日期
    print("\n2. 共同日期检查:")
    print(f"共同日期数量: {len(common_dates)}")
    print(f"共同日期范围: {common_dates.min()} 到 {common_dates.max()}")

    # 检查点3：因子预处理
    print("\n3. 因子预处理检查:")
    factor_name = 'EWMAVOL'
    preprocessor = FactorPreprocessor(factors_data)
    processed_factor = preprocessor.process_factor(factor_name)
    print(f"原始因子数据形状: {factors_data[factor_name].shape}")
    print(f"处理后因子数据形状: {processed_factor.shape}")
    
    # 修改计方法
    sample_date = processed_factor.index[0]
    sample_data = processed_factor.loc[sample_date]
    print(f"\n处理后因子样本统计 (日期: {sample_date}):")
    print(f"均值: {sample_data.mean():.4f}")
    print(f"标准差: {sample_data.std():.4f}")
    print(f"最小值: {sample_data.min():.4f}")
    print(f"最大值: {sample_data.max():.4f}")
    print(f"中位数: {sample_data.median():.4f}")
    print(f"25分位数: {sample_data.quantile(0.25):.4f}")
    print(f"75分位数: {sample_data.quantile(0.75):.4f}")
    print(f"2.5分位数: {sample_data.quantile(0.025):.4f}")
    print(f"97.5分位数: {sample_data.quantile(0.975):.4f}")

    # 检查点4：投资组合构建
    print("\n4. 投资组合构建查:")
    backtest = BacktestSystem(integrated_data)
    sample_date = common_dates[0]
    portfolios = backtest.form_portfolios(factor_name, sample_date)
    print(f"分组数量: {len(portfolios)}")
    
    # 检查每组详细息
    for i, portfolio in enumerate(portfolios):
        group_factor_values = processed_factor.loc[sample_date][portfolio]
        print(f"\n第{i+1}组:")
        print(f"股票数量: {len(portfolio)}")
        print(f"因子值范围: {group_factor_values.min():.4f} 到 {group_factor_values.max():.4f}")
        print(f"因子值均: {group_factor_values.mean():.4f}")
        print(f"因子值标准差: {group_factor_values.std():.4f}")

    # 检查点5：收益率计算
    print("\n5. 收益率计算检查:")
    monthly_dates = backtest.monthly_dates
    print(f"月度调仓日期数量: {len(monthly_dates)}")
    print("\n月度调仓日期示例（前12个月）:")
    for date in monthly_dates[:12]:
        print(date.strftime('%Y-%m-%d'))
    
    # 计算第个的收益率
    if len(monthly_dates) > 1:
        first_portfolio = portfolios[0]
        start_date = monthly_dates[0]
        end_date = monthly_dates[1]
        first_month_returns = backtest.calculate_portfolio_returns(
            first_portfolio, 
            start_date, 
            end_date
        )
        
        print("\n第一组第一个月收益率统计:")
        print(f"收益率计算期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        print(f"交易天数: {len(first_month_returns)}")
        print(f"日均收益率: {first_month_returns.mean():.4%}")
        if len(first_month_returns) > 1:
            print(f"收益率标准差: {first_month_returns.std():.4%}")
        print(f"月度累计收益率: {(1 + first_month_returns).prod() - 1:.4%}")
        print(f"最大日收益率: {first_month_returns.max():.4%}")
        print(f"最小日收益率: {first_month_returns.min():.4%}")
        
        print("\n每日收益率明细:")
        for date, ret in first_month_returns.items():
            print(f"{date.strftime('%Y-%m-%d')}: {ret:.4%}")

    # 检查点6：完整回测结果
    print("\n6. 完整回测结果检查:")
    factor_name = 'EWMAVOL'
    
    # 调试信息
    print("\n数据检查:")
    print("市场收益率类型:", type(market_ret.index[0]))
    print("因子数据类型:", type(factors_data[factor_name].index[0]))
    print("\n市场收益率日期示例:", market_ret.index[:5])
    print("因子数据日期示:", factors_data[factor_name].index[:5])
    
    results = backtest.run_backtest(factor_name)
    
    print("\n分组收益率统计:")
    for group in ['Group_1', 'Group_2', 'Group_3', 'Group_4', 'Group_5']:
        returns = results['returns'][group]
        print(f"\n{group}:")
        print(f"年化收益率: {(1 + returns.mean())**252 - 1:.4%}")
        print(f"年化波动率: {returns.std() * np.sqrt(252):.4%}")
        print(f"夏普比率: {((1 + returns.mean())**252 - 1) / (returns.std() * np.sqrt(252)):.4f}")
        print(f"最大回撤: {(returns.cumsum() - returns.cumsum().expanding().max()).min():.4%}")
    
    print("\nIC统计:")
    ic = results['ic']
    print(f"IC均值: {ic.mean():.4f}")
    print(f"IC标准差: {ic.std():.4f}")
    print(f"ICIR: {ic.mean()/ic.std():.4f}")
    print(f"IC>0占比: {(ic > 0).mean():.4%}")

    # 运行因子分析
    results = run_factor_analysis('EWMAVOL', integrated_data)
    analyzer = ResultAnalyzer(results, integrated_data.get_market_return())
    analyzer.generate_factor_analysis_report('factor_name')  # 会同时打印结果并保存到Excel
    analyzer.export_detailed_results('factor_name')  # 导出详细回测结果

    # 在主程序中添加以下代码
    if __name__ == "__main__":
        # 假设已经创建了integrated_data和backtest对象
        
        # 创建回测系统
        backtest = BacktestSystem(integrated_data)
        
        # 指定因子名称
        factor_name = 'EWMAVOL'  # 或其他您想分析的因子名称
        
        # 运行回测
        results = backtest.run_backtest(factor_name)
        
        # 可视化U型特征
        backtest.visualize_u_shape(factor_name)
        
        # 图像会保存为 'EWMAVOL_u_shape_analysis.png'
        print(f"图像已保存为: {factor_name}_u_shape_analysis.png")

    # 分析多个因子
    factors_to_analyze = ['EWMAVOL', 'GARCHVOL']
    
    # 运行多因子分析并生成Excel报告
    summary_table = analyze_multiple_factors(integrated_data, factors_to_analyze)
    
    print("\n因子分析汇总表格:")
    print(summary_table.to_string())