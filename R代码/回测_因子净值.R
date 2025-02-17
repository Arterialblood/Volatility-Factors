# 设置工作目录（根据实际情况修改路径）
setwd("your_working_directory_path")

# 加载必要的库
library(data.table)
library(tidyverse)
library(zoo)
START_DATE <- as.Date("2021-01-01")
END_DATE <- as.Date("2024-11-01")
# 读取市场收益率数据
read_market_ret <- function() {
  market_ret <- fread("raw_data/hs300_index_returns.csv", key = "日期")
  setnames(market_ret, old = "日期", new = "date")
  market_ret[, date := as.Date(date)]
  return(market_ret)
}

# 读取个股收益率数据
read_stock_returns <- function() {
  stock_returns <- fread("raw_data/hs300_stock_returns.csv", key = "日期")
  setnames(stock_returns, old = "日期", new = "date")
  stock_returns[, date := as.Date(date)]
  stock_returns <- stock_returns[-1, ]  # 对应Python中去除第一行（如果需求一致的话）
  stock_returns[, lapply(.SD, as.numeric)]  # 转换列的数据类型为数值型，类似Python中的astype(float)
  return(stock_returns)
}

# 读取因子数据（以读取EWMAVOL因子为例，其他因子类似）
read_factors_data <- function(factor_name) {
  factor_path <- paste0("factors/", factor_name, ".csv")
  factor_data <- fread(factor_path, key = "日期")
  setnames(factor_data, old = "日期", new = "date")
  factor_data[, date := as.Date(date)]
  return(factor_data)
}

# 筛选数据在指定日期范围内
filter_dates <- function(data) {
  data <- data[date >= START_DATE & date <= END_DATE]
  return(data)
}

# 获取共同日期（用于对齐不同数据集的日期）
get_common_dates <- function(market_ret, stock_returns, factors_data_list) {
  all_dates <- market_ret[, date]
  for (factor_data in factors_data_list) {
    all_dates <- intersect(all_dates, factor_data[, date])
  }
  return(all_dates)
}

# 对齐数据到共同日期
align_data <- function(market_ret, stock_returns, factors_data_list, common_dates) {
  market_ret <- market_ret[date %in% common_dates]
  stock_returns <- stock_returns[date %in% common_dates]
  factors_data_list <- lapply(factors_data_list, function(factor_data) {
    factor_data[date %in% common_dates]
  })
  return(list(market_ret = market_ret, stock_returns = stock_returns, factors_data = factors_data_list))
}
# 缩尾处理函数
winsorize <- function(data, lower_percentile = 0.025, upper_percentile = 0.975) {
  if (is.vector(data)) {
    lower_bound <- quantile(data, lower_percentile)
    upper_bound <- quantile(data, upper_percentile)
    data[data < lower_bound] <- lower_bound
    data[data > upper_bound] <- upper_bound
    return(data)
  } else if (is.data.frame(data)) {
    result <- data
    for (i in seq_len(nrow(data))) {
      daily_data <- result[i, ]
      lower_bound <- quantile(daily_data, lower_percentile)
      upper_bound <- quantile(daily_data, upper_percentile)
      result[i, which(daily_data < lower_bound)] <- lower_bound
      result[i, which(daily_data > upper_bound)] <- upper_bound
    }
    return(result)
  }
}

# 标准化处理函数
standardize <- function(data) {
  if (is.vector(data)) {
    mean_value <- mean(data)
    std_value <- sd(data)
    if (std_value!= 0) {
      return((data - mean_value) / std_value)
    } else {
      return(data * 0)
    }
  } else if (is.data.frame(data)) {
    result <- data
    for (i in seq_len(nrow(data))) {
      daily_data <- result[i, ]
      mean_value <- mean(daily_data)
      std_value <- sd(daily_data)
      if (std_value!= 0) {
        result[i, ] <- (daily_data - mean_value) / std_value
      } else {
        result[i, ] <- 0
      }
    }
    return(result)
  }
}

# 完整的因子预处理流程
process_factor <- function(factor_data) {
  factor_data <- winsorize(factor_data)
  factor_data <- standardize(factor_data)
  factor_data <- winsorize(factor_data, lower_percentile = 0.01, upper_percentile = 0.99)
  factor_data <- standardize(factor_data)
  return(factor_data)
}
# 获取每月第一个交易日（对应Python中获取月度调仓日期的功能）
get_monthly_dates <- function(dates) {
  year_month <- format(dates, "%Y-%m")
  is_month_start <- year_month!= lag(year_month, default = first(year_month))
  monthly_dates <- dates[is_month_start]
  return(monthly_dates)
}

# 根据因子值将股票分组
form_portfolios <- function(factor_name, date, factor_data, stock_returns, n_groups = 5) {
  factor_values <- factor_data[date == factor_data[, date], factor_name, with = FALSE]
  factor_values <- factor_values[order(factor_values[[1]])]
  group_size <- nrow(factor_values) %/% n_groups
  
  print(paste0("分组信息 - 日期: ", date))
  print(paste0("总股票数: ", nrow(factor_values)))
  print(paste0("每组大约股票数: ", group_size))
  
  portfolios <- list()
  for (i in 0:(n_groups - 1)) {
    start_idx <- i * group_size + 1
    end_idx <- if (i < n_groups - 1) (i + 1) * group_size else nrow(factor_values)
    portfolio <- factor_values[start_idx:end_idx, which(colnames(factor_values) == factor_name)]
    group_factors <- factor_values[start_idx:end_idx, ]
    
    print(paste0("\n第", i + 1, "组:"))
    print(paste0("股票数量: ", length(portfolio)))
    print(paste0("因子值范围: ", min(group_factors[[1]]), " 到 ", max(group_factors[[1]])))
    print(paste0("因子值均值: ", mean(group_factors[[1]])))
    print(paste0("因子值标准差: ", sd(group_factors[[1]])))
    
    portfolios[[i + 1]] <- portfolio
  }
  
  return(portfolios)
}

# 计算组合在给定时间段的收益率
calculate_portfolio_returns <- function(portfolio_stocks, start_date, end_date, stock_returns) {
  period_returns <- stock_returns[date >= start_date & date <= end_date,..portfolio_stocks]
  
  print(paste0("组合收益率计算信息:"))
  print(paste0("期间: ", start_date, " 到 ", end_date))
  print(paste0("组合股票数: ", length(portfolio_stocks)))
  print(paste0("有效收益率数据的股票数: ", mean(rowSums(!is.na(period_returns)))))
  
  weight <- 1 / length(portfolio_stocks)
  weighted_returns <- rowMeans(period_returns, na.rm = TRUE)
  
  return(weighted_returns)
}

# 计算IC值
calculate_ic <- function(factor_name, factor_data, stock_returns, monthly_dates) {
  ic_series <- rep(NA, length(monthly_dates) - 1)
  names(ic_series) <- monthly_dates[-length(monthly_dates)]
  
  print("IC计算详细信息:")
  for (i in seq_len(length(monthly_dates) - 1)) {
    current_date <- monthly_dates[i]
    next_date <- monthly_dates[i + 1]
    
    current_factors <- factor_data[date == current_date, factor_name, with = FALSE]
    next_month_returns <- stock_returns[date > current_date & date <= next_date, lapply(.SD, mean)]
    
    common_stocks <- intersect(current_factors[, which(colnames(current_factors) == factor_name)], names(next_month_returns))
    if (length(common_stocks) > 0) {
      ic <- cor(current_factors[, common_stocks, with = FALSE][[1]], unlist(next_month_returns[common_stocks]))
      ic_series[current_date] <- ic
      
      print(paste0("\n日期: ", current_date))
      print(paste0("因子值数量: ", length(common_stocks)))
      print(paste0("因子值范围: ", min(current_factors[, common_stocks, with = FALSE][[1]]), " 到 ", max(current_factors[, common_stocks, with = FALSE][[1]])))
      print(paste0("下月收益率范围: ", min(next_month_returns[common_stocks]), " 到 ", max(next_month_returns[common_stocks])))
      print(paste0("IC值: ", ic))
    }
  }
  
  return(ic_series[!is.na(ic_series)])
}

# 计算RankIC值
calculate_rank_ic <- function(factor_name, factor_data, stock_returns, monthly_dates) {
  rank_ic_series <- rep(NA, length(monthly_dates) - 1)
  names(rank_ic_series) <- monthly_dates[-length(monthly_dates)]
  
  print("RankIC计算详细信息:")
  for (i in seq_len(length(monthly_dates) - 1)) {
    current_date <- monthly_dates[i]
    next_date <- monthly_dates[i + 1]
    
    current_factors <- factor_data[date == current_date, factor_name, with = FALSE]
    next_month_returns <- stock_returns[date > current_date & date <= next_date, lapply(.SD, mean)]
    
    common_stocks <- intersect(current_factors[, which(colnames(current_factors) == factor_name)], names(next_month_returns))
    if (length(common_stocks) > 0) {
      rank_ic <- cor(rank(current_factors[, common_stocks, with = FALSE][[1]]), rank(unlist(next_month_returns[common_stocks])))
      rank_ic_series[current_date] <- rank_ic
      
      print(paste0("\n日期: ", current_date))
      print(paste0("股票数量: ", length(common_stocks)))
      print(paste0("因子值排名范围: ", min(rank(current_factors[, common_stocks, with = FALSE][[1]])), " 到 ", max(rank(current_factors[, common_stocks, with = FALSE][[1]]))))
      print(paste0("收益排名范围: ", min(rank(unlist(next_month_returns[common_stocks]))), " 到 ", max(rank(unlist(next_month_returns[common_stocks])))))
      print(paste0("RankIC值: ", rank_ic))
    }
  }
  
  return(rank_ic_series[!is.na(rank_ic_series)])
}

# 计算绩效指标（对应Python中 `calculate_performance_metrics` 函数的功能）
calculate_performance_metrics <- function(returns) {
  cumulative_returns <- cumprod(1 + returns)
  total_return <- cumulative_returns[length(cumulative_returns)] - 1
  
  years <- length(returns) / 252
  annual_return <- (1 + total_return) ^ (1 / years) - 1
  
  annual_volatility <- sd(returns) * sqrt(252)
  
  sharpe_ratio <- if (annual_volatility!= 0) annual_return / annual_volatility else 0
  
  rolling_max <- rollapply(cumulative_returns, width = which.max(cumulative_returns), FUN = max, align = "right", fill = NA)
  drawdowns <- cumulative_returns / rolling_max - 1
  max_drawdown <- min(drawdowns, na.rm = TRUE)
  
  win_rate <- mean(returns > 0)
  
  return(list(annual_return = annual_return,
              annual_volatility = annual_volatility,
              sharpe_ratio = sharpe_ratio,
              max_drawdown = max_drawdown,
              win_rate = win_rate))
}

# 运行完整的回测（对应Python中 `run_backtest` 函数的功能）
run_backtest <- function(factor_name, integrated_data) {
  factor_data <- integrated_data$factors_data[[factor_name]]
  stock_returns <- integrated_data$stock_returns
  monthly_dates <- get_monthly_dates(integrated_data$market_ret[, date])
  
  all_dates <- integrated_data$market_ret[, date]
  portfolio_returns <- data.table(date = all_dates, matrix(NA, nrow = length(all_dates), ncol = 5))
  setnames(portfolio_returns, old = colnames(portfolio_returns)[-1], new = paste0("Group_", 1:5))
  
  print("开始回测:")
  print(paste0("月度调仓日期数量: ", length(monthly_dates)))
  
  for (i in seq_len(length(monthly_dates) - 1)) {
    start_date <- monthly_dates[i]
    end_date <- monthly_dates[i + 1]
    
    print(paste0("\n处理期间: ", start_date, " 到 ", end_date))
    
    portfolios <- form_portfolios(factor_name, start_date, factor_data, stock_returns)
    
    for (group_id in seq_along(portfolios)) {
      print(paste0("\n计算第", group_id, "组收益率"))
      returns <- calculate_portfolio_returns(portfolios[[group_id]], start_date, end_date, stock_returns)
      
      print(paste0("当期收益率日期范围: ", min(returns), " 到 ", max(returns)))
      print(paste0("收益率天数: ", length(returns)))
      
      mask <- portfolio_returns$date > start_date & portfolio_returns$date <= end_date
      portfolio_returns[mask, paste0("Group_", group_id)] <- returns
    }
  }
  
  print("Portfolio returns summary:")
  print(paste0("Shape: ", dim(portfolio_returns)))
  print(paste0("Columns: ", paste0(colnames(portfolio_returns), collapse = ", ")))
  print(head(portfolio_returns))
  
  ic <- calculate_ic(factor_name, factor_data, stock_returns, monthly_dates)
  rank_ic <- calculate_rank_ic(factor_name, factor_data, stock_returns, monthly_dates)
  
  long_short_returns <- portfolio_returns[, Group_1] - portfolio_returns[, Group_5]
  
  performance <- list()
  for (group in colnames(portfolio_returns)[-1]) {
    performance[[group]] <- calculate_performance_metrics(portfolio_returns[, get(group)])
  }
  performance[["long_short"]] <- calculate_performance_metrics(long_short_returns)
  
  return(list(returns = portfolio_returns,
              long_short_returns = long_short_returns,
              ic = ic,
              rank_ic = rank_ic,
              performance = performance))
}
# 计算超额收益相关指标
calculate_excess_metrics <- function(returns, market_returns) {
  print("市场收益率统计:")
  print(paste0("数据起始日期: ", min(market_returns$date)))
  print(paste0("数据结束日期: ", max(market_returns$date)))
  print(paste0("数据点数: ", nrow(market_returns)))
  print(paste0("日均收益率: ", mean(market_returns[, return])) )
  print(paste0("收益率标准差: ", sd(market_returns[, return])) )
  
  market_cum_returns <- cumprod(1 + market_returns[, return])
  market_total_return <- market_cum_returns[length(market_cum_returns)] - 1
  years <- nrow(market_returns) / 252
  market_annual_return <- (1 + market_total_return) ^ (1 / years) - 1
  
  print(paste0("市场总收益: ", market_total_return))
  print(paste0("年化收益率: ", market_annual_return))
  
  strategy_cum_returns <- cumprod(1 + returns)
  strategy_total_return <- strategy_cum_returns[length(strategy_cum_returns)] - 1
  years <- length(returns) / 252
  strategy_annual