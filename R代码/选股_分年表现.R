# 设置日期范围
START_DATE <- as.Date("2021-01-01")
END_DATE <- as.Date("2024-11-01")

# 读取数据函数
read_data <- function(file_path, index_col = NULL) {
  data <- read_csv(file_path)
  if (!is.null(index_col)) {
    data <- data %>% tibble::column_to_rownames(index_col) %>% as.data.frame()
    data[, ] <- lapply(data[, ], function(x) as.numeric(as.character(x)))
    data <- zoo::zoo(data)
  }
  return(data)
}
# 读取市场收益率数据
market_ret <- read_data("raw_data/hs300_index_returns.csv", index_col = "日期")
# 读取个股收益率数据并去除第一行（根据Python代码逻辑）
stock_returns <- read_data("raw_data/hs300_stock_returns.csv", index_col = "日期")[-1, ]

# 读取因子数据
factors_data <- list(
  EWMAVOL = read_data("factors/EWMAVOL.csv", index_col = "日期"),
  GARCHVOL = read_data("factors/GARCHVOL.csv", index_col = "日期"),
  RANKVOL = read_data("factors/rankvol_processed.csv", index_col = "日期"),
  RVOL = read_data("factors/RVOL.csv", index_col = "日期"),
  VOL_3M = read_data("factors/vol_3m_processed.csv", index_col = "日期")
)

# 筛选日期范围
market_ret <- market_ret[paste0(START_DATE, "/", END_DATE)]
stock_returns <- stock_returns[paste0(START_DATE, "/", END_DATE)]
factors_data <- lapply(factors_data, function(x) x[paste0(START_DATE, "/", END_DATE)])

# 获取所有数据集的共同日期
common_dates <- Reduce(intersect, lapply(list(market_ret, factors_data), function(x) index(x)))

# 对齐所有数据到共同日期
market_ret <- market_ret[common_dates]
factors_data <- lapply(factors_data, function(x) x[common_dates])

# 打印调试信息
cat("数据读取调试信息:\n")
cat("市场收益率数据:\n")
print(head(market_ret))
cat("\n因子数据 (EWMAVOL):\n")
print(head(factors_data$EWMAVOL))
cat("\n个股收益率数据:\n")
print(head(stock_returns))

cat("\n共同日期数量:", length(common_dates), "\n")
if (length(common_dates) > 0) {
  cat("第一个日期:", common_dates[1], "\n")
  cat("最后一个日期:", common_dates[length(common_dates)], "\n")
}
# 整合后的数据对象类
IntegratedFactorData <- setClass(
  "IntegratedFactorData",
  slots = list(
    factors = "list",
    market_returns = "zoo",
    stock_returns = "zoo",
    dates = "Date"
  )
)

# 创建整合后的数据对象实例的函数
create_integrated_data <- function(factors_data, market_ret, stock_returns, common_dates) {
  integrated_data <- new("IntegratedFactorData",
                         factors = factors_data,
                         market_returns = market_ret,
                         stock_returns = stock_returns,
                         dates = common_dates)
  return(integrated_data)
}

integrated_data <- create_integrated_data(factors_data, market_ret, stock_returns, common_dates)
# 因子预处理类
FactorPreprocessor <- setClass(
  "FactorPreprocessor",
  slots = list(
    factor_data = "list"
  )
)

# 缩尾处理函数
setMethod("winsorize", signature("FactorPreprocessor", "ANY"), function(factor_preprocessor, data, lower_percentile = 0.025, upper_percentile = 0.975) {
  if (is.vector(data)) {
    lower_bound <- quantile(data, lower_percentile)
    upper_bound <- quantile(data, upper_percentile)
    return(pmin(pmax(data, lower_bound), upper_bound))
  } else {
    result <- data
    for (date in index(data)) {
      daily_data <- coredata(result[date, ])
      lower_bound <- quantile(daily_data, lower_percentile)
      upper_bound <- quantile(daily_data, upper_percentile)
      result[date, ] <- pmin(pmax(daily_data, lower_bound), upper_bound)
    }
    return(result)
  }
})

# 标准化处理函数
setMethod("standardize", signature("FactorPreprocessor", "ANY"), function(factor_preprocessor, data) {
  if (is.vector(data)) {
    mean_val <- mean(data)
    std_val <- sd(data)
    return(if (std_val!= 0) (data - mean_val) / std_val else data * 0)
  } else {
    result <- zoo(matrix(0, nrow = nrow(data), ncol = ncol(data)), order.by = index(data))
    for (date in index(data)) {
      daily_data <- coredata(data[date, ])
      mean_val <- mean(daily_data)
      std_val <- sd(daily_data)
      result[date, ] <- if (std_val!= 0) (daily_data - mean_val) / std_val else 0
    }
    return(result)
  }
})

# 完整的因子预处理流程函数
setMethod("process_factor", signature("FactorPreprocessor", "character"), function(factor_preprocessor, factor_name) {
  factor <- winsorize(factor_preprocessor, factor_preprocessor@factor_data[[factor_name]])
  factor <- standardize(factor_preprocessor, factor)
  factor <- winsorize(factor_preprocessor, factor, lower_percentile = 0.01, upper_percentile = 0.99)
  factor <- standardize(factor_preprocessor, factor)
  return(factor)
})

preprocessor <- new("FactorPreprocessor", factor_data = integrated_data@factors)
# 回测系统类
BacktestSystem <- setClass(
  "BacktestSystem",
  slots = list(
    data = "IntegratedFactorData",
    preprocessor = "FactorPreprocessor",
    monthly_dates = "Date"
  )
)

# 获取每月第一个交易日的函数（辅助函数）
get_monthly_dates <- function(dates) {
  year_month <- format(dates, "%Y-%m")
  is_month_start <- year_month!= c("", year_month[-length(year_month)])
  monthly_dates <- dates[is_month_start]
  return(monthly_dates)
}

# 回测系统类的初始化函数
setMethod("initialize", "BacktestSystem", function(.Object, integrated_data) {
  .Object@data <- integrated_data
  .Object@preprocessor <- new("FactorPreprocessor", factor_data = integrated_data@factors)
  .Object@monthly_dates <- get_monthly_dates(integrated_data@dates)
  return(.Object)
})

# 根据因子值将股票分组的函数
setMethod("form_portfolios", signature("BacktestSystem", "character", "Date", "numeric"), function(backtest_system, factor_name, date, n_groups = 5) {
  factor_values <- process_factor(backtest_system@preprocessor, factor_name)[date, ]
  sorted_stocks <- sort(factor_values, index.return = TRUE)
  group_size <- floor(length(sorted_stocks) / n_groups)
  
  cat("\n分组信息 - 日期:", date, "\n")
  cat("总股票数:", length(sorted_stocks), "\n")
  cat("每组大约股票数:", group_size, "\n")
  
  portfolios <- list()
  for (i in 0:(n_groups - 1)) {
    start_idx <- i * group_size + 1
    end_idx <- if (i < n_groups - 1) (i + 1) * group_size else length(sorted_stocks)
    portfolio <- names(sorted_stocks)[start_idx:end_idx]
    
    group_factors <- sorted_stocks[start_idx:end_idx]
    cat("\n第", i + 1, "组:\n")
    cat("股票数量:", length(portfolio), "\n")
    cat("因子值范围:", min(group_factors), "到", max(group_factors), "\n")
    cat("因子值均值:", mean(group_factors), "\n")
    cat("因子值标准差:", sd(group_factors), "\n")
    
    portfolios[[i + 1]] <- portfolio
  }
  
  return(portfolios)
})

# 计算组合在给定时间段的收益率的函数
setMethod("calculate_portfolio_returns", signature("BacktestSystem", "list", "Date", "Date"), function(backtest_system, portfolio_stocks, start_date, end_date) {
  stock_rets <- backtest_system@data@stock_returns
  period_returns <- stock_rets[paste0(">", start_date, "/", end_date)]
  portfolio_returns <- period_returns[, portfolio_stocks, drop = FALSE]
  
  cat("\n组合收益率计算信息:\n")
  cat("期间:", start_date, "到", end_date, "\n")
  cat("组合股票数:", length(portfolio_stocks), "\n")
  cat("有效收益率数据的股票数:", sum(!is.na(portfolio_returns)), "\n")
  
  weight <- 1 / length(portfolio_stocks)
  weighted_returns <- rowMeans(portfolio_returns, na.rm = TRUE)
  
  return(weighted_returns)
})

# 计算IC值的函数
setMethod("calculate_ic", signature("BacktestSystem", "character", "zoo"), function(backtest_system, factor_name, forward_returns) {
  factor_values <- process_factor(backtest_system@preprocessor, factor_name)
  ic_series <- zoo(, order.by = backtest_system@monthly_dates[-length(backtest_system@monthly_dates)])
  
  cat("\nIC计算详细信息:\n")
  for (i in 1:(length(backtest_system@monthly_dates) - 1)) {
    current_date <- backtest_system@monthly_dates[i]
    next_date <- backtest_system@monthly_dates[i + 1]
    
    if (current_date %in% index(factor_values)) {
      current_factors <- factor_values[current_date, ]
      mask <- index(backtest_system@data@stock_returns) > current_date &
        index(backtest_system@data@stock_returns) <= next_date
      next_month_returns <- rowMeans(backtest_system@data@stock_returns[mask, ], na.rm = TRUE)
      
      common_stocks <- intersect(names(current_factors), names(next_month_returns))
      if (length(common_stocks) > 0) {
        ic <- cor(current_factors[common_stocks], next_month_returns[common_stocks])
        ic_series[current_date] <- ic
        
        cat("\n日期:", current_date, "\n")
        cat("因子值数量:", length(common_stocks), "\n")
        cat("因子值范围:", min(current_factors[common_stocks]), "到", max(current_factors[common_stocks]), "\n")
        cat("下月收益率范围:", min(next_month_returns[common_stocks]), "到", max(next_month_returns[common_stocks]), "\n")
        cat("IC值:", ic, "\n")
      }
    }
  }
  
  return(ic_series[!is.na(ic_series)])
})

# 计算RankIC值的函数
setMethod("calculate_rank_ic", signature("BacktestSystem", "character", "zoo"), function(backtest_system, factor_name, forward_returns) {
  factor_values <- process_factor(backtest_system@preprocessor, factor_name)
  rank_ic_series <- zoo(, order.by = backtest_system@monthly_dates[-length(backtest_system@monthly_dates)])
  
  cat("\nRankIC计算详细信息:\n")
  for (i in 1:(length(backtest_system@monthly_dates) - 1)) {
    current_date <- backtest_system@monthly_dates[i]
    next_date <- backtest_system@monthly_dates[i + 1]
    
    if (current_date %in% index(factor_values)) {
      current_factors <- factor_values[current_date, ]
      mask <- index(backtest_system@data@stock_returns) > current_date &
        index(backtest_system@data@stock_returns) <= next_date
      next_month_returns <- rowMeans(backtest_system@data@stock_returns[mask, ], na.rm = TRUE)
      
      common_stocks <- intersect(names(current_factors), names(next_month_returns))
      if (length(common_stocks) > 0) {
        rank_ic <- cor(rank(current_factors[common_stocks]), rank(next_month_returns[common_stocks]))
        rank_ic_series[current_date] <- rank_ic
        
        cat("\n日期:", current_date, "\n")
        cat("股票数量:", length(common_stocks), "\n")
        cat("因子值排名范围:", min(rank(current_factors[common_stocks])), "到", max(rank(current_factors[common_stocks])), "\n")
        cat("收益排名范围:", min(rank(next_month_returns[common_stocks])), "to", max(rank(next_month_returns[common_stocks])), "\n")
        cat("RankIC值:", rank_ic, "\n")
      }
    }
  }
  
  return(rank_ic_series[!is.na(rank_ic_series)])
})

# 计算绩效指标的函数
setMethod("calculate_performance_metrics", signature("BacktestSystem", "zoo"), function(backtest_system, returns) {
  cumulative_returns <- cumprod(1 + returns)
  total_return <- cumulative_returns[length(cumulative_returns)] - 1
  
  years <- length(returns) / 252
  annual_return <- (1 + total_return) ^ (1 / years) - 1
  
  annual_volatility <- sd(returns) * sqrt(252)
  
  sharpe_ratio <- annual_return / annual_volatility
  sharpe_ratio[is.na(sharpe_ratio) | annual_volatility == 0] <- 0
  
  rolling_max <- rollapply(cumulative_returns, width = length(cumulative_returns), FUN = max, align = "right", fill = NA)
  drawdowns <- cumulative_returns / rolling_max - 1
  max_drawdown <- min(drawdowns, na.rm = TRUE)
  
  win_rate <- mean(returns > 0)
  
  return(list(
    annual_return = annual_return,
    annual_volatility = annual_volatility,
    sharpe_ratio = sharpe_ratio,
    max_drawdown = max_drawdown,
    win_rate = win_rate
  ))
}

# 运行完整回测的函数
setMethod("run_backtest", signature("BacktestSystem", "character"), function(backtest_system, factor_name) {
  backtest_system@current_factor <- factor_name
  monthly_dates <- backtest_system@monthly_dates
  
  all_dates <- index(backtest_system@data@market_returns)
  portfolio_returns <- zoo(matrix(NA, nrow = length(all_dates), ncol = 0), order.by = all_dates)
  
  cat("\n开始回测:\n")
  cat("月度调仓日期数量:", length(monthly_dates), "\n")
  
  for (i in 1:(length(monthly_dates) - 1)) {
    start_date <- monthly_dates[i]
    end_date <- monthly_dates[i + 1]
    
    cat("\n处理期间:", start_date, "到", end_date, "\n")
    
    portfolios <- form_portfolios(backtest_system, factor_name, start_date)
    
    for (group_id in 1:length(portfolios)) {
      cat("\n计算第", group_id, "组收益率\n")
      returns <- calculate_portfolio_returns(backtest_system, portfolios[[group_id]], start_date, end_date)
      
      cat("当期收益率日期范围:", start(returns), "到", end(returns), "\n")
      cat("收益率天数:",