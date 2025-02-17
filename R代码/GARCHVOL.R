# 加载必要的包
library(tidyverse)
library(rugarch)
library(fpp2)
library(parallel)
library(progress)

# 定义计算GARCH波动率的函数
calculate_garch_vol <- function(stock, data) {
  predictions <- rep(NA, nrow(data))
  for (i in 60:nrow(data)) {
    window_data <- data[(i - 60 + 1):i, ]
    stock_returns <- window_data[, stock]
    factors <- window_data[, c("MKT", "SMB", "HML")]
    
    # 去除含有缺失值的行
    valid_rows <- complete.cases(stock_returns, factors)
    stock_returns <- stock_returns[valid_rows]
    factors <- factors[valid_rows]
    
    if (length(stock_returns) < 30) {
      next
    }
    
    # 构建线性回归模型
    X <- cbind(1, as.matrix(factors))
    y <- as.vector(stock_returns)
    beta <- solve(t(X) %*% X) %*% t(X) %*% y
    
    y_pred <- X %*% beta
    residuals <- y - y_pred
    
    # 对残差进行缩放（如果有需要，此处缩放逻辑和Python中保持一致）
    residuals <- residuals * 100
    
    # 拟合GARCH(1,1)模型
    spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), 
                       mean.model = list(armaOrder = c(0, 0), include.mean = FALSE))
    fit <- ugarchfit(spec = spec, data = residuals)
    
    # 进行预测
    forecast <- ugarchforecast(fit, n.ahead = 20)
    pred_vol_value <- sqrt(forecast@forecast$sigmaFor[20])
    predictions[i] <- pred_vol_value
  }
  return(predictions)
}

# 主函数入口
if (Sys.getenv("RSTUDIO") == "1") {
  # 读取数据（修改为你的实际文件路径）
  returns <- read.csv("C:/Users/86187/Desktop/hs300_stock_returns.csv")
  fama <- read.csv("C:/Users/86187/Desktop/Fama因子_filtered.csv")
  
  # 将日期列转换为Date格式
  returns$日期 <- as.Date(returns$日期)
  fama$日期 <- as.Date(fama$日期)
  
  # 按日期合并数据
  merged_data <- inner_join(returns, fama, by = "日期") %>% 
    arrange(日期) %>% 
    as.data.frame()
  
  # 获取股票列名（除日期列外）
  stock_columns <- colnames(returns)[colnames(returns)!= "日期"]
  
  # 创建一个数据框来存储预测的特质波动率
  pred_vol <- data.frame(日期 = merged_data$日期)
  for (stock in stock_columns) {
    pred_vol[, stock] <- NA
  }
  
  # 使用并行处理
  cl <- makeCluster(detectCores())
  clusterExport(cl, list("calculate_garch_vol", "merged_data"), envir = environment())
  
  pb <- progress_bar$new(total = length(stock_columns))
  results <- parLapply(cl, stock_columns, function(stock) {
    pb$tick()
    calculate_garch_vol(stock, merged_data)
  })
  
  stopCluster(cl)
  
  # 将结果填入数据框
  for (i in seq_along(stock_columns)) {
    pred_vol[, stock_columns[i]] <- results[[i]]
  }
  
  # 保存结果
  write.csv(pred_vol, "C:/Users/86187/Desktop/GARCHVOL.csv", row.names = FALSE)
}