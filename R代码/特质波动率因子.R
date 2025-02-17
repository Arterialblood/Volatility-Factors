# install.packages("readr")
# install.packages("dplyr")
# install.packages("tidyr")
library(readr)
library(dplyr)
library(tidyr)

# 读取数据
returns <- read_csv("raw_data/hs300_stock_returns.csv")
fama <- read_csv("factors/Fama因子_filtered.csv")

# 将日期列转换为Date格式
returns$日期 <- as.Date(returns$日期)
fama$日期 <- as.Date(fama$日期)

# 按日期合并数据
merged_data <- inner_join(returns, fama, by = "日期") %>% 
  arrange(日期) %>% 
  mutate(index = 1:nrow(.)) %>%  # 创建索引列用于后续操作
  as.data.frame()

# 创建一个与returns结构相似的data.frame来存储特质波动率
stock_columns <- colnames(returns)[colnames(returns)!= "日期"]
idio_vol <- data.frame(日期 = merged_data$日期)
for (col in stock_columns) {
  idio_vol[[col]] <- rep(NA, nrow(merged_data))
}

# 对每只股票计算特质波动率
for (stock in stock_columns) {
  residuals <- c()
  for (i in 60:nrow(merged_data)) {
    # 获取过去60天的数据
    window_data <- merged_data[(i - 60 + 1):i, ]
    stock_returns <- window_data[, stock]
    factors <- window_data[, c("MKT", "SMB", "HML")]
    
    # 去除任何包含NA的行
    valid_rows <- complete.cases(stock_returns, factors)
    stock_returns <- stock_returns[valid_rows]
    factors <- factors[valid_rows]
    
    if (length(stock_returns) < 30) {  # 如果有效数据少于30天，跳过
      residuals <- c(residuals, NA)
      next
    }
    
    # 构建回归模型的自变量矩阵（添加截距项）
    X <- cbind(1, as.matrix(factors))
    y <- as.matrix(stock_returns)
    
    # 进行回归拟合
    fit <- lm.fit(X, y)
    beta <- fit$coefficients
    
    # 计算残差
    y_pred <- X %*% beta
    residual <- y - y_pred
    
    # 计算残差的标准差作为特质波动率
    idio_volatility <- sd(residual)
    residuals <- c(residuals, idio_volatility)
  }
  # 添加前60天的NA值
  residuals <- c(rep(NA, 60), residuals)
  idio_vol[, stock] <- residuals
}

# 保存结果
write.csv(idio_vol, "factors/RVOL.csv", row.names = FALSE)

# 读取RVOL.csv
rvol <- read.csv("factors/RVOL.csv")

# 删除前60行数据
rvol <- rvol[-(1:60), ]

# 重置行索引
rownames(rvol) <- 1:nrow(rvol)

# 打印数据形状
cat("数据形状:", dim(rvol), "\n")

# 检查是否存在缺失值
cat("\n各列缺失值数量:\n")
print(colSums(is.na(rvol)))

# 打印总的缺失值数量
cat("\n总缺失值数量:", sum(is.na(rvol)))