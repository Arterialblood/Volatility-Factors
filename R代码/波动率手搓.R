# install.packages("readr")
# install.packages("dplyr")
# install.packages("tidyr")
# install.packages("zoo")
library(readr)
library(dplyr)
library(tidyr)
library(zoo)

# 读取数据
index_returns <- read_csv("raw_data/hs300_index_returns.csv")
index_returns$日期 <- as.Date(index_returns$日期)
index_returns <- as.data.frame(index_returns) %>% 
  tibble::column_to_rownames("日期")

stock_returns <- read_csv("raw_data/hs300_stock_returns.csv")
stock_returns$日期 <- as.Date(stock_returns$日期)
stock_returns <- as.data.frame(stock_returns) %>% 
  tibble::column_to_rownames("日期")

# 删除空值
index_returns <- na.omit(index_returns)
stock_returns <- stock_returns[rowSums(is.na(stock_returns))!= ncol(stock_returns), ]
stock_returns <- stock_returns[, colSums(is.na(stock_returns))!= nrow(stock_returns)]

# 计算VOL_3M因子（60天滚动窗口标准差）
calculate_vol_3m <- function(returns) {
  rollapply(returns, width = 60, FUN = sd, by.column = TRUE, fill = NA, align = "right")
}
vol_3m <- apply(stock_returns, 2, calculate_vol_3m) %>% 
  as.data.frame() %>% 
  tibble::rownames_to_column("日期") %>% 
  as.data.frame() %>% 
  mutate(日期 = as.Date(日期)) %>% 
  tibble::column_to_rownames("日期")

# 创建factors文件夹（如果不存在）
if (!dir.exists("factors")) {
  dir.create("factors")
}

# 打印结果信息
cat("VOL_3M因子计算完成\n")
cat("因子数据形状:", dim(vol_3m), "\n")
cat("\n前5行数据预览:\n")
print(head(vol_3m))

# 保存结果
write.csv(vol_3m, "factors/vol_3m.csv", row.names = TRUE)
cat("\n因子数据已保存至 'factors/vol_3m.csv'\n")

# VOL_3M因子数据预处理
vol_3m <- read.csv("factors/vol_3m.csv")
vol_3m$日期 <- as.Date(vol_3m$日期)
vol_3m <- as.data.frame(vol_3m) %>% 
  tibble::column_to_rownames("日期")

cat("处理前:\n")
cat("数据形状:", dim(vol_3m), "\n")
cat("空值数量:", sum(is.na(vol_3m)), "\n")

vol_3m <- vol_3m[rowSums(is.na(vol_3m))!= ncol(vol_3m), ]
vol_3m <- vol_3m[, colSums(is.na(vol_3m))!= nrow(vol_3m)]

cat("\n处理后:\n")
cat("数据形状:", dim(vol_3m), "\n")
cat("空值数量:", sum(is.na(vol_3m)), "\n")

write.csv(vol_3m, "factors/vol_3m_processed.csv", row.names = TRUE)
cat("\n处理后的数据已保存至 factors/vol_3m_processed.csv\n")

# 计算RANKVOL因子
# 对每日收益率计算横截面分位数
daily_ranks <- apply(stock_returns, 1, rank, ties.method = "random") / ncol(stock_returns) %>% 
  t() %>% 
  as.data.frame()

# 计算60天滚动窗口的分位数标准差
rankvol <- apply(daily_ranks, 2, function(x) rollapply(x, width = 60, FUN = sd, by.column = TRUE, fill = NA, align = "right")) %>% 
  as.data.frame() %>% 
  tibble::rownames_to_column("日期") %>% 
  as.data.frame() %>% 
  mutate(日期 = as.Date(日期)) %>% 
  tibble::column_to_rownames("日期")

# 打印RANKVOL因子信息
cat("\nRANKVOL因子计算完成\n")
cat("因子数据形状:", dim(rankvol), "\n")
cat("\n前5行数据预览:\n")
print(head(rankvol))

# 保存RANKVOL因子
write.csv(rankvol, "factors/rankvol.csv", row.names = TRUE)
cat("\nRANKVOL因子数据已保存至 'factors/rankvol.csv'\n")

# RANKVOL因子数据预处理
rankvol <- read.csv("factors/rankvol.csv")
rankvol$日期 <- as.Date(rankvol$日期)
rankvol <- as.data.frame(rankvol) %>% 
  tibble::column_to_rownames("日期")

cat("\nRANKVOL处理前:\n")
cat("数据形状:", dim(rankvol), "\n")
cat("空值数量:", sum(is.na(rankvol)), "\n")

rankvol <- rankvol[rowSums(is.na(rankvol))!= ncol(rankvol), ]
rankvol <- rankvol[, colSums(is.na(rankvol))!= nrow(rankvol)]

cat("\nRANKVOL处理后:\n")
cat("数据形状:", dim(rankvol), "\n")
cat("空值数量:", sum(is.na(rankvol)), "\n")

write.csv(rankvol, "factors/rankvol_processed.csv", row.names = TRUE)
cat("\n处理后的数据已保存至 factors/rankvol_processed.csv\n")