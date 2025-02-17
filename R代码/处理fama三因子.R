# 安装并加载所需的包，如果未安装需先安装
# install.packages("readr")
# install.packages("dplyr")
install.packages("data.table")
library(readr)
library(dplyr)
library(data.table)

# 1. 处理Fama三因子数据
# 读取原始CSV文件
df <- read_csv("raw_data/STK_MKT_THRFACDAY.csv")

# 提取MarkettypeID为'P9707'的行
fama_factors <- df[df$MarkettypeID == "P9707", ]

# 重命名列以符合要求
colnames(fama_factors) <- c("日期", "MKT", "SMB", "HML")

# 只保留需要的列
fama_factors <- fama_factors[, c("日期", "MKT", "SMB", "HML")]

# 保存为新的CSV文件
write_csv(fama_factors, "Fama因子.csv", col_names = TRUE)

# 2. 处理数据对齐
# 读取hs300成分股收益率数据
returns <- read_csv("raw_data/hs300_stock_returns.csv")

# 删除第一行空值（删除全为空的行）
returns <- returns[!apply(returns, 1, function(x) all(is.na(x))), ]

# 读取Fama因子数据
fama <- read_csv("Fama因子.csv")

# 确保日期格式一致，将字符型日期转换为Date类型
returns$日期 <- as.Date(returns$日期)
fama$日期 <- as.Date(fama$日期)

# 根据returns的日期筛选fama数据
fama_filtered <- fama[fama$日期 %in% returns$日期, ]

# 保存筛选后的Fama因子数据
write_csv(fama_filtered, "factors/Fama因子_filtered.csv", col_names = TRUE)

cat("原始Fama因子数据行数: ", nrow(fama), "\n")
cat("hs300成分股收益率数据行数: ", nrow(returns), "\n")
cat("筛选后Fama因子数据行数: ", nrow(fama_filtered), "\n")

# 再次保存为新的CSV文件
write_csv(fama_factors, "Fama因子.csv", col_names = TRUE)