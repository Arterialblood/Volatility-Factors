# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:55:02 2020

@author: zw
"""
#%% 4.1.1 序列
import pandas as pd
pd.__version__

#针对表4-1中2018年9月3日的数据生成序列，生成序列需要运用 Series函数，要输入涉及的索引index。首先演示手动输入生成序列，具体的代码如下：
return_series1 = pd.Series(data=[0.003731,-0.001838,-0.003087,-0.024112],index=['中国石油','工商银行','上汽集团','宝钢股份'])

#其次，借助第3章例3-5中创建的数组return_array，用数组生成针对2018年9月3日数据的序列，也可以得到和上面同样的效果。具体的代码如下：
import numpy as np
return_list = [0.003731,	0.021066,	-0.004854,	0.006098,	-0.006060, -0.001838,	0.001842,	-0.016544,	-0.003738,	0.003752, -0.003087,	-0.000344,	-0.033391,	0.007123,	0.004597, -0.024112,	0.011704,	-0.029563,	-0.014570,	0.016129]
return_array = np.array(return_list)
return_array = return_array.reshape(4,5)

return_series2 = pd.Series(data=return_array[:,0],index=['中国石油','工商银行','上汽集团','宝钢股份'])

#当然，也可以运用时间作为索引，就以工商银行在2018年9月3日至9月7日这5个交易日的涨跌幅作为例子进行演示，具体的代码如下：
return_series3 = pd.Series(data=[-0.001838,0.001842,-0.016544,-0.003738,0.003752],index=['2018-9-3','2018-9-4','2018-9-5','2018-9-6','2018-9-7'])

#%% 4.1.2 数据框
#【例4-2】沿用本章例4-1的信息，将日期作为行索引，将股票名称作为列名，运Data Frame函数创建数据框，具体的代码如下:
date=['2018-9-3','2018-9-4','2018-9-5','2018-9-6','2018-9-7']
stock=['中国石油','工商银行','上汽集团','宝钢股份']
return_dataframe=pd.DataFrame(data=return_array.T,index=date,columns=stock) #.T是转置的意思

#【例4-3】针对例4-2生成的数据框return_dataframe，依次以Excel、CSV、txt格式导出并存放在用户计算机的桌面，具体的代码如下:
return_dataframe.to_excel('D:/myTempData_Return.xlsx')
return_dataframe.to_csv('D:/myTempData_Return.csv')
return_dataframe.to_csv('D:/myTempData_Return.txt')

#%% 4.1.3 外部数据导入
#【例4-4】以从外部 Excel文件导入沪深300指数在2018年每个交易日的开盘高点位、最低点位以及收盘点位数据并且创建数据框。图4-2是拟导入的 Excel文件的部分截图。
HS300_excel1 = pd.read_excel(r'D:\Zhangzw\Python\Python金融数据分析\RawData\第4章\沪深300指数.xlsx',sheet_name="Sheet1",header=0,index_col=0)

#【例4-5】仅以Tushare作为例子来演示如何通过API接口在 Python中导入沪深300指数2018年每个交易日的开盘点位、最高点位、最低点位以及收盘点位，具体的代码如下:
import tushare
print(tushare.__version__)

import tushare as ts

#方式一
ts.set_token('78a093e7e3bffc2c93d67dba9614d0b53641476cad3c5a6797115824')
#这种方式设置token我们会吧token保存到本地，所以我们在使用的时候只需设置一次，失效之后，我们可以替换为新的token

#方式二
pro = ts.pro_api('78a093e7e3bffc2c93d67dba9614d0b53641476cad3c5a6797115824')
#这种在初始化接口的时候设置token

PFYH_excel3 = pro.daily(ts_code='600000.SH', start_date='20220301', end_date='20220331')

#%% 4.2.1中文字体的可视化
from pylab import mpl  #导入子模块mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #以黑体字显示中文
mpl.rcParams['axes.unicode_minus']=False #解决保存图像是负号‘_’为显示方块的问题

#【例4-6】沿用例4-4的沪深300指数2018年的日交易数据，对数据框进行可视化，具体的代码如下：
HS300_excel1.plot(kind='line',subplots=True,sharex=True,sharey=True,layout=(2,2),figsize=(10,8),title=u'2018年沪深300指数走势图',grid=True,fontsize=13)

#%% 4.3.1
#【例4-7】沿用例4-4中从外部Excel表导入的沪深300指数2018年的日交易数据，分别查看具体的行索引名和列名，相关的代码如下：
HS300_excel1.index  #查看行索引名
HS300_excel1.columns  #查看列名

#【例4-8】沿用例4-4中的日交易数据，查看该数据框的行数和列数，相关的代码如下：
HS300_excel1.shape  #查看数据框的行数和列数

#【例4-9】沿用例4-4的日交易数据，查看样本量、均值、方差、最大值、最小值、分位数等涉及时间序列的统计指标，相关的代码如下：
HS300_excel1.describe()  #数据框数据的描述性统计

yy = HS300_excel1.loc['2018-12-18']  #索引输出对应数据
HS300_excel1.iloc[-9]  #输出第-9行

#%% 4.3.2数据框的索引与截取
#【例4-10】沿用例4-4的日交易数据，进行一般性截取的演示，具体的代码如下：
HS300_excel1[:3] #截取数据框前3行的数据
HS300_excel1[2:8] #截取数据框第3行-第8行的数据
zz = HS300_excel1.iloc[7:13,1:3] #截取数据框第8-13行，第2-3列的数据

#【例4-11】沿用例4-4的日交易数据，找出收盘点位超过4300点的相关数据，具体的代码如下:
HighPrice = HS300_excel1[HS300_excel1['收盘点位']>=4300] #截取数据框

#【例4-12】沿用例4-4的日交易数据，选取开盘点位超过4100点并且收盘点位小于4200点的相关数据，具体的代码如下：
HighPrice2 = HS300_excel1[(HS300_excel1['开盘点位']>=4100)&( HS300_excel1['收盘点位']<=4200)] #截取数据框

#【例4-13】沿用例4-4的日交易数据，分别输出按照行索引由小到大、由大到小排序的结果，具体的代码如下：
AscendingPrice = HS300_excel1.sort_index(axis=0,ascending=True)
AscendingPrice2 = HS300_excel1.sort_index(axis=0,ascending=False)

#【例4-14】沿用例4-4的日交易数据，分别输出按照收盘点位由小到大排序、最高点位由大到小排序的结果，具体的代码如下
AscendingPrice3 = HS300_excel1.sort_values(by='收盘点位',ascending=True)

#%% 4.3.4
#【例4-15】沿用例4-4的日交易数据，将其中一个列名“收盘点位”修改为“收盘价格”，具体代码如下：
HS300_columnschange = HS300_excel1.rename(columns={'收盘点位':'close'})

#【例4-16】导入“浦发银行”“上海机场”“中国石化”这3只股票在2018年3月6日收盘价的数据并且生成新的数据框，假定在这个数据框中存在缺省值的情况，需要对这些缺省值进行处理，具体的代码如下：
stock = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第4章\关于2018年3月份股票价格数据.xlsx',sheet_name="Sheet1",header=0,index_col=0)

#表4-6是运用不同方法对例4-16涉及的缺失值进行处理以及相应的代码。
stock_dropna = stock.dropna()
stock_fillzero = stock.fillna(value=0)
stock_ffill = stock.fillna(method='ffill')
stock_ffill2 = stock.fillna(method='bfill')

#%% 4.4.1
#【例4-17】从外部导入沪深300指数在2016年至2017年每日交易价格的数据并且生成一个新的数据框，这个数据框待会将用于按行拼接，具体的代码如下：
HS300_excel2 = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第4章\沪深300指数.xlsx',sheet_name="Sheet2",header=0,index_col=0)  #注意导入的是sheet2

#【例4-18】从外部导入2018年3月东方航空、宝钢股份这两只股票的日收盘价格数据且生成一个新的数据框，该数据框将用于按列拼接，具体的代码如下：
stock2 = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第4章\关于2018年3月份股票价格数据.xlsx',sheet_name="Sheet2",header=0,index_col=0)  #注意导入的是sheet2

#%% 4.4.2 
#【例4-19】以例4-4和例4-17生成的两个沪深300指数的日交易价格数据框按行拼接，最终生成2016年至2018年期间沪深300指数每日交易价格的数据框，具体的代码如下:
HS300_new = pd.concat([HS300_excel2,HS300_excel1],axis=0)
HS300_new.head()
HS300_new.tail()

#【例4-20】运用例4-16生成的浦发银行、上海机场、中国石化这3只股票2018年3月的日收盘价形成的数据框 stock，和例4-19生成的东方航空、宝钢股份两只股票同期的日收盘价的数据框 stock2，将这两个数据框按列拼接，具体的代码如下：
stock_new = pd.concat([stock,stock2],axis=1)  #按列拼接

#%% 4.4.3
#【例4-21】沿用例4-20中涉及的两个数据框stock、stock2，运用函数 merge按列拼接，具体的代码如下：
stock_new2 = pd.merge(left=stock,right=stock2,left_index=True,right_index=True)

#【例4-22】沿用例4-20中涉及的两个数据框 stock、stock2，运用函数join按列拼接,具体的代码如下：
stock_new3 = stock.join(stock2,on='日期')

#%% 4.5.1
HS300_diff = HS300_new.diff()

HS300_new.idxmax()

HS300_new.idxmin()

HS300_new.kurt()

HS300_new.max()

HS300_new.mean()

HS300_new.median()

HS300_new.min()

HS300_perc = HS300_new.pct_change()

HS300_new.quantile(q=0.2) #计算20%分位数

HS300_new.skew()

HS300_shift1 = HS300_new.shift(1)

HS300_new.std()

HS300_perc.sum()

HS300_new.var()

HS300_cumsum = HS300_perc.cumsum()

HS300_cumchag = HS300_perc.cumprod()

HS300_new.corr()

#%% 4.5.2 
#首先解决中文显示的问题
from pylab import mpl  #导入子模块mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #以黑体字显示中文
mpl.rcParams['axes.unicode_minus']=False #解决保存图像是负号‘_’为显示方块的问题


#【例4-23】以例4-19生成的数据框作为分析对象，生成收盘点位的20日均值，将该平均值与每日的收盘点位进行可视化（见图4-4），具体的代码如下：
HS300_meanclose = HS300_new['收盘点位'].rolling(window=20).mean() #生成一个20日平均收盘点位的序列
HS300_meanclose = HS300_meanclose.to_frame()  #将序列变成数据框
HS300_meanclose = HS300_meanclose.rename(columns={'收盘点位':'20日平均收盘点位'})
HS300_close = HS300_new['收盘点位'].to_frame()#生成每日收盘点位的序列
HS300_new1 = pd.concat([HS300_close,HS300_meanclose],axis=1) #生成包括每日收盘点位，20日平均收盘点位的新数据框
HS300_new1.plot(figsize=(10,7),title='2016-2018年沪深300指数走势',grid=True,fontsize=12)

#【例4-24】以例4-19生成的数据框作为分析对象，生成60天时间窗口的沪深300指数收盘点位的移动波动率（移动标准差），并且进行可视化（见图4-5），具体的代码如下：
HS300_rollingstd  = HS300_new['收盘点位'].rolling(window=40).std()
HS300_rollingstd  = HS300_rollingstd.to_frame()
HS300_rollingstd  = HS300_rollingstd.rename(columns={'收盘点位':'60移动收盘点位的波动率'})
HS300_rollingstd.plot(figsize=(10,7),title=u'2016-2018年沪深300指数波动率的走势',grid=True,fontsize=12)

#【例4-25】以例4-19生成的数据框作为分析对象，生成30天时间窗口的沪深300指数开盘点位、最高点位、最低点位以及收盘点位之间的移动相关系数，具体的代码如下：
HS300_rollingcorr  = HS300_new.rolling(window=30).corr()
HS300_rollingcorr  = HS300_rollingcorr.dropna()
HS300_rollingcorr.head()
HS300_rollingcorr.tail()


