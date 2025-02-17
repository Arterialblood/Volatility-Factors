# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:38:09 2020

@author: zw
"""
import matplotlib
matplotlib.__version__
import matplotlib.pyplot as plt  #导入matplotlib子模块pyplot

from pylab import mpl  #导入子模块mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #以黑体字显示中文
mpl.rcParams['axes.unicode_minus']=False #解决保存图像是负号‘_’为显示方块的问题

#%% 5.2.1
#【例5-1】假定某一个股票的贝塔值处于[0.5,2.0]的区间之中，金融市场的无风险利率是3%，市场收益率是12%，运用子模块pyplot中的plot函数绘制该股票的证券市场线（图5-1）。具体的代码如下：
import numpy as np

beta = np.linspace(0.5,2.0,100)
Rf = 0.02
Rm = 0.12
Ri =Rf+beta*(Rm-Rf)  #CAPM模型
plt.figure(figsize=(9,6))
plt.plot(beta,Ri,'r-',label='证券市场线',lw=2.0)
plt.plot(1.0,Rf+1.0*(Rm-Rf),'o',lw=2.5)  #图中画一个贝塔值等于1的收益率点
plt.axis('tight')
plt.xlabel(u'贝塔值',fontsize=14)
plt.xlim(0.4,2.1)
plt.ylabel(u'单一股票收益率',fontsize=14,rotation=90)
plt.ylim(0.07,0.22)
plt.title(u'资本资产定价模型',fontsize=14)
plt.annotate(u'贝塔等于1的收益',fontsize=14,xy=(1.0,0.12),xytext=(0.8,0.15),arrowprops=dict(facecolor='b',shrink=0.05))
plt.legend(loc=0,fontsize=14)
plt.grid()

#%% 5.2.2
#【例5-2】以在第4章中例4-19中拼接生成的存放2016年至2018年期间沪深300指数（见图5-2）每日开盘点位、最高点位、最低点位、收盘点位的数据框作为演示对象，运用subplot函数绘制相应的2×2图形（每一行有两个图、每一列也有两个图），具体的代码如下：
import pandas as pd
HS300_new = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第5章\沪深300指数（2016-2018）.xlsx',sheet_name="Sheet1",header=0,index_col=0)  #注意导入的是sheet1
plt.figure(figsize=(11,9))
#################
plt.subplot(2,2,1)
plt.plot(HS300_new['开盘点位'],'r-',label=u'沪深300开盘点位',lw=2.0)
plt.xticks(fontsize=13,rotation=30)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'点位',fontsize=13,rotation=30)
plt.legend(loc=0,fontsize=13)
plt.grid()
##################第1行第2个
plt.subplot(2,2,2)
plt.plot(HS300_new['最高点位'],'b-',label=u'沪深300最高点位',lw=2.0)
plt.xticks(fontsize=13,rotation=30)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'点位',fontsize=13,rotation=30)
plt.legend(loc=0,fontsize=13)
plt.grid()
##################第2行第1个
plt.subplot(2,2,3)
plt.plot(HS300_new['最低点位'],'b-',label=u'沪深300最低点位',lw=2.0)
plt.xticks(fontsize=13,rotation=30)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'点位',fontsize=13,rotation=30)
plt.legend(loc=0,fontsize=13)
plt.grid()
##################第2行第2个
plt.subplot(2,2,4)
plt.plot(HS300_new['收盘点位'],'b-',label=u'沪深300收盘点位',lw=2.0)
plt.xticks(fontsize=13,rotation=30)
plt.xlabel(u'日期',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'点位',fontsize=13,rotation=30)
plt.legend(loc=0,fontsize=13)
plt.grid()

#%% 5.2.3 二维数据图
#【例5-2-1】利用标准正态分布生成两列数据，并且在图中表示出来：
import numpy as np
import matplotlib.pyplot as plt
# 单轴双数据折线图
np.random.seed(2000)
y = np.random.standard_normal((20, 2)).cumsum(axis=0) #cumsum函数：沿着指定轴的元素累加和所组成的数组，其形状应与输入原数组一致
plt.figure(figsize=(7, 4))
plt.plot(y[:,0],lw=1.5,label='1st')  #第一列数据折线
plt.plot(y[:,1],lw=1.5,label='2nd')  #第二列数据折线
plt.plot(y[:,0],'ro') #用红色的点给第一列数据标出位置
plt.grid(True)  
plt.legend(loc=0) #在最佳位置显示数据示例
plt.axis('tight') #所有数据可见（缩小限值）
plt.xlabel('index')
plt.ylabel('value')
plt.title('示例：简单的二维折线图')

# 常用双轴图 包含两个数据集、 两个y轴的图表
#【例5-2-2】利用标准正态分布生成两列数据，并且用双轴图表示出来：
y[:,0] = y[:,0] * 10
fig, ax1 = plt.subplots()
plt.plot(y[:, 0], 'b', lw=1.5, label='1st')
plt.plot(y[:, 0], 'ro')
plt.grid(True)
plt.legend(loc=8)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value 1st')
plt.title('示例：简单的双轴图')
ax2 = ax1.twinx()
plt.plot(y[:, 1], 'g', lw=1.5, label='2nd')
plt.legend(loc=0)
plt.ylabel('value 2nd')

#%%5.3.1
#【例5-3】以3.5节讨论的运用 NumPy模块获取基于不同分布的随机数作为演示直方图的数据源，同时在本例中选择正态分布、对数正态分布、卡方分布以及贝塔分布（图5-3），并且从每个分布中随机抽取1000个样本，最后以2×2子图的方式呈现。具体的代码如下：
import numpy.random as npr
I=1000   #从每个分布中随机抽取1000个样本
x_norm = npr.normal(loc=0.8,scale=1.5,size=I)
x_logn = npr.lognormal(mean=0.5,sigma=1.0,size=I)
x_chi = npr.chisquare(df=4,size=I)
x_beta = npr.beta(a=2,b=4,size=I)

plt.figure(figsize=(12,10))
#################第1行第1个
plt.subplot(2,2,1)
plt.hist(x_norm,label=u'正态分布的抽样',bins=20,facecolor='y',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=90)
plt.legend(loc=0,fontsize=13)
plt.grid(True)
#################第1行第2个
plt.subplot(2,2,2)
plt.hist(x_logn,label=u'对数正态分布的抽样',bins=20,facecolor='r',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=90)
plt.legend(loc=0,fontsize=13)
plt.grid(True)
#################第2行第1个
plt.subplot(2,2,3)
plt.hist(x_chi,label=u'卡方分布的抽样',bins=20,facecolor='b',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=90)
plt.legend(loc=0,fontsize=13)
plt.grid(True)
#################第2行第2个
plt.subplot(2,2,4)
plt.hist(x_beta,label=u'贝塔分布的抽样',bins=20,facecolor='c',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=90)
plt.legend(loc=0,fontsize=13)
plt.grid(True)

#%% 5.3.2
#【例5-4】从均值为0.8、标准差为1.5的正态分布中随机抽取两组样本值，每组各1000个样本值，通过直方图并且以堆叠的形式展示（见图5-4），具体的代码如下：
x_norml = npr.normal(loc=0.8,scale=1.5,size=(I,2))
plt.figure(figsize=(8,5))
plt.hist(x_norml,label=[u'正态分布的抽样组1',u'正态分布的抽样组2'],stacked=True,bins=30,edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=0)
plt.title(u'正态分布随机抽取的两组样本值堆叠的直方图')
plt.legend(loc=0,fontsize=13)
plt.grid(True)

#【例5-5】沿用例5-4中的两组样本值数据，通过直方图并且以并排的形式展示（见图5-5），具体的代码如下：
plt.figure(figsize=(8,5))
plt.hist(x_norml,label=[u'正态分布的抽样组1',u'正态分布的抽样组2'],bins=30,edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=0)
plt.title(u'正态分布随机抽取的两组样本值并排的直方图')
plt.legend(loc=0,fontsize=13)
plt.grid(True)

#%% 5.4.1
#【例5-6】沿用4.1节的例4-1的信息，针对表5-5描述的2018年9月3日至9月7日这5个交易日中相关股票的涨跌幅情况，生成2018年9月3日-7日这4只股票的涨跌幅的条形图，并且以2×2子图的方式呈现，需要运用生成条形图的函数bar。具体代码如下：
import pandas as pd
import numpy as np
return_list = [0.003731,	0.021066,	-0.004854,	0.006098,	-0.006060, -0.001838,	0.001842,	-0.016544,	-0.003738,	0.003752, -0.003087,	-0.000344,	-0.033391,	0.007123,	0.004597, -0.024112,	0.011704,	-0.029563,	-0.014570,	0.016129]
return_array = np.array(return_list)
return_array = return_array.reshape(4,5)
date=['2018-9-3','2018-9-4','2018-9-5','2018-9-6','2018-9-7']
stock=['中国石油','工商银行','上汽集团','宝钢股份']
return_dataframe=pd.DataFrame(data=return_array.T,index=date,columns=stock)

plt.figure(figsize=(12,10))
#################第1行第1个
plt.subplot(2,2,1)
plt.bar(x=return_dataframe.columns,height=return_dataframe.iloc[0],width=0.5,label=u'2018年9月3日涨跌幅',facecolor='y')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(-0.035, 0.025)
plt.ylabel(u'涨跌幅',fontsize=13,rotation=90)
plt.legend(loc=0,fontsize=13)
plt.grid(True)
#################第1行第2个
plt.subplot(2,2,2)
plt.bar(x=return_dataframe.columns,height=return_dataframe.iloc[2],width=0.5,label=u'2018年9月5日涨跌幅',facecolor='y')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(-0.035, 0.025)
plt.ylabel(u'涨跌幅',fontsize=13,rotation=90)
plt.legend(loc=0,fontsize=13)
plt.grid(True)
#################第2行第1个
plt.subplot(2,2,3)
plt.bar(x=return_dataframe.columns,height=return_dataframe.iloc[3],width=0.5,label=u'2018年9月6日涨跌幅',facecolor='y')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(-0.035, 0.025)
plt.ylabel(u'涨跌幅',fontsize=13,rotation=90)
plt.legend(loc=0,fontsize=13)
plt.grid(True)
#################第2行第2个
plt.subplot(2,2,4)
plt.bar(x=return_dataframe.columns,height=return_dataframe.iloc[3],width=0.5,label=u'2018年9月7日涨跌幅',facecolor='y')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(-0.035, 0.025)
plt.ylabel(u'涨跌幅',fontsize=13,rotation=90)
plt.legend(loc=0,fontsize=13)
plt.grid(True)

#%% 5.4.2 
#【例5-7】沿用例5-6的相关信息，将2018年9月6日和9月7日这两个交易日的4只股票的涨跌幅放置在一张水平条形图中进行展示，需要运用生成水平条形图的函数barh，具体的代码如下：
plt.figure(figsize=(9,6))
plt.barh(y=return_dataframe.columns,width=return_dataframe.iloc[3],height=0.5,label=u'2018年9月6日涨跌幅')
plt.barh(y=return_dataframe.columns,width=return_dataframe.iloc[4],height=0.5,label=u'2018年9月7日涨跌幅')
plt.xticks(fontsize=13)
plt.xlabel(u'涨跌幅',fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'水平条形图可视化股票的涨跌幅',fontsize=13)
plt.legend(loc=0,fontsize=13)
plt.grid(True)

#%% 5.5
#【例5-8】以沪深300指数与上证180指数2016年至2018年的日涨跌幅作为分析对象，用于演示散点图的绘制方法，具体分为两个步骤。
import pandas as pd
HS300_sz180 = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第5章\沪深300指数与上证180指数的日涨跌幅_2016_2018.xlsx',header=0,index_col=0)  #注意导入的是sheet1
HS300_sz180.describe()
HS300_sz180.corr()

plt.figure(figsize=(9,6))
plt.scatter(x=HS300_sz180.iloc[:,0],y=HS300_sz180.iloc[:,1],c='b',marker='o')
plt.xticks(fontsize=14)
plt.xlabel(u'沪深300指数涨跌幅',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(u'上证180指数涨跌幅',fontsize=14,rotation=90)
plt.title(u'沪深300指数与上证180指数的涨跌幅散点图',fontsize=14)
plt.grid()

#%% 5.6
#【例5-9】国际货币基金组织（IMF）的特别提款权被称为“纸黄金”，是该组织分配给会员国的一种使用资金的权利。目前，特别提款权的价值是由美元、欧元、人民币、日元、英镑等一篮子储备货币所决定，从2016年10月1日至今，这5种货币在特别提款权中的比重由表5-6所示。
currency = ['美元','欧元','人民币','英镑','日元']
perc = [0.4173,0.3093,0.1092,0.0833,0.0809]
plt.figure(figsize=(9,7))
plt.pie(perc,labels=currency,autopct='%2.3f%%')  #autopct 控制饼图内百分比设置,可以使用format '%2.3f'指小数点前后位数:2位整数，3位小数
plt.axis('equal')
plt.legend(loc=1,fontsize=13)
plt.title(u'特别提款权中不同币种的占比',fontsize=13)

#%% 5.7 股价蜡烛图
"""
# 首先需要安装新模块
# 安装mplfinance库要求pandas和matplotlib，然后在prompt中操作：
# pip install mplfinance --upgrade --user -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import mplfinance as mpf
import pandas as pd

def import_csv():	# 导入股票数据
    df = pd.read_csv('D:\Zhangzw\Python\Python金融数据分析\RawData\第5章\A_20200801_20200831.csv')
    df.rename(
            columns={
            'date': 'Date', 'open': 'Open', 
            'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'}, 
            inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])    # 转换为日期格式
    df.set_index(['Date'], inplace=True)    # 将日期列作为行索引
    return df

df = import_csv()

kwargs = dict(type='candle', mav=(2, 5, 7), volume=True, title='Stock Price', ylabel='price', ylabel_lower='volume',figratio=(10, 8), figscale=1.3, linecolor='g')
mc = mpf.make_marketcolors(# 设置marketcolors
	up='red', # up:设置K线线柱颜色，up意为收盘价大于等于开盘价
	down='green', # down:与up相反，这样设置与国内K线颜色标准相符
	edge='i', # edge:K线线柱边缘颜色(i代表继承自up和down的颜色)，下同。详见官方文档)
	wick='i', # wick:灯芯(上下影线)颜色
	volume='in', # volume:成交量直方图的颜色
	inherit=True)# inherit:是否继承，选填

mystyle = mpf.make_mpf_style(# 设置图形风格
	gridaxis='both', # gridaxis:设置网格线位置
	gridstyle='-.', # gridstyle:设置网格线线型
	y_on_right=False, # y_on_right:设置y轴位置是否在右
	marketcolors=mc)

mpf.plot(df, **kwargs, style=mystyle) #如果我们不确定要往函数中传入多少个参数，或者我们想往函数中以列表和元组的形式传参数时，那就使要用*args；如果我们不知道要往函数中传入多少个关键词参数，或者想传入字典的值作为关键词参数时，那就要使用**kwargs。实际上，在一个字典前使用”**”可以unpack字典




























