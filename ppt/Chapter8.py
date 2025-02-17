# -*- coding: utf-8 -*-
# """
# Created on Sun Oct 29 12:38:09 2020

# @author: zw
# """
#%% 8.1.2  主要的股票指数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

# index_data = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第8章\国内A股主要股指的日收盘数据（2014-2018）.xlsx',sheet_name='Sheet1',header=0,index_col=0) #导入外部数据
# index_data.plot(subplots=True,layout=(2,2),figsize=(10,10),fontsize=13,grid=True)


#%% 8.2.1  投资组合的主要变量
x=np.random.random(5) #从均匀分布中随机抽取5个从0到1的随机数
weights=x/np.sum(x)    #生成一个权重的数组
print(weights)
round(sum(weights),2)  #验证生成的权重随机数是否合计等于1
data=pd.DataFrame()
import akshare as ak
for i in ['600843','603218','002946','600763','600584']:
    data[i]= ak.stock_zh_a_hist(symbol=i, period="daily", start_date="20221201", end_date='20241201')['收盘']

#【例8-1】假定投资组合配置了5只A股股票（具体信息见表8-1），数据是2016年至2018年期间的每个2018年期间的每个交易日收盘价格。下面就通过 Python计算投资组合的预期收益率和年波动率，具体分为4个步骤。
#data = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第8章\构建投资组合的五只股票数据（2016-2018）.xlsx',sheet_name='Sheet1',header=0,index_col=0)  #导入外部数据
(data/data.iloc[0]).plot(figsize=(8,6))    #将股价按首个交易日进行归一处理并可视化

R=np.log(data/data.shift(1))    #按照对数收益率的计算公式得到股票收益率
R=R.dropna()                    #删除缺省的数据
R.describe()
R.hist(bins=40,figsize=(10,10))   #将股票收益率按照直方图方式展示

#第3步：计算每只股票的平均收益率、波动率以及协方差，由于运用的是日数据，因此需要进行年化处理，代码如下：
R_mean=R.mean()*252          #计算股票的年化平均收益率
print(R_mean)

R_cov=R.cov()*252          #计算股票的协方差矩阵并且年化处理
print(R_cov)

R_corr=R.corr()          #计算股票的相关系数矩阵
print(R_corr)

R_vol=R.std()*np.sqrt(252)          #计算股票收益率的年化波动率
print(R_vol)

# 第4步：运用前面生成的随机权重数计算投资组合的预期收益率和收益波动率，代码如下：
R_port=np.sum(weights*R_mean)      #计算投资组合的预期收益率
print('投资组合的预期收益率：',round(R_port,4))

vol_port=np.sqrt(np.dot(weights,np.dot(R_cov,weights.T)))     #计算投资组合收益波动率
print('投资组合收益波动率',round(vol_port,4))


#%% 8.2.2  投资组合的有效前沿
#【例8-2】沿用前面例8-1的信息，针对投资组合配置的5只股票，运用 Python随机生成1000组权重的数组，进而绘制投资组合的可行集（见图84），具体的代码如下：
Rp_list=[]     #建立一个初始的投资组合收益率数列
Vp_list=[]     #建立一个初始的投资组合收益波动率数列
for i in np.arange(1000):       #生成1000个不同权重的预期收益率与收益波动率
    x=np.random.random(5)
    weights=x/sum(x)
    Rp_list.append(np.sum(weights*R_mean))
    Vp_list.append(np.sqrt(np.dot(weights,np.dot(R_cov,weights.T))))

plt.figure(figsize=(8,6))
plt.scatter(Vp_list,Rp_list)
plt.xlabel(u'波动率',fontsize =13)
plt.ylabel(u'收益率',fontsize =13,rotation=90)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(0.1,0.28)
plt.ylim(-0.1,0.2)
plt.title(u'投资组合收益率与波动率的关系',fontsize=13)
plt.grid('True')
plt.show()

#【例8-3】沿用前面例8-1的信息，同时给定投资组合的预期收益率等于10%，运用Python计算使得投资组合收益波动率最小情况下的每只股票的配置权重，具体的代码如下：
import scipy.optimize as sco      #导入Scipy的子模块optimize
def f(w):                       #定义一个需要求解最优化的函数
    w=np.array(w)               #设置投资组合中每只股票的权重
    Rp_opt=np.sum(w*R_mean)     #计算最优投资组合的预期收益率
    Vp_opt=np.sqrt(np.dot(w,np.dot(R_cov,w.T)))        #计算最优投资组合的收益波动率
    return np.array([Rp_opt,Vp_opt])                   #以数组的格式输出结果

def Vmin_f(w):                 #定义一个得到最小波动率的权重函数
    return f(w)[1]             #输出前面定义的函数f(w)结果的第二个元素

cons=({'type':'eq','fun':lambda x:np.sum(x)-1},{'type':'eq','fun':lambda x:f(x)[0]-0.1})
#以字典格式一次输入权重的约束条件，预测收益率等于10%
bnds=tuple((0,1) for x in range(len(R_mean)))     #以元组格式输入权重的边界条件
len(R_mean)*[1.0/len(R_mean),]         #用于生成一个权重相等的数组

result=sco.minimize(Vmin_f,len(R_mean)*[1.0/len(R_mean),],method='SLSQP',bounds=bnds,constraints=cons)
print('投资组合预期收益率10%时上海机场的权重',round(result['x'][0],4))
print('投资组合预期收益率10%时宝钢股份的权重',round(result['x'][1],4))
print('投资组合预期收益率10%时海通证券的权重',round(result['x'][2],4))
print('投资组合预期收益率10%时工商银行的权重',round(result['x'][3],4))
print('投资组合预期收益率10%时中国石油的权重',round(result['x'][4],4))


#【例8-4】依然沿用例8-1的信息，计算该投资组合收益波动率的全局最小值，以及与该最小波动率相对应的预期收益率，具体的代码如下：
cons_vmin=({'type':'eq','fun':lambda x:np.sum(x)-1})    #仅设置权重和等于1的约束条件

result_vmin=sco.minimize(Vmin_f,len(R_mean)*[1.0/len(R_mean),],method='SLSQP',bounds=bnds,constraints=cons_vmin)
Rp_vmin=np.sum(R_mean*result_vmin['x'])
Vp_vmin=result_vmin['fun']
print('波动率在可行集是全局最小值时的投资组合预期收益率',round(Rp_vmin,4))
print('在可行集是全局最小的波动率',round(Vp_vmin,4))


#【例8-5】依然沿用例8-1的信息，最终完成对有效前沿的创建并可视化（见图8-5），具体的代码如下： 
Rp_target=np.linspace(Rp_vmin,0.25,100)    #生成投资组合的目标收益率数组
Vp_target=[]
for r in Rp_target:
    cons_new=({'type':'eq','fun':lambda x:np.sum(x)-1},{'type':'eq','fun':lambda x:f(x)[0]-r})  
    #以字典格式输入预期收益率等于目标收益率的约束条件和权重的约束条件
    result_new=sco.minimize(Vmin_f,len(R_mean)*[1.0/len(R_mean),],method='SLSQP',bounds=bnds,constraints=cons_new)
    Vp_target.append(result_new['fun'])

plt.figure(figsize=(8,6))
plt.scatter(Vp_list,Rp_list)
plt.plot(Vp_target,Rp_target,'r-',label=u'有效前沿',lw=2.5)
plt.plot(Vp_vmin,Rp_vmin,'y*',label=u'全局最小波动率',markersize=14)
plt.xlabel(u'波动率',fontsize =13)
plt.ylabel(u'收益率',fontsize =13,rotation=90)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(0.15,0.28)
plt.ylim(-0.1,0.25)
plt.title(u'投资组合的有效前沿',fontsize=13)
plt.legend(fontsize=13)
plt.grid('True')
plt.show()

#%%8.2.3  资本市场线
#【例8-6】依然沿用例8-1的信息，同时假定无风险利率是2%年，通过 Python模拟资本市场线并且可视化，具体分为两个步骤。
def F(w):                       #定义一个新的需要求解最优化的函数
    Rf=0.02                     #无风险利率为2%
    w=np.array(w)               #设置投资组合中每只股票的权重
    Rp_opt=np.sum(w*R_mean)     #计算最优投资组合的预期收益率
    Vp_opt=np.sqrt(np.dot(w,np.dot(R_cov,w.T)))        #计算最优投资组合的收益波动率
    SR=(Rp_opt-Rf)/Vp_opt                              #定义投资组合的夏普比率
    return np.array([Rp_opt,Vp_opt,SR])                   #以数组的格式输出结果

def SRmin_F(w):              #定义一个使负的夏普比率最小化的函数，也就是夏普比率最大
    return -F(w)[2]

cons_SR=({'type':'eq','fun':lambda x:np.sum(x)-1})   #设置权重的约束条件
result_SR=sco.minimize(SRmin_F,len(R_mean)*[1.0/len(R_mean),],method='SLSQP',bounds=bnds,constraints=cons_SR)

Rf=0.02                         
slope=-result_SR['fun']     #资本市场线的斜率
Rm=np.sum(R_mean*result_SR['x']) #市场组合的预期收益率
Vm=(Rm-Rf)/slope                 #市场组合收益波动率
print('市场组合的预期收益率',round(Rm,4))
print('市场组合的波动率',round(Vm,4))

#第2步：模拟资本市场线并且可视化（见图8-6），具体的代码如下
Rp_cml=np.linspace(0.02,0.25)      #刻画资本市场线的投资组合预期收益率数组
Vp_cml=(Rp_cml-Rf)/slope            #刻画资本市场线的投资组合收益波动率数组

plt.figure(figsize=(8,6))
plt.scatter(Vp_list,Rp_list)
plt.plot(Vp_target,Rp_target,'r-',label=u'有效前沿',lw=2.5)
plt.plot(Vp_cml,Rp_cml,'b--',label=u'资本市场线',lw=2.5)
plt.plot(Vm,Rm,'g*',label=u'市场组合',markersize=14)
plt.plot(Vp_vmin,Rp_vmin,'y*',label=u'全局最小波动率',markersize=14)
plt.xlabel(u'波动率',fontsize =13)
plt.ylabel(u'收益率',fontsize =13,rotation=90)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(0.0,0.3)
plt.ylim(-0.1,0.25)
plt.title(u'投资组合理论的可视化',fontsize=13)
plt.legend(fontsize=13)
plt.grid('True')
plt.show()


#%%8.3.1  系统风险与非系统风险
stocks_data=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第8章\上证180指数成分股日收盘价（2016-2018年（剔除该期间上市的股票））.xlsx',sheet_name='Sheet1',header=0,index_col=0)  #导入数据

return_stocks=np.log(stocks_data/stocks_data.shift(1))  #建立股票日收益率时间序列
return_stocks=return_stocks.dropna()    #缺失数据的处理
n=len(return_stocks.columns)            #计算全部股票的数量
vol_port=np.zeros(n)                    #生成存放投资组合收益波动率的初始数组

for i in range(1,n+1):
    weight=np.ones(i)/i             #依次生成投资组合中每只股票的等权重数组
    return_cov=252*return_stocks.iloc[:,:i].cov()  #依次计算投资组合中股票收益率的年化协方差
    vol_port[i-1]=np.sqrt(np.dot(weight,np.dot(return_cov,weight.T))) #依次计算并存放投资组合的年化波动率

N_list=np.arange(n)+1  #生成从1到155的数组

plt.figure(figsize=(8,6))
plt.plot(N_list,vol_port,'r-',lw=2.0)
plt.xlabel(u'投资组合的股票数量',fontsize =13)
plt.ylabel(u'投资组合波动率',fontsize =13,rotation=90)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'投资组合中的股票数量与投资组合的波动率之间的关系',fontsize=13)
plt.legend(fontsize=13)
plt.grid('True')
plt.show()


#%% 8.3.2  模型数学表达式及运用
def Ri_CAPM(beta,Rm,Rf):
    '''构建资本资产定价木星来计算投资资产的预期收益
    beta：代表了单一股票或者某个股票组合的贝塔值：
    Rm：代表了市场收益率：
    Rf：代表了无风险利率。'''
    return Rf+beta*(Rm-Rf)

# 【例8-8】假定有4只股票，分别是A股票、B股票、C股票和D股票，股票的贝塔值分别是 =0.5， =0.8， =1.2和 =1.5，市场组合的预期收益率是10%，无风险利率是3%，运用资本资产定价模型计算这4只股票的预期收益率。
R_f =0.03 #无风险利率
R_m =0.1  #市场组合的预期收益率
beta_array =np.array([0.5,0.8,1.2,1.5])    #输入贝塔值的数组

Ri=Ri_CAPM(beta=beta_array,Rm=R_m,Rf=0.03)
print('贝塔等于0.5时投资资产的预期收益率：',round(Ri[0],4))
print('贝塔等于0.8时投资资产的预期收益率：',round(Ri[1],4))
print('贝塔等于1.2时投资资产的预期收益率：',round(Ri[2],4))
print('贝塔等于1.5时投资资产的预期收益率：',round(Ri[3],4))

#【例8-9】利用8.2节例8-1中涉及的上海机场和宝钢股份这两只股票，以沪深300指数作为市场组合，基于2016年至2018年的日收盘价数据计算这2只股票的贝塔值。下面，直接运用 Python进行相应的计算，具体分为3个步骤。
data_index=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第8章\沪深300指数（2016-2018年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)   #导入外部数据

R_index=np.log(data_index/data_index.shift(1))  #按照对数收益率的计算公式得到沪深300值数的日收益率时间序列
R_index=R_index.dropna()             #缺失数据的处理
R_index.describe()

#第2步：计算上海机场股票的贝塔值，并且需要运用到6.2节介绍 Stats Models的子模块api，具体的代码如下：
import statsmodels.api as sm           #导入StatsModels的子模块api
R_index_addcons=sm.add_constant(R_index)   #对自变量的样本值添加一列常数值
model_shjc=sm.OLS(endog=R.iloc[:,0],exog=R_index_addcons)  #构建计算上海机场股票的普通最小二乘法的线性回归模型
result_shjc=model_shjc.fit()           #拟合线性回归模型
result_shjc.summary()

result_shjc.params  #直接输出线性回归魔性的常数项和贝塔值

#第3步：类似于第2步的做法，计算宝钢股份股票的贝塔值，具体的代码如下：
model_bggf=sm.OLS(endog=R.iloc[:,1],exog=R_index_addcons)  #构建计算宝钢股份股票的普通最小二乘法的线性回归模型
result_bggf=model_bggf.fit()
result_bggf.summary()

result_bggf.params

#%%8.4.3 几何布朗运动
#第1步：导入数据并且计算得到股票的平均年化收益率和年化波动率（见图8-8），具体的代码如下：
S=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第8章\东方航空股票价格（2014-2018）.xlsx',sheet_name='Sheet1',header=0,index_col=0)    #导入外部数据
S.describe()
S.tail()

R=np.log(S/S.shift(1))    #得到股票2014至2018年日收益率
R=R.dropna()
R.plot(figsize=(8,6))

mu=R.mean()*252           #股票的预期年化收益率
sigma=R.std()*np.sqrt(252)#股票收益的年化波动率
print('股票的预期年化收益率',round(mu,4))
print('股票收益的年化波动率',round(sigma,4))

#第2步：输入需要进行模拟的相关参数，在这一步中将运用到Pandas模块中的时间戳索引函数 DatetimeIndex，用该函数生成从2019年1月2日至2021年12月31日并且是工作日的时间数据组，具体代码如下：
import numpy.random as npr  #导入NumPy的子模块random
#date=pd.DatetimeIndex(start='2019-01-02',end='2021-12-31',freq='B')#生成2019至2021年的工作日数组
date=pd.date_range(start='2019-01-02',end='2021-12-31',freq='B')#生成2019至2021年的工作日数组
N=len(date)                  #将N赋值为是date的长度
I=500                        #模拟的路径数
dt=1.0/252                   #单位时间的区间长度
mu=np.array(mu)              #将数据结构调整为数组
sigma=np.array(sigma)        #将数据结构调整为数组
S_GBM=np.zeros((N,I))        #建立存放服从几何布朗运动的未来股价的初始数组
S_GBM[0]=4.73                #将模拟的起点设为2019年1月2日的收盘价

#第3步：运用for语句并生成模拟的未来股价时间序列，具体的代码如下：
for t in range(1,N): 
    epsilon=npr.standard_normal(I)
    S_GBM[t]=S_GBM[t-1]*np.exp((mu-0.5*sigma**2)*dt+sigma*epsilon*np.sqrt(dt))
S_gbm=pd.DataFrame(S_GBM,index=date)   #将模拟的数值转化为带有时间索引的数据框
S_gbm.describe()

#最后：将模拟的结果可视化（见图8-9），具体代码如下：
plt.figure(figsize=(8,6))
plt.plot(S_gbm)
plt.xlabel(u'日期',fontsize =13)
plt.ylabel(u'股价',fontsize =13,rotation=90)
plt.xticks(fontsize=13,rotation=30)
plt.yticks(fontsize=13)
plt.title(u'服从几何布朗运动的股价模拟全部路径（2019-2021年）',fontsize=13)
plt.grid('True')
plt.show()

#图8-9是将全部500条模拟路径进行了可视化，不难发现未来3年的股价绝大多数是处于0~20元的区间之中，下面就将模拟的前10条路径进行可视化（见图8-10）
plt.figure(figsize=(8,6))
plt.plot(S_GBM.iloc[:,0:10])    #将模拟的前10条路径可视化
plt.xlabel(u'日期',fontsize =13)
plt.ylabel(u'股价',fontsize =13,rotation=90)
plt.xticks(fontsize=13,rotation=30)
plt.yticks(fontsize=13)
plt.title(u'服从几何布朗运动的股价模拟前10条路径（2019-2021年）',fontsize=13)
plt.grid('True')
plt.show()

#%% 8.5  投资组合的绩效评估
def SR(Rp,Rf,Vp):
    '''计算夏普比率（Sharpe Ratio）
    Rp：表示投资组合的年化收益率
    Rf：表示年化无风险利率
    Vp：表示投资组合的年化收益波动率'''
    return (Rp-Rf)/Vp

#【例8-11】假定以表8-3中的4只公募基金作为评估对象，选择的观测期间是从2016年至2018年，观测频率是每个交易日；同时在计算夏普比率过程中，无风险利率假定选择银行一年期存款基准利率1.5%，具体的计算分3步进行。
#第1步：导入外部的数据并且绘制每个基金净值的日走势图（见图8-11），具体的代码如下：
fund=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第8章\四只开放式股票型基金的净值（2016-2018年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)    #导入外部数据
fund.plot(figsize=(8,6))

#第2步：计算2016年至2018年这3年平均的基金年化收益率和波动率，同时，运用前面自定义计算夏普比率的函数SR对这4只基金进行计算，具体的代码如下：
R_fund=np.log(fund/fund.shift(1))     #生成基金日收益率的时间序列
R_fund=R_fund.dropna()                   
R_mean=R_fund.mean()*252              #计算全部3年的平均年化收益率
Sigma=R_fund.std()*np.sqrt(252)       #计算全部3年的年化波动率
R_f=0.015                             #一年期银行存款基准利率
SR_3years=SR(Rp=R_mean,Rf=R_f,Vp=Sigma)
print('2016至2018年平均3年的夏普比率\n',SR_3years)  #输入\n表示输出时换行

#第3步：计算2016至2018年期间每一年的夏普比率，具体的代码如下
R_fund2016=R_fund.loc['2016-01-01':'2016-12-31']    #获取2016年的日收益率
R_fund2017=R_fund.loc['2017-01-01':'2017-12-31']    #获取2017年的日收益率
R_fund2018=R_fund.loc['2018-01-01':'2018-12-31']    #获取2018年的日收益率
R_mean_2016=R_fund2016.mean()*252                   #计算2016年的年化收益率
Sigma_2016=R_fund2016.std()*np.sqrt(252)            #计算2016年的年化波动率
R_mean_2017=R_fund2017.mean()*252                   #计算2017年的年化收益率
Sigma_2017=R_fund2017.std()*np.sqrt(252)            #计算2017年的年化波动率
R_mean_2018=R_fund2018.mean()*252                   #计算2018年的年化收益率
Sigma_2018=R_fund2018.std()*np.sqrt(252)            #计算2018年的年化波动率
SR_2016=SR(Rp=R_mean_2016,Rf=R_f,Vp=Sigma_2016)     #计算2016年的夏普比率
SR_2017=SR(Rp=R_mean_2017,Rf=R_f,Vp=Sigma_2017)     #计算2017年的夏普比率
SR_2018=SR(Rp=R_mean_2018,Rf=R_f,Vp=Sigma_2018)     #计算2018年的夏普比率
print('2016年的夏普比率\n',SR_2016)
print('2017年的夏普比率\n',SR_2017)
print('2018年的夏普比率\n',SR_2018)

#%% 8.5.2索提诺比率
def SOR(Rp,Rf,Vpl):
    '''计算索提诺比率（Sortino ratio）
    Rp：表示投资组合的年化收益率
    Rf：表示年化无风险利率
    Vpl：并表示投资组合收益率的年化下行标准差'''
    return (Rp-Rf)/Vpl

#【例8-12】依然沿用前面例81的信息，运用 Python计算4只基金的索提诺比率，具体分为两个步骤。
#第1步：计算每只基金收益率的下行标准差，具体代码如下：
Vp_lower=np.zeros_like(R_mean)  #生成防止基金收益率下行标准差的初始数组
for i in range(len(Vp_lower)):
    R_neg=R_fund.iloc[:,i][R_fund.iloc[:,i]<0]    #生成基金收益率为负的时间序列
    Vp_lower[i]=np.sqrt(252)*np.sqrt(np.sum(R_neg**2)/len(R_neg))  #计算收益的年化下行标准差
    print(R_fund.columns[i],'收益率下行标准差',round(Vp_lower[i],4))

SOR_3years=SOR(Rp=R_mean_2016,Rf=R_f,Vpl=Vp_lower)
print('2016至2018年平均3年的索提诺比率\n',SOR_3years)

#%% 8.5.3  特雷诺比率
def TR(Rp,Rf,beta):
    '''计算特雷诺比率（Treynor ratio）
    Rp：表示投资组合的年化收益率
    Rf：表示年化无风险利率
    beta：表示投资组合的贝塔值。'''
    return (Rp-Rf)/beta

#【例8-13】依然沿用前面例8-11的信息，计算4只基金的特雷诺比率，同时在计算投资组合的贝塔值是运用沪深300指数作为市场组合，具体的过程分为3个步骤。
#第1步：导人外部沪深300指数的数据并且计算每个基金的贝塔值、具体的代码如下：
HS300=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第8章\沪深300指数（2016-2018年）.xlsx',sheet_name='Sheet1',header=0,index_col=0)    #导入外部数据
R_HS300=np.log(HS300/HS300.shift(1))    #计算并生成沪深300指数的日收益率序列
R_HS300=R_HS300.dropna()

import statsmodels.api as sm              #导入StatsModels的子模块api
betas=np.zeros_like(R_mean)      #生成一个放置基金贝塔值的初始数组
cons=np.zeros_like(R_mean)       #生成一个防止线性回归方程的常数项初始数组
X=R_HS300                        #设置自变量的样本值
X_addcons=sm.add_constant(X)     #对自变量的样本值增加一列常数项
for i in range(len(R_mean)):     
    Y=R_fund.iloc[:,i]           #设定因变量的样本值
    model=sm.OLS(endog=Y,exog=X_addcons)   #构建普通最小二乘法线性回归模型
    result=model.fit()           #生成一个线性回归的结果对象
    cons[i]=result.params[0]     #生成线性回归方程的常数项数组
    betas[i]=result.params[1]    #生成基金贝塔值的数组
    print(R_fund.columns[i],'贝塔值',round(betas[i],4))  


#第2步：将线性回归的结果进行可视化（见图8-12），具体的代码如下：
X_list=np.linspace(np.min(R_HS300),np.max(R_HS300),200)   #生成一个X轴的数组
plt.figure(figsize=(11,10))
for i in range(len(R_mean)):
    plt.subplot(2,2,i+1)
    plt.scatter(X,R_fund.iloc[:,i])
    plt.plot(X_list,cons[i]+betas[i]*X_list,'r-',label=u'线性回归拟合',lw=2.0)
    plt.xlabel(u'指数',fontsize =13)
    plt.ylabel(u'基金',fontsize =13,rotation=0)
    plt.xticks(fontsize=13,rotation=30)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.grid('True')
plt.show()


#第3步：计算得出每只基金2016年至2018年这3年平均的特雷诺比率，具体的代码如下：
TR_3years=TR(Rp=R_mean,Rf=R_f,beta=betas)
print('2016至2018年平均3年的索提诺比率\n',round(TR_3years,4))


#%% 8.5.4  信息比率
def IR(Rp,Rb,TD):
    '''计算信息比率（Information ratio）
    Rp：表示投资组合的年化收益率；
    Rb：表示基准组合的年化收益率；
    TD：表示跟踪误差。'''
    return (Rp-Rb)/TD

#【例8-14】依然沿用前面例8-11的开放式股票型基金的信息演示运用 Python计算信息比率，同时在计算投资跟踪偏离度和跟踪误差过程中，运用沪深300指数作为基准组合，具体的过程分为两个步骤。
#第1步：计算每个基金的跟踪误差，具体的代码如下
TE_fund=np.zeros_like(R_mean)    #生成一个放置基金跟踪误差的初始数组
for i in range(len(R_mean)):
    TD=np.array(R_fund.iloc[:,i])-np.array(R_HS300.iloc[:,0])   #生成一个基金跟踪偏高率的数组
    TE_fund[i]=TD.std()*np.sqrt(252)    #生成基金的变化跟踪误差数组
    print(R_fund.columns[i],'跟踪误差',round(TE_fund[i],4))

#第2步：利用前面已经定义计算信息比率的函数IR测算得出每个基金的信息比率，具体的代码如下
R_mean_HS300=R_HS300.mean()*252     #计算沪深300指数的年化收益率
R_mean_HS300=np.array(R_mean_HS300) #将沪深300指数年化收益率变成数组
IR_3years=IR(Rp=R_mean,Rb=R_mean_HS300,TD=TE_fund)
print('2016至2018年平均的信息比率\n',round(IR_3years,4))



#%% 试用图书馆的数据库（如Wind、CSMAR等），下载贵州茅台（600519.SZ）自上市以来（2001.8.27-2019.11.20）的周收益率数据，然后利用该时间序列数据计算平均收益率、标准差、偏度、峰度、下行标准差、95%的置信度下的VaR值、夏普比、索提诺比（假设无风险收益为0，可以借助Excel、Python、Matlab等软件）。
#step1：导入数据
import numpy as np
import pandas as pd
Gzmt = pd.read_csv('D:\Zhangzw\Career\投资组合\投资学第9版PPT_中文\作业\Gzmt600519.CSV',header=0,index_col=0,encoding="gbk") #导入外部数据
#step2：数据整理及初步显示
R=Gzmt['涨跌幅']    #按照对数收益率的计算公式得到股票收益率
R=R.dropna()                    #删除缺省的数据
R.describe()                    #基本数据统计
R.hist(bins=40,figsize=(10,10))   #将股票收益率按照直方图方式展示

#step3：计算每只股票的平均收益率、标准差、偏度、峰度、下行标准差、95%的置信度下的VaR值、夏普比、索提诺比，由于运用的是日数据，因此需要进行年化处理，代码如下：
R_mean=R.mean()         #计算股票的平均收益率
R_vol=R.std()         #计算股票收益率的波动率
R_skew=R.skew()      #计算股票收益率的偏度
R_kurt=R.kurt()     #计算股票收益率的峰度
#计算下行标准差
R_neg=R[R<0]    #生成收益率为负的时间序列
R_vol_lower=np.sqrt(np.sum(R_neg**2)/len(R_neg))  #计算收益的下行标准差
#95%的置信度下的VaR值
R_VaR_5 = np.percentile(R, 5)    #用历史法简单计算
#夏普比率
R_SR=(R_mean-0)/R_vol
#索提诺比
R_SOR=(R_mean-0)/R_vol_lower

print('股票的平均周收益率 ',R_mean)
print('股票周收益率的波动率 ',R_vol)
print('股票周收益率的偏度 ',R_skew)
print('股票周收益率的峰度 ',R_kurt)
print('股票周收益率95%的置信度下的VaR值 ',R_VaR_5)
print('股票周收益率的夏普比率 ',R_SR)
print('股票周收益率的索提诺比 ',R_SOR)







