# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:38:09 2020

@author: zw
"""
import scipy                                  #导入SciPy模块
scipy.__version__                             #查看版本信息

#%% 6.1.1
#【例6-1】假定变量是服从标准正态分布，需要计算当该变量处于区间【-1】的概率通过 SciPy模块求解，具体的计算分为两个步骤。
# 第1步：导入SciPy的子模块 integrate，并且在 Python中自定义一个标准正态分布的概率密度函数，具体的代码如下
import scipy.integrate as sci                       #导入SciPy的子模块integrate

def f(x):
  import numpy as np                                  #导入Numpy模块
  return 1/pow(2*np.pi,0.5)*np.exp(-0.5*x**2)         #输出标准正态分布的函数概率密度的表达式

#表6-2  integrate子模块中的积分函数与运用
sci.quad(func=f,a=-1.0,b=1.0)

sci.fixed_quad(func=f,a=-1.0,b=1.0)

sci.quadrature(func=f,a=-1.0,b=1.0)

sci.romberg(function=f,a=-1.0,b=1.0)


#%%   【例6-2】以2018年12月28日的远期国债到期收益率作为例子，远期国债到期收益率的信息如表6-4所示，考虑到表中缺少2年期、4年期的远期国债收益率，因此需要通过插值法得到相关的收益率。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

from scipy import interpolate                                              #导入SciPy的子模块interpolate
t=np.array([0.25,0.5,0.75,1.0,3.0,5.0])                                    #生成仅包含已有期限的数组       
t_new=np.array([0.25,0.5,0.75,1.0,2.0,3.0,4.0,5.0])                       #生成包括2年和4年的新数组 
rates=np.array([0.27344,0.27898,0.28382,0.2882,0.30414,0.31746])  #生成仅包含已有利率的数组
types=['nearest','zero','slinear','quadratic','cubic']                    #生成包含插值方法的列表
plt.figure(figsize=(8,6))
for i in types:                                                           #用for循环计算不同插值方法的结果并输出
    f=interpolate.interp1d(x=t,y=rates,kind=i)
    rates_new=f(t_new)
    print(i,rates_new)
    plt.plot(t_new,rates_new,'o')
    plt.plot(t_new,rates_new,'-',label=i)
    plt.xticks(fontsize=14)
    plt.xlabel(u'期限',fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(u'收益率',rotation=90)
    plt.legend(loc=0,fontsize=14)
    plt.grid()
plt.title(u'用插值法求2年期和4年期的远期国债到期收益率',fontsize=14)


#%% 6.1.3 求解方程组
#【例6-3】沿用3.1节例3-1的相关股票信息：除了已知在2018年9月3日至9月6日每只股票的涨跌幅以外，同时也已知整个投资组合的收益率（见表6-5的最后一列）。假定在这些交易日，投资组合中每只股票的权重保持不变，根据这些已知的信息求解这4只股票在整个投资组合中所占的权重。
from scipy import linalg                                                 #导入SciPy的子模块linalg
stock_return=np.array([[0.003731,-0.001838,-0.003087,-0.024112],[0.021066,0.001842,-0.000344,0.011704],[-0.004854,-0.016544,-0.033391,-0.029563],
                      [0.006098,-0.003738,0.007123,-0.01457]])            #创建包含4只股票涨跌幅的数组
port_return=np.array([-0.0105654,0.0070534,-0.0256367,-0.0038289])        #创建投资组合收益率的数组
weight=linalg.solve(a=stock_return,b=port_return)                         #计算每只股票的权重
stock=np.array(['中国石油','工商银行','上汽集团','宝钢股份'])
for i in range(0,4):
    print(stock[i],round(weight[i],2))


def g(w):                           #定义求解每只股票权重的方程组
    w1,w2,w3,w4 = w
    eq1=0.003731*w1-0.001838*w2-0.003087*w3-0.024112*w4+0.0105654       #第一个等于0的方程式
    eq2=0.021066*w1+0.001842*w2-0.000344*w3+0.011704*w4-0.0070534       #第二个等于0的方程式
    eq3=-0.004854*w1-0.016544*w2-0.033391*w3-0.029563*w4+0.0256367      #第三个等于0的方程式
    eq4=0.006098*w1-0.003738*w2+0.007123*w3-0.01457*w4+0.0038289        #第四个等于0的方程式
    return [eq1,eq2,eq3,eq4]
import scipy.optimize as sco                                           #导入SciPy的子模块optimize
result=sco.fsolve(g,[0.25,0.25,0.25,0.25])                              #求方程组的解   
result


#%% 6.1.4
#【例6-4】假定一家投资机构拟配置4只A股股票，分别是贵州茅台、工商银行、上汽集团、宝钢股份，表6-6列出了这4只股票的相关信息。
#该投资机构的资金为1亿元，以12月28日的收盘价投资，希望实现投资组合收益率的最大化，同时要求整个投资组合的贝塔值不超过1.4，此外，每只股票不允许卖空，需计算应该配置的每只股票权重和股数。
import scipy.optimize as sco                                           #导入SciPy的子模块optimize
P=np.array([590.01,5.29,26.67,6.50])                                   #输入股票价格
R=np.array([0.349032,0.155143,0.132796,0.055905])                     #输入股票收益率
b=np.array([1.64,1.41,1.21,1.06])                                      #输入股票贝塔值
def f(w):                                                              #定义求最优值得函数
    w=np.array(w)
    return -np.sum(R*w)
cons=({'type':'eq', 'fun': lambda w: np.sum(w)-1},{'type':"ineq",'fun':lambda w: 1.4-np.sum(w*b)})
bnds=((0,1),(0,1),(0,1),(0,1))

result=sco.minimize(f,[0.25,0.25,0.25,0.25], method="SLSQP", bounds=bnds,constraints=cons)       #计算最优的解
result

result['x'].round(3)
-f(result['x']).round(3)

shares=100000000*result["x"]/P
shares=shares.round(0)                                                      #结果去整数，因为最少是1股
print('贵州茅台的股数:',shares[0])
print('工商银行的股数:',shares[1])
print ('上汽集团的股数:',shares[2])
print('宝钢股份的股数:',shares[3])

#%% 
#【例6-5】沿用例6-4的信息，但是由于该投资机构的风险偏好降低，因此需要改变一个约束条件，即把原先的整体投资组合的贝塔值“不超过1.4”变更为“不超过1.2”，其他的条件都不变，需要求出投资组合中每只股票配置的最优权重和股数，具体分为两个步骤。
cons_new= ({'type':'eq', 'fun': lambda w: np.sum(w)-1},{'type': 'ineq','fun': lambda w: 1.2-np.sum(w*b)}) #新的约束条件   
result_new=sco.minimize(f,[0.25,0.25,0.25,0.25],method='SLSQP',bounds=bnds,constraints=cons_new)
result_new['x'].round(3)

-f(result_new['x']).round(3)

shares_new=100000000*result_new["x"]/P
print('贵州茅台的股数:',shares_new[0].round(0))
print('工商银行的股数:',shares_new[1].round(0))
print('上汽集团的股数:',shares_new[2].round(0))
print('宝钢股份的股数:',shares_new[3].round(0))


#%% 6.1.5
#【例6-6】沿用5.5节例5-8中运用的沪深300指数和上证180指数2016年至2018年的日涨跌幅数据，用于演示子模块stats中的统计函数及其用法，具体代码如下：
import scipy.stats as st
HS300_sz180 = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第5章\沪深300指数与上证180指数的日涨跌幅_2016_2018.xlsx',header=0,index_col=0)  #注意导入的是sheet1
HS300_sz180.describe()

st.describe(HS300_sz180)
st.kurtosis(HS300_sz180)
st.moment(HS300_sz180,moment=2)
st.mode(HS300_sz180)
st.skew(HS300_sz180)

#【例6-7】假定变量服从均值为0.5、标准差为1.5的正态分布，需要生成该变量随机抽取10000的随机数并且用直方图展示（见图6-2），随机抽样就需要运用表6-9中的rvs函数，具体的代码如下：
I=10000                                                                 #设计随机抽样的次数
rand_norm=st.norm.rvs(loc=0.5,scale=1.5,size=I)                         #从均值为0.5、标准差为1.5的正态分布中抽取样本
plt.figure(figsize=(9,6))
plt.hist(rand_norm,bins=30,facecolor='y',edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值')
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=90)
plt.title(u'正态分布的抽样')
plt.grid(True)


#【例6-8】假定变量服从均价为0.5、标准差为1.5的正态分布，计算该变量小于0.2的概率，需要运用表6-9中的cdf函数，具体的代码如下：
st.norm.cdf(x=0.2,loc=0.5,scale=1.5)

#【例6-9】假定变量服从均值为0.5、标准差为1.5的正态分布，计算当变量等于0对的概率密度函数值，需要运用表6-9中的pdf函数，具体的代码如下：
st.norm.pdf(x=0,loc=0.5,scale=1.5)

#【例6-10】假定变量服从均值为0.8、标准差为1.5的正态分布，计算当概率等于80%对应的变量值，需要运用表6-9中的ppf函数，具体的代码如下：
st.norm.ppf(q=0.8,loc=0.5,scale=1.5)


#【例6-11】沿用例6-6的信息，检验沪深300指数、上证180指数2016年至2018年的涨跌幅数据是否服从正态分布，并且依次运用 kstest检验、 Anderson-Darling检验Shapiro-Wilk检验等方法进行检验，相关的检验共有4个步骤。
st.kstest(rvs=HS300_sz180.iloc[:,0],cdf='norm',args=(0,0.01))            #检验沪深300指数
st.kstest(rvs=HS300_sz180.iloc[:,1],cdf='norm',args=(0,0.01))           #检验上证180指数    

st.anderson(x=HS300_sz180.iloc[:,0],dist='norm')                        #检验沪深300指数
st.anderson(x=HS300_sz180.iloc[:,1],dist='norm')                        #检验上证180指数


st.shapiro(HS300_sz180.iloc[:,0])
st.shapiro(HS300_sz180.iloc[:,1])                                      #检验沪深300指数

st.normaltest(HS300_sz180,axis=0)                                      #同步检验沪深300指数和上证180指数


#%% 6.2
import statsmodels                                                     #导入StatsModels模块
statsmodels.__version__

#【例6-12】以2016年至2018年期间工商银行A股股价涨跌幅作为因变量，沪深300指数的涨跌幅作为自变量，构建普通最小二乘法回归模型，具体分为两个步骤完成。
import statsmodels.api as sm                                          #导入StatsModels的子模块api
ICBC_HS300=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第6章\工商银行与沪深300指数.xlsx',sheet_name='Sheet1',header=0,index_col=0) #导入外部数据
ICBC_HS300=ICBC_HS300.dropna()                                         #删除缺失值 
Y=ICBC_HS300.iloc[:,0]
X=ICBC_HS300.iloc[:,1]
X_addcons=sm.add_constant(X)     
model=sm.OLS(endog=Y,exog=X_addcons)                               #构建普通最小二乘法的线性回归模型
result=model.fit()                                                    #生成一个线性回归的结果对象
result.summary()                                                      #输出线性回归的结果信息  

result.params                                                           #仅输出回归模型的截距项和贝塔值

import matplotlib.pyplot as plt
import pylab as mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.scatter(X,Y,c="b",marker="o")
plt.plot(X,result.params[0]+result.params[1]*X,'r-',lw=2.5)            #生成拟合的一条直线
plt.xticks(fontsize=14)
plt.xlabel(u'沪深300指数涨跌幅',fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(u'工商银行股票涨跌幅',rotation=90)
plt.title(u'工商银行股票与沪深300指数的线性回归',fontsize=14)
plt.grid()


#%% 6.3.4  arch模块
import scipy
import arch
arch.__version__

#【例6-13】沿用前面例6-6的相关信息，对2016年至2018年沪深300指数的涨跌幅构建波动率模型，选用的模型是ARCH（1）模型和 GARCH（1，1）模型，具体的过程分为5个步骤。
from arch import arch_model                                           #从arch模块中导入arch_model函数
model_arch=arch_model(y=HS300_sz180.iloc[:,0],mean='Constant',lags=0,vol='ARCH',p=1,o=0,q=0,dist='normal') 
#构建ARCH(1)模型
model_garch=arch_model(y=HS300_sz180.iloc[:,0],mean='Constant',lags=0,vol='GARCH',p=1,o=0,q=1,dist='normal')
#构建GARCH(1,1)模型

result_arch=model_arch.fit()                                          #对ARCH模型进行拟合
result_arch.summary()                                                 #对拟合结果进行输出

result_garch=model_garch.fit()                                                 #对GARCH模型进行拟合
result_garch.summary()                                                         #对拟合结果进行输出

result_garch.params

import numpy as np
vol=np.sqrt(result_garch.params[1]/(1-result_garch.params[2]-result_garch.params[3]))
print('利用GARCH(1,1)模型得到的长期波动率（每日）：',round(vol,4))

result_arch.plot()                       #ARCH模型结果的可视化

result_garch.plot()                     #GARCH模型结果的可视化

forecast_3days = result_garch.forecast(horizon=3)
forecast_3days.variance.dropna()
#%% 6.4  datetime模块
import datetime as dt                        #导入datatime模块

T1=dt.datetime(2018,12,28)
T1

T2=dt.datetime(2018,8,8,14,38,58,88)
T2

now=dt.datetime.now()
today=dt.datetime.today()
now
today


#%% 6.4.2
T2.year
T2.month
T2.weekday()
T2.day
T2.isocalendar()
T2.date()
T2.hour
T2.minute
T2.second
T2.microsecond
T2.ctime()
now.ctime()

#%% 6.4.3
# 表6-15时间对象的比较以及代码
T1.__eq__(T2)
T1==T2

T1.__ge__(T2)
T1>=T2

T1.__gt__(T2)
T1>T2

T1.__le__(T2)
T1<today

T2.__lt__(today)
T2<today

T2.__ne__(today)
T2!=today

# 表6-16
T_delta=T1-T2
T_delta.days

T_delta2=today-T2
T_delta2.seconds

T_delta2.microseconds
