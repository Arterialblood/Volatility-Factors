# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 07:17:54 2020

@author: zw
"""
#%% 10.1.2  股指期权合约
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
option_data=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第10章\SZ50ETF购3月2500合约每日价格数据.xlsx',sheet_name="Sheet1",header=0,index_col=0)		#导入外部数据
option_data.plot(figsize=(8,6),title=u'50ETF购3月2500合约的旧交易价格走势图',grid=True,fontsize=13)

#%% 10.2.2 看涨期权到期时的盈亏
#【例10-1】假定A投资者买入基础资产为100股W股票、执行价格为50元股的欧式看涨期权。假定W股票的当前市场价格为46元/股，期权到期日为4个月以后，购买1股W股票的期权价格(期权费)是6元，投资首最初投资为600元(100×6)，也就是一份看涨期权的期权费是600元。由于期权是欧式期权，因此A投资者只能生合约到期日才能行使期权。下面，考虑两种典型的情形：
S=np.linspace(30,70,100)		#模拟看涨期权到期时的股价
K=50							#看涨期权的执行价格
C=6							    #看涨期权的期权费
call1=100*np.maximum(S-K,0)		#看涨期权到期时不考虑期权费的收益
call2=100*np.maximum(S-K-C,-C)	#看涨期权到期时考虑期权费以后的收益

plt.figure(figsize=(12,6))
p1=plt.subplot(1,2,1)
p1.plot(S,call1,'r--',label=u'不考虑期权费的看涨期权多头收益',lw=2.5)
p1.plot(S,call2,'r-',label=u'考虑期权费的看涨期权多头收益',lw=2.5)
p1.set_xlabel(u'股票价格',fontsize=12)
p1.set_ylabel(u'盈亏',fontsize=12,rotation=0)
p1.set_title(u'看涨期权到期日多头的盈亏图',fontsize=13)
p1.legend(fontsize=12)
p1.grid('True')
p2=plt.subplot(1,2,2)
p2.plot(S,-call1,'b--',label=u'不考虑期权费的看涨期权空头收益',lw=2.5)
p2.plot(S,-call2,'b-',label=u'考虑期权费的看涨期权空头收益',lw=2.5)
p2.set_xlabel(u'股票价格',fontsize=12)
p2.set_ylabel(u'盈亏',fontsize=12,rotation=0)
p2.set_title(u'看涨期权到期日空头的盈亏图',fontsize=13)
p2.legend(fontsize=13)
p2.grid('True')

#%% 10.2.3看跌期权到期时的盈亏
S=np.linspace(50,90,100)			#设定看跌期权到期时的股价
K=70							#看跌期权的执行价格
P=7 							#看跌期权的期权费
put1=100*np.maximum(K-S,0)		#看跌期权到期时不考虑期权费的收益
put2=100*np.maximum(K-S-P,-P) 	#看跌期权到期时考虑期权费以后的收益

plt.figure(figsize=(12,6))
p3=plt.subplot(1,2,1)
p3.plot(S,put1,'r--',label=u'不考虑期权费的看跌期权多头收益',lw=2.5)
p3.plot(S,put2,'r-',label=u'考虑期权费的看跌期权跌多头收益',lw=2.5)
p3.set_xlabel(u'股票价格',fontsize=12)
p3.set_ylabel(u'盈亏',fontsize=12,rotation=0)
p3.set_title(u'看跌期权到期日多头的盈亏图',fontsize=13)
p3.legend(fontsize=12)
p3.grid('True')
p4=plt.subplot(1,2,2)
p4.plot(S,-put1,'b--',label=u'不考虑期权费的看跌期权空头收益',lw=2.5)
p4.plot(S,-put2,'b-',label=u'考虑期权费的看跌期权空头收益',lw=2.5)
p4.set_xlabel(u'股票价格',fontsize=12)
p4.set_ylabel(u'盈亏',fontsize=12,rotation=0)
p4.set_title(u'看跌期权到期日空头的盈亏图',fontsize=13)
p4.legend(fontsize=13)
p4.grid('True')

#%% 10.2.4  看跌-看涨平价关系式
def call_parity(p,S,K,r,T):
    '''通过看跌-看涨平价关系式计算欧式看涨期权的价格
    P，代表欧式看跌期权的价格;
    S，代表期权基础资产的价格;
    K，代表期权的执行价格;
    r，代表无风险收益率;
    T，代表期权合约的剩余期限。'''
    import numpy as np
    return p+S-K*np.exp (-r*T)

def put_parity(c, S, K, r, T):
    '''通过看跌-看涨平价关系式计算欧式看跌期权的价格
    c,代表欧式看涨期权的价格;
    S,代表期权基础资产的价格;
    K,代表期权的执行价格;
    r,代表无风险收益率;
    T,代表期权合约的剩余期限。'''
    import numpy as np
    return c+K*np.exp (-r*T)-S

call=call_parity(p=0.3, S=20,K=18, r=0.05, T=0.25) #计算看涨期权价格
put=put_parity (c=2.3, S=20, K=18, r=0.05, T=0.25) #计算看跌期权价格
print('运用平价关系式得到的看涨期权价格: ', round (call, 3))
print('运用平价关系式得到的看跌期权价格: ', round (put, 3))

#%% 10.3
def call_BS (S, K,sigma, r,T):
    '''运用布莱克一斯科尔斯一默顿定价模型计算欧式看涨期权价格
    S:代表期权基础资产的价格;
    K:代表期权的执行价格;
    sigma:代表基础资产价格百分比变化的年化波动率
    r:代表无风险收益率；
    T:代表期权合约的剩余期限。'''
    import numpy as np
    from scipy.stats import norm #从SciPy的子模块stats中导入norm函数
    d1=(np.log(S/K)+ (r+pow(sigma,2)/2)*T)/(sigma*np.sqrt (T))
    d2=d1-sigma*np.sqrt (T)
    return S*norm.cdf(d1)-K*np.exp(-r*T) *norm.cdf (d2)

def put_BS (S, K, sigma, r, T):
    '''11运用布莱克-斯科尔斯-默顿定价模型计算欧式看跌期权价格
    S:代表期权基础资产的价格;
    K:代表期权的执行价格；
    sigma:代表基础资产价格百分比变化的年化波动率
    r:代表无风险收益率;
    T:代表期权合约的剩余期限。'''
    import numpy as np
    from scipy.stats import norm #从Scipy的子模块stats中导入norm函数
    d1=(np.log(S/K)+(r+pow(sigma, 2)/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

call=call_BS(S=5.29, K=6, sigma=0.24,r=0.04,T=0.5)
put=put_BS(S=5.29,K=6,sigma=0.24, r=0.04,T=0.5)
print ('根据布莱克-斯科外斯-默顿模型计算的看涨期权价格: ', round(call, 4))
print ('根据布莱克-斯科尔斯-默顿模型计算的看跌期权价格: ', round (put,4))

#%% 10.4.1  期权价格与基础资产价格的关系
S_list=np.linspace (5.0,7.0,100)  #生成基础资产价格的数组
call_list1 = call_BS(S=S_list,K=6, sigma=0.24, r=0.04, T=0.5) #计算看涨期权的价格
put_list1 = put_BS(S=S_list,K=6, sigma=0.24, r=0.04,T=0.5) #计算看跌期权的价格
plt.figure(figsize=(8,6) )
plt.plot(S_list,call_list1,'b-',label=u'看涨期权',lw=2.5)
plt.plot (S_list,put_list1,'r-',label=u'看跌期权',lw=2.5)
plt.xlabel (u'股票价格',fontsize=13)
plt.ylabel (u'期权价格',fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'股票价格与股票期权价格的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt. show()

#%% 10.4.2 期权价格与执行价格的关系
K_list=np.linspace(5.0,7.0,100)  #生成期权执行价格的数组
call_list2 = call_BS(S=5.29, K=K_list, sigma=0.24, r=0.04,T=0.5)
put_list2 = put_BS(S=5.29, K=K_list, sigma=0.24,r=0.04,T=0.5)
plt.figure(figsize=(8,6))
plt.plot (K_list,call_list2, 'b-', label=u'看涨期权',lw=2.5)
plt.plot (K_list,put_list2, 'r-',label=u'看跌期权',lw=2.5)
plt.xlabel (u'执行价格', fontsize=13)
plt.ylabel (u'期权价格' , fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'执行价格与股票期权价格的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show ()

#%% 10.4.3  期权价格与波动率的关系
sigma_list=np.linspace (0.05,0.35, 100)  #生成波动率的数组
call_list3 = call_BS(S=5.29, K=6.0, sigma=sigma_list,r=0.04,T=0.5)
put_list3 = put_BS(S=5.29, K=6.0, sigma=sigma_list,r=0.04,T=0.5)

plt.figure(figsize=(8,6))
plt.plot (sigma_list,call_list3, 'b-', label=u'看涨期权',lw=2.5)
plt.plot (sigma_list,put_list3, 'r-',label=u'看跌期权',lw=2.5)
plt.xlabel (u'波动率', fontsize=13)
plt.ylabel (u'期权价格' , fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'波动率与股票期权价格的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show ()

#%% 10.4.4  期权价格与无风险收益率的关系
r_list=np.linspace(0.01, 0.10, 100)  #生成无风险收益率的数组
call_list4=call_BS(S=5.29, K=6.0,sigma=0.24,r=r_list,T=0.5)
put_list4=put_BS(S=5.29, K=6.0,sigma=0.24,r=r_list,T=0.5)
plt.figure(figsize=(8,6))
plt.plot(r_list,call_list4, 'b-', label=u'看涨期权',lw=2.5)
plt.plot(r_list,put_list4, 'r-',label=u'看跌期权',lw=2.5)
plt.xlabel (u'无风险利率', fontsize=13)
plt.ylabel (u'期权价格' , fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'无风险利率与股票期权价格的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show ()

#%% 10.4.5  期权价格与期权期限的关系
T_list=np.linspace(0.01, 3.0, 100)  #生成无风险收益率的数组
call_list5=call_BS(S=5.29, K=6.0,sigma=0.24,r=0.04,T=T_list)
put_list5=put_BS(S=5.29, K=6.0,sigma=0.24,r=0.04,T=T_list)
plt.figure(figsize=(8,6))
plt.plot(T_list,call_list5, 'b-', label=u'看涨期权',lw=2.5)
plt.plot(T_list,put_list5, 'r-',label=u'看跌期权',lw=2.5)
plt.xlabel (u'期权期限', fontsize=13)
plt.ylabel (u'期权价格' , fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'期权期限与股票期权价格的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show ()

#%% 10.5.1  期权的Delta
def delta_option(S, K, sigma, r,T, optype, positype):
    '''计算欧式期权的Delta值
    S:代表期权基础资产的价格;
    K:代表期权的执行价格;
    sigma:代表基础资产价格百分比变化的波动率;
    r:代表无风险收益率;
    T：代表期权合约的剩余期限;
    optype:代表期权类型,输入'call'表示看涨期权,输入'put'表示看跌期权;
    positype:代表期权头寸方向,输入'long'表示多头,输入'short'表示空头。'''
    import numpy as np
    from scipy.stats import norm #从sciPy的子模块stats中导入norm函数
    dl = (np.log(S/K)+(r+pow(sigma,2)/2)*T)/( sigma*np.sqrt(T) )
    if optype=='call':
        if positype=='long':
            delta=norm.cdf (dl)
        else:
            delta=-norm.cdf(dl)
    else:
        if positype=='long':
            delta=norm.cdf(dl)-1           
        else:
            delta=1-norm.cdf(dl)
    return delta

#【例10-10】沿用前面例10-4的工商银行股票期权，假定股票的当前价格是5元/股，其他的参数均不变，运用前面Python定义的计算期权Delta值的函数delta_option，分别计算看涨、看跌期权的多头与空头的Delta值。具体的代码如下：
delta1 = delta_option(S=5, K=6, sigma=0.24,r=0.04, T=0.5, optype='call', positype='long')
delta2 = delta_option(S=5,K=6,sigma=0.24, r=0.04,T=0.5, optype='call',positype='short')
delta3=delta_option(S=5, K=6, sigma=0.24, r=0.04,T=0.5, optype='put',positype='long')
delta4=delta_option(S=5, K=6,sigma=0.24, r=0.04,T=0.5, optype='put',positype='short')
print ('看涨期权多头的Delta值: ',round (delta1, 4) )
print ('看涨期权空头的Delta值: ',round(delta2,4) )
print ('看跌期权多头的Delta值: ',round (delta3, 4) )
print ('看跌期权空头的Delta值: ',round (delta4, 4) )

# 【例10-11】沿用前面例10-4的工商银行股票期权信息，对基础资产股票价格设定—个取值是在区间[4.0，8.0]的等差数列，其他的数保持不变，运用Python将基础资产股票价格与期权多头Deta值之间的对应关系可视(见图10-9)，具体的代码如下：
S_list=np.linspace (4.0,8.0, 100) #生成股票价格的数组
Delta_call=delta_option(S=S_list, K=6,sigma=0.24,r=0.04,T=0.5,optype='call',positype='long')#计算看涨期权的Delta值
Delta_put=delta_option(S=S_list, K=6,sigma=0.24,r=0.04,T=0.5,optype='put',positype='long')  #计算看跌期权的Delta值
plt.figure (figsize=(8,6))
plt.plot (S_list, Delta_call, 'b-',label=u'看涨期权多头',lw=2.5)
plt.plot (S_list, Delta_put, 'r-',label=u'看跌期权多头',lw=2.5)
plt.xlabel (u'股票价格',fontsize=13)
plt.ylabel (u'Delta', fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.ylim(-1, 1)
plt.title (u'股票价格与期权多头Delta的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show ()

#  【例10-12】沿用前面例10-4的工商银行股票期权信息，对期权的期限设定一个取值是在区间[0.1，5.0]的等差数列，同时将期权分为实值期权、平价期权和虚值期权3类，运用Python将期权的期限与看涨期权多头Delta值之间的对应关系可视化(见图10-10)，具体的代码如下:
T_list=np.linspace (0.1, 5.0,100) #生成期权期限的数组
Delta_call1=delta_option(S=7,K=6,sigma=0.24,r=0.04,T=T_list, optype='call',positype='long')#实值看涨期权的Delta值
Delta_call2=delta_option(S=6,K=6,sigma=0.24,r=0.04,T=T_list, optype='call',positype='long')#平价看涨期权的Delta值
Delta_call3=delta_option (S=5,K=6,sigma=0.24,r=0.04,T=T_list, optype='call',positype="1ong")#虚值看涨期权的Delta
plt.figure (figsize=(8, 6) )
plt.plot(T_list, Delta_call1, 'b-',label=u'实值看涨期权多头',lw=2.5)
plt.plot(T_list, Delta_call2, 'r-',label=u'平价看涨期权多头',lw=2.5)
plt.plot(T_list, Delta_call3, 'g-',label=u'虚值看涨期权多头',lw=2.5)
plt.xlabel (u'期权期限',fontsize=13)
plt.ylabel (u'Delta', fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'期权期限与看涨期权多头Delta的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show()
                              
#%%  10.5.2  期权的Gamma
def gamma_option (S, K, sigma,r,T):
    '''计算欧式期权的gamma 值
    S:代表期权基础资产的价格；
    K：代表期权的执行价格；
    sigma:代表基础资产价格百分比变化的波动率；
    r：代表无风险收益率；
    T:代表期权合约的剩余期限。'''
    import numpy as np
    from scipy.stats import norm #从Scipy的子模块stats中导入norm函数
    d1=(np.log(S/K)+(r+pow(sigma,2)/2)*T)/(sigma*np.sqrt(T))
    return np.exp(-pow(d1,2)/2)/(S*sigma*np.sqrt(2*np.pi*T))

#【例10-13】沿用前面例10-4的工商银行股票期权信息，同时假定股票的当前价格是5元/股，其他的参数均不变，运用前面Python定义的计算期权Gamma值的函数
gamma=gamma_option (S=5, K=6, sigma=0.24, r=0.04, T=0.5)
print ('计算得到的欧式期权Gamma值: ', round (gamma, 4))

#【例10-14】沿用前面例10-4的工商银行股票期权信息，对基础资产价格设定一个取值是在区间[4.0，8.0]的等差数列，其他的参数保持不变，并运用Python将期权的基础资产价格(股票价格)与期权Gamma值之间的对应关系可视化(见图10-11)，具体的代码如下：
S_list=np.linspace(4.0,8.0,100) #生成股票价格的数组
gamma_list1=gamma_option (S=S_list, K=6,sigma=0.24,r=0.04, T=0.5)
plt.figure(figsize=(8,6))
plt.plot(S_list,gamma_list1, 'b-',lw=2.5)
plt.xlabel (u'股票价格',fontsize=13)
plt.ylabel (u'Gamma', fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'股票价格与期权Gamma的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show()

#【例10-15】沿用前面例10-4的工商银于股票期权信息，对期权的期限设定一个取值是在区间[0.1，5.0]的等差数列，同时将期权分为实值期权、平价期权和虚值期权这3类，运用Python将看涨期权与期权Gamma值之间的对应关系可视化（见图10-2），具体的代码如下：
T_list=np.linspace (0.1,5.0,100)#生成期救期限的教组
gamma1=gamma_option(S=7,K=6, sigma=0.24,r=0.04, T=T_list) #实值看涨期权Gamma
gamma2=gamma_option(S=6,K=6, sigma=0.24, r=0.04,T=T_list) #平价看涨期权Gamma
gamma3=gamma_option(S=5,K=6, sigma=0.24, r=0.04,T=T_list) #虚值看涨期权Garma
plt.figure (figsize=(8,6))
plt.plot (T_list, gamma1, 'b-',label=u'实值看涨期权',lw=2.5)
plt.plot (T_list, gamma2, 'r-',label=u'平价看涨期权',lw=2.5)
plt.plot (T_list, gamma3, 'g-',label=u'虚值看涨期权',lw=2.5)
plt.xlabel (u'期权期限',fontsize=13)
plt.ylabel(u'Gamma',fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.ylim (0.0, 0.9)
plt.title(u'期权期限与期权Gamma的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show()

#%% 10.5.3  期权的Theta
def theta_option (S, K, sigma, r,T,optype):
    '''计算欧式期权的Theta值
    S:代表期权基础资产的价格;
    K:代表期权的执行价格;
    sigma:代表基础资产价格百分比变化的波动率;
    r:代表无风险收益率;
    T:代表期权合约的剩余期限;
    optype:代表期权的类型,输入'call 表示看涨期权,输入'put'表示看跌期权。'''
    import numpy as np
    from scipy.stats import norm #从SciPy的子模块stats中导入norm函数
    d1=(np.log(S/K)+ (r+pow(sigma,2)/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    theta_call=-(S*sigma*np.exp(-pow(d1,2)/2))/(2*np.sqrt(2*np.pi*T))-r*K*np.exp(-r*T)*norm.cdf(d2)
    if optype=='call':
        theta=theta_call
    else:
        theta=theta_call+r*K*np.exp(-r*T)
    return theta

#【例10-16】沿用前面例10-4的工商银行股票期权信息，假定股票的当前价格是5元/股，其他的参数均不变，运用前面Python定义的计算期权Theta的函数theta_option，分别计算看涨、看跌期权的Theta值，具体的代码如下：
theta_call=theta_option (S=5, K=6, sigma=0.24, r=0.04, T=0.5, optype='call')
theta_put=theta_option (S=5, K=6, sigma=0.24, r=0.04,T=0.5, optype='put')
print ('计算得到的欧式看涨期权Theta值', round(theta_call, 6) )
print ('计算得到的欧式看涨期权每日历天Theta值', round(theta_call/365, 6))
print('计算得到的欧式看涨期权每交易日Theta值', round(theta_call/252, 6))
print('计算得到的欧式看跌期权Theta值', round(theta_put, 6) )
print('计算得到的欧式看跌期权每日历天Theta值', round (theta_put/365, 6))
print('计算得到的欧式看跌期权每交易日Theta值', round (theta_put/252,6) )

#【例10-17】沿用前面例10-4的工商银行股票期权信息，对基础资产价格设定一个取值是在区间[1.0，11.0]的等差数列，其他的参数保持不变，并运用Python将期权的基础资产价格(股票价格)与期权Theta值之间的对应关系可视化(见图10-13)，具体的代码如下:
S_list=np.linspace (1.0,11.0,100) #生成股票价格的数组
theta_list1=theta_option (S_list, K=6, sigma=0.24, r=0.04,T=0.5, optype='call')
theta_list2=theta_option (S_list, K=6, sigma=0.24, r=0.04, T=0.5,optype='put')
plt.figure (figsize=(8, 6))
plt.plot (S_list, theta_list1, 'b-', label=u'看涨期权', lw=2.5)
plt.plot (S_list, theta_list2, 'r-', label=u'看跌期权', lw=2.5)
plt.xlabel (u'股票价格',fontsize=13)
plt.ylabel(u'Theta',fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title(u'股票价格与期权Theta的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show()

#【例10-18】沿用前面例10-4的工商银行股票期权信息，对期权的期限设定一个取值是在区间[0.1，5.0]的等差数列，同时将期权分为实值期权、平价期权和虚值期权这3类，运用Python将看涨期权期限与期权Theta值之间的对应关系可视化(见图10-14)，具体的操作如下：
T_list=np.linspace(0.1,5.0,100)#生成期权期限的数zu
theta1=theta_option (S=7,K=6, sigma=0.24,r=0.04,T=T_list, optype='call') #实值看涨期权的Theta
theta2=theta_option (S=6,K=6, sigma=0.24, r=0.04,T=T_list,optype='call') #平价看涨期权的Theta
theta3=theta_option (S=5, K=6, sigma=0.24,r=0.04,T=T_list,optype='call') #虚值看涨期权的Theta
plt.figure (figsize= (8,6))
plt.plot (T_list, theta1,'b-',label=u'实值看涨期权',lw=2.5)
plt.plot (T_list, theta2, 'r-',label=u'平价看涨期权',lw=2.5)
plt.plot(T_list, theta3, 'g-',label=u'虚值看涨期权',lw=2.5)
plt.xlabel (u'期权期限',fontsize=13)
plt.ylabel (u'Theta', fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title(u'期权期限与期权Theta的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show()

#%% 10.5.4  期权的Vega
def vega_option(S, K,sigma, r,T):
    '''计算欧式期权的Vega值
    s:代表期权基础资产的价格
    K:代表期权的执行价格,
    sigma:代表基础资产价格百分比变化的波动率;
    r:代表无风险收益率
    T:代表期权合约的剩余期限。'''
    import numpy as np
    dl=(np.log (S/K)+(r+pow(sigma,2)/2)*T)/(sigma*np.sqrt(T))
    return S*np.sqrt (T)*np.exp(-pow(dl,2)/2)/np.sqrt (2*np.pi)

#【例10-19】沿用前面例10-4的工商银于股票期权信息，假定股票的当前价格是58元/股，其他的参数均不变，运用前面Python定义的计算期权Vega值的函数vega_option，求解该股票期权的Vega值以及当波动率增1%时期权价格的变动情况，具体的代码如下：
vega=vega_option (S=5.8, K=6, sigma=0.24, r=0.04, T=0.5)
print ('计算得到期权的Vega值:',round (vega, 4))
print ('波动率增加1%时期权价格的变动:', round (vega*0.01,4))

# 【例10-20】沿用前面例10-4的工商银行股票期权信息，对基础资产价格设定一个取值是在区间[3.0，10.0]的等差数列，并用Python将期权的基础资产价格(股票价格)与期权Vega值之间的对应关系可视化(见图10-15)，具体的代码如下：
S_list=np.linspace(3.0,10.0, 100)  #生成股票价格的数组
vega_list=vega_option (S_list,K=6,sigma=0.24, r=0.04,T=0.5)
plt.figure (figsize=(8,6))
plt.plot (S_list, vega_list, 'b-',lw=2.5)
plt.xlabel (u'股票价格',fontsize=13)
plt.ylabel (u'Vega', fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt. yticks (fontsize=13)
plt.title(u'股票价格与期权Vega的关系', fontsize=13)
plt.grid ('True')
plt.show ()

# 【例10-21】沿用前面例10-4的工商银行股票期权信息，对期权的期限设定一个取值是在区间[0.1，5.0]的等差数列，同时将期权分为实值期权、平价期权和虚值期权这3类，运用Python将看涨期权期限与期权Vega值之间的对应关系可视化(见图10-16)，具体的代码如下：
T_list=np.linspace (0.1,5.0, 100) #生成期权期限的数组
vega1=vega_option (S=8, K=6, sigma=0.24, r=0.04,T=T_list) #实值看涨期权的Vega
vega2=vega_option (S=6, K=6, sigma=0.24, r=0.04,T=T_list) #平价看涨期权的Vega
vega3=vega_option (S=4, K=6, sigma=0.24, r=0.04,T=T_list) #虚值看涨期权的Vega
plt.figure(figsize=(8,6))
plt.plot (T_list, vega1,'b-',label=u'实值看涨期权',lw=2.5)
plt.plot (T_list,vega2,'r-', label=u'平价看涨期权',lw=2.5)
plt.plot (T_list,vega3, 'g-',label=u'虚值看涨期权',lw=2.5)
plt.xlabel (u'期权期限',fontsize=13)
plt.ylabel ('Vega', fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'期权期限与期权Vega的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show()


#%% 10.5.5  期权的Rho
def rho_option (S,K, sigma, r,T , optype) :
    '''计算欧式期权的Rho值
    S:代表期权基础资产的价格；
    K:代表期权的执行价格;
    sigma:代表基础资产价格百分比变化的波动率;
    r:代表无风险收益率；
    T:代表期权合约的剩余期限;
    optype:代表期权的类型,输入'call'表示看涨期权,输入'put'表示看跌期权。'''
    import numpy as np
    from scipy.stats import norm #从SciPy的子模块stats 中导入norm
    d1=(np.log (S/K)+(r+pow(sigma,2)/2) *T)/(sigma*np.sqrt (T) )
    d2=d1-sigma*np.sqrt (T)
    if optype=='call':
        rho=K*T*np.exp (-r*T) *norm.cdf (d2)
    else:
        rho=-K*T*np.exp(-r*T)*norm.cdf (-d2)
    return rho

#【例10-22】沿用前面例10-4的工商银行股票期权信息，假定工商银行股票的当前价格是5元/股，其他的参数均不变，运用前面Python定义的计算期权Rho的函数rho_option，分别求出看涨、看跌期权的Rho值，具体的代码如下：
rho_call=rho_option (S=5, K=6, sigma=0.24, r=0.04, T=0.5, optype='call')
rho_put=rho_option (S=5,K=6, sigma=0.24,r=0.04, T=0.5, optype='put')
print ('计算得到的看涨期权的Rho值: ', round (rho_call,4))
print ('计算得到的看跌期权的Rho值: ',round (rho_put, 4))
print('当无风险利率增加1%时着涨期权价值的变化: ',round(rho_call*0.01,4))
print('当无风险利率增加1%时看跌期权价值的变化: ',round(rho_put*0.01,4))

#【例10-23】沿用前面例10-4的工商银行股票期权信息，对基础资产价格设定一个取值是在区间[3.0，10.0]的等差数列，并用Python将期权的基础资产价格(股票价格)与期权Rho值之间的对应关系可视化(见图10-17)，具体的代码如下：
S_list=np.linspace (3.0,10.0,100) #生成股票价格的数组
rho_clist=rho_option (S=S_list, K=6, sigma=0.24, r=0.04, T=0.5,optype='call')
rho_plist=rho_option (S=S_list, K=6, sigma=0.24, r=0.04, T=0.5, optype= 'put')
plt.figure (figsize= (8, 6))
plt.plot (S_list, rho_clist, 'b-',label=u'看涨期权',lw=2.5)
plt.plot (S_list,rho_plist,'r-',label=u'看跌期权',lw=2.5)
plt.xlabel (u'股票价格', fontsize=13)
plt.ylabel ('Rho', fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title(u'股票价格与期权Rho的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show()
#股票价格与期权Rho的关系

#【例10-24】沿用前面例10-4的工商银行股票期权信息，对期权的期限设定一个取值是在区间[0.1，5.0]的等差数列，同时将期权分为实值朝权、平价期权和虚值期权这3类，运用Python将看涨期权期限与期权Rho值之司的对应关系可视化(见图10-18)，具体的代码如下：
T_list=np.linspace (0.1,5.0,100) #生成期权期限的数组
rho1=rho_option (S=8,K=6, sigma=0.24,r=0.04,T=T_list, optype='call') #实值看涨期权的Bo
rho2=rho_option (S=6, K=6,sigma=0.24,r=0.04,T=T_list, optype='call') #平价看涨期权的Rho
rho3=rho_option (S=4, K=6, sigma=0.24,r=0.04,T=T_list, optype='call') #虚值看涨期权的Rho
plt.figure(figsize=(8,6))
plt.plot (T_list, rho1, 'b-', label=u'实值看涨期权',lw=2.5)
plt.plot (T_list, rho2, 'r-',label=u'平价看涨期权',lw=2.5)
plt.plot (T_list,rho3, 'g-',label=u'虚值看涨期权',lw=2.5)
plt.xlabel (u'期权期限',fontsize=13)
plt.ylabel ('rho', fontsize=13, rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'期权期限与期权rho的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt.show()


#%% 10.6.1
def impvol_call_Newton( C, S, K, r,T):
    '''运用布莱克-斯科尔斯-默顿定价模型计算看涨期权的隐含波动率,并且使用的方法是牛顿迭代法
    C:代表看涨期权的市场价格
    S:代表期权基础资产的价格
    K:代表期权的执行价格,
    r:代表无风险收益率
    T:代表期权合约的剩余期限。'''
    def call_BS(S,K, sigma, r, T):
        import numpy as np
        from scipy.stats import norm #从SciPy子模块stats中导入norm
        d1=(np.log (S/K) + (r+pow(sigma, 2)/2)*T)/(sigma*np.sqrt(T))
        d2=d1-sigma*np.sqrt(T)
        return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf (d2)
    sigma0=0.2
    diff=C-call_BS(S, K, sigma0, r, T)
    i=0.0001  #设置一个标量
    while abs(diff)>0.0001:  #用while语句
        diff=C-call_BS(S,K, sigma0, r, T)
        if diff>0:
            sigma0 +=i
        else:
            sigma0 -=i
    return sigma0
    
def impvol_put_Newton (P,S, K,r, T):
    '''运用布莱克-斯科尔斯一默顿定价模型计算看跌期权的隐含波动率,并且使用的方法是牛顿送代法
    P:看跌期权的市场价格;
    s:代表期权基础资产的价格;
    K:代表期权的执行价格；
    r:代表无风险收益率;
    T:代表期权合约的剩余期限。'''
    def put_BS(S,K , sigma, r,T):
        import numpy as np
        from scipy.stats import norm
        d1=(np.log (S/K)+(r+pow(sigma, 2)/2) *T)/(sigma*np.sqrt (T))
        d2=d1-sigma*np.sqrt(T)
        return K*np.exp(-r*T) *norm.cdf (-d2)-S*norm.cdf (-d1)
    sigma0=0.2
    diff=P-put_BS (S, K,sigma0,r,T)
    i=0.0001
    while abs(diff)>0.0001:
        diff=P-put_BS(S, K, sigma0, r, T)
        if diff>0:
            sigma0 +=i
        else:
            sigma0 -=i
    return sigma0

#【例10-25】依然沿用前面例10-4的工商银行股票期权作为分析对象，同时假定看涨期权的市场价格为0.1566元、看跌期权的市场价格是0.7503元，其他的参数均不变，通过前面Python定义的运用牛顿迭代去计算隐含波动率的函数impvol_call_Newton和impvol_put_Newton，求解期权的隐含波动率，具体的代码如下：
imp_vol1=impvol_call_Newton (C=0.1566, S=5.29, K=6,r=0.04, T=0.5)
imp_vol2=impvol_put_Newton (P=0.7503, S=5.29, K=6, r=0.04, T=0.5)
print ("用牛顿送代法计算得到看涨期权隐含波动率: ", round (imp_vol1,4))
print('用牛顿选代法计算得到看跌期权隐含波动率: ', round (imp_vol2, 4))

#%% 10.6.2
def impvol_call_Binary (C,S, K, r,T):
    '''运用布莱克-斯科尔斯-默顿定价模型计算看涨期权的隐含波动率,并且使用的迭代方法是二分查找法
    C:代表看涨期权的市场价格
    S:代表期权基础资产的价格;
    K:代表期权的执行价格;
    r:代表无风险收益率;
    T:代表期权合约的剩余期限。'''
    def call_BS(S, K, sigma, r,T):
        import numpy as np
        from scipy.stats import norm #从sciFy的子模块stats中导入norm函数
        d1=(np.log (S/K)+(r+pow (sigma, 2)/2)*T)/(sigma*np.sqrt (T) )
        d2=d1-sigma*np.sqrt (T)
        return S*norm.cdf(d1)-K*np.exp (-r*T) *norm.cdf (d2)
    sigma_min=0.001    #设定波动率的初始最小值
    sigma_max=1.000    #设定波动率的初始最大值
    sigma_mid=(sigma_min+sigma_max)/2
    call_min=call_BS(S,K, sigma_min,r, T)
    call_max=call_BS(S, K, sigma_max,r, T)
    call_mid=call_BS(S, K, sigma_mid,r, T)
    diff=C-call_mid
    if C<call_min or C>call_max:
        print ('Error')
    while abs (diff) >1e-6:
        diff=C-call_BS(S, K,sigma_mid,r, T)
        sigma_mid=(sigma_min+sigma_max)/2
        call_mid=call_BS(S, K, sigma_mid, r, T)
        if C>call_mid:
            sigma_min=sigma_mid
        else:
            sigma_max=sigma_mid
    return sigma_mid

def impvol_put_Binary(P, S, K,r,T):
    '''运用布莱克一斯科尔斯一默顿定价模型计算看跌期权的隐含波动率,并且使用的送代方法是二分查找法
    P:代表看跌期权的市场价格;
    S:代表期权基础资产的价格;
    K:代表期权的执行价格;
    r:代表无风险收益率;
    T:代表期权合约的剩余期限。'''
    def put_BS(S, K, sigma, r,T):
        import numpy as np
        from scipy.stats import norm #从sciFy的子模块stats中导入norm函数
        d1=(np.log (S/K)+(r+pow (sigma, 2)/2)*T)/(sigma*np.sqrt (T) )
        d2=d1-sigma*np.sqrt (T)
        return K*np.exp (-r*T) *norm.cdf(-d2) - S*norm.cdf(-d1)
    sigma_min=0.001    #设定波动率的初始最小值
    sigma_max=1.000    #设定波动率的初始最大值
    sigma_mid=(sigma_min+sigma_max)/2
    put_min=put_BS(S,K, sigma_min,r, T)
    put_max=put_BS(S, K, sigma_max,r, T)
    put_mid=put_BS(S, K, sigma_mid,r, T)
    diff=P-put_mid
    if P<put_min or P>put_max:
        print ('Error')
    while abs (diff) >1e-6:
        diff=P-put_BS(S, K,sigma_mid,r, T)
        sigma_mid=(sigma_min+sigma_max)/2
        put_mid=put_BS(S, K, sigma_mid, r, T)
        if P>put_mid:
            sigma_min=sigma_mid
        else:
            sigma_max=sigma_mid
    return sigma_mid

#【例10-26】沿用前面例10-25的信息，依然是假定看涨期权的市场价格为0.1566元、看跌期权的市场价格是0.7503元，通过前面Python定义的运用二分查找法计算隐含波动率的函数impvol_call_Binary和impvol_put_Binary，分别求出看涨、看跌期权的隐含波动率， 具体的代码如下：
imp_vol3=impvol_call_Binary (C=0.1566, S=5.29, K=6, r=0.04, T=0.5)
imp_vol4=impvol_put_Binary (P=0.7503,S=5.29, K=6, r=0.04, T=0.5)
print ('用二分查找法计算得到看涨期权隐含波动率: ', round (imp_vol3,4))
print ('用二分查找法计算得到看跌期权隐含波动率: ', round (imp_vol4,4))

#%% 10.7.1  波动率微笑
#第1步：在Python中输入相关的变量，具体的代码如下: 
import datetime as dt   #导入datetime模块
T1=dt.datetime (2017, 12,29) #计算隐合波动率的日期
T2=dt.datetime (2018, 6,27) #期权到期
T_delta= (T2-T1).days/365 #期权的剩余期限
S0=2.859    #50ETF基金净值
Call_list=np.array ([0.2841,0.2486,0.2139,0.1846,0.1586,0.1369,0.1177]) #输入50ETF认购期权收盘价格
Put_list=np.array ([0.0464,0.0589, 0.0750, 0.0947,0.1183,0.1441, 0.1756]) #输入5OETE
K_list=np.array ([2.7, 2.75,2.8, 2.85, 2.9,2.95, 3.0]) #输入期权执行价格
shibor=0.048823   #6个月期的SHIBOR利率

#第2步:计算上证50ETF认购期权(看涨期权)的隐含波动率,并且需要运用for语句,具体的代码如下:
sigma_clist=np.zeros_like (Call_list) #构建存放隐含波动率的初始数组
for i in np.arange (len(Call_list)) :
    sigma_clist[i]=impvol_call_Newton(C=Call_list [i], S=S0, K=K_list[i], r=shibor,T=T_delta)
print ('通过看涨期权计算得到的隐含波动率: ',sigma_clist)

#第3步:计算上证50ETF认沽期权(看跌)的隐含波动率,同样需要运用for语句,具体的代码如下：
sigma_plist=np.zeros_like (Put_list) #构建存放隐含波动率的初始数组
for i in np.arange (len(Put_list) ):
    sigma_plist[i]=impvol_put_Newton (P=Put_list[i], S=S0,K=K_list[i], r=shibor, T=T_delta)
print('通过看跌期权计算得到的隐含波动率: ',sigma_plist)

#第4步:将期权执行价格与隐含波动率的关系可视化,也就是绘制出波动率微笑曲线
plt.figure(figsize=(8,6))
plt.plot (K_list, sigma_clist, 'b-',label=u'50EFT认购期权(看涨) ',lw=2.5)
plt.plot (K_list, sigma_plist,'r-',label=u'50EFr认沾期权(看跌)' ,lw=2.5)
plt.xlabel (u'期权的执行价格',fontsize=13)
plt.ylabel (u'隐合波动率',fontsize=13,rotation=90)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title(u'期权执行价格与期权隐含波动率的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid ('True')
plt. show()


#%% 10.7.2  波动率斜偏
#第1步：在Python中输入相关变量，具体的代码如下：
T1=dt.datetime (2018, 12,28)#计算隐含波动率的日期
T2=dt.datetime (2019, 6,26)#期权到期日
T_delta=(T2-T1).days/365#期权的剩余期限
S0=2.289#50ETE基金净值
Call_list=np.array ([0.2866,0.2525,0.2189,0.1912,0.1645, 0.1401,0.1191,0.0996,0.0834, 0.0690,0.0566,0.0464,0.0375])#输入50ETE认购期权的收盘价格
Put_list=np.array ( [0.0540,0.0689,0.0866,0.1061,0.1294, 0.1531,0.1814,0.2122,0.2447, 0.2759,0.3162,0.3562,0.3899])##翰入50BTE认洁期权的收盘价格
K_list=np.array ([2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45, 2.5,2.55,2.6,2.65,2.7])#输入期权执行价格
shibor=0.03297#6个月期的SHIBOR利率

#第2步：计算上证50ETF认购期权(看涨期权)的隐含波动率，并且需要运用for语句，具体的代码如下:
sigma_clist=np.zeros_like(Call_list) #构建一个初始的隐含波动率数组
for i in np.arange (len (Call_list)):
    sigma_clist[i]=impvol_call_Binary(C=Call_list[i], S=S0, K=K_list[i], r=shibor, T=T_delta)
print ('通过看涨期权计算得到的隐含波动率: ',sigma_clist)

#第3步：计算上证50ETF认沽期权(看跌期权)的隐含波动率，同样需要运用for语句，具体的代码如下:
sigma_plist=np.zeros_like (Put_list) #构建一个初始的隐含波动率数组
for i in np.arange (len (Put_list)):
    sigma_plist[i]=impvol_put_Binary (P=Put_list[i], S=S0, K=K_list[i], r=shibor, T=T_delta)
print ('通过看跌期权计算得到的隐含波动率: ',sigma_plist)

 
#第4步：将期权执行价格与隐含波动率的系可视化，也就是绘制出波动率斜偏曲线，具体的代码如下:
plt.figure (figsize= (8, 6) )
plt.plot (K_list, sigma_clist, 'b-', label=u'50EFT认购期权（看涨)',lw=2.5)
plt.plot (K_list,sigma_plist, 'r-',label=u'50EET认沽期权(看跌)' ,lw=2.5)
plt.xlabel (u'期权的执行价格', fontsize=13)
plt.ylabel (u'隐含波动率', fontsize=13, rotation=90)
plt.ylim (0.18,0.27)
plt.xticks (fontsize=13)
plt.yticks (fontsize=13)
plt.title (u'期权执行价格与期权隐含波动率的关系', fontsize=13)
plt.legend (fontsize=13)
plt.grid('True')
plt. show()

















