# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:38:09 2020

@author: zw
"""
#%% 9.1.2  股指期货合约的介绍
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

data_IC1903 = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第9章\股指期货合约IC1903.xlsx',sheet_name='Sheet1',header=0,index_col=0) #导入外部数据
data_IC1903.plot(figsize=(10,9),subplots=True,layout=(2,2),grid=True,fontsize=13)

data_T1903=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第9章\国债期货合约T1903.xlsx',sheet_name='Sheet1',header=0,index_col=0) #导入外部数据
data_T1903.plot(figsize=(10,9),subplots=True,layout=(2,2),grid=True,fontsize=13)


#%% 9.1.4  参与期货交易的动机
#【例9-1】国内一家从事股票投资的A基金公司，在2019年1月2日购买了金额3000万元的沪深300指数ETF基金，购买时的沪深300指数恰好为3000点，这意味着指数点位上涨或下跌1个点，A公司的投资盈利或亏损1万元。为了完全对冲指数下跌的风险，基金经理可以运用沪深300股指期货合约的空头头寸进行套期保值。
spot=np.linspace(2500,3500,100)    #模拟的沪深300指数点位
profit1=spot-3000                   #现货投资（沪深300ETF指数基金）的收益
profit2=3000-spot                   #用于套期保值的期货合约收益
profit_total=profit1+profit2        #包含期货在内的整个投资组合收益

plt.figure(figsize=(8,6))
plt.plot(spot,profit1,label=u'沪深300指数基金',lw=2.5)
plt.plot(spot,profit2,label=u'沪深300指数期货合约',lw=2.5)
plt.plot(spot,profit_total,label=u'套期保值的投资组合',lw=2.5)
plt.xlabel(u'沪深300指数点位',fontsize =13)
plt.xticks(fontsize=13)
plt.ylabel(u'盈亏（万元）',rotation=0,fontsize =13)
plt.yticks(fontsize=13)
plt.title(u'空头套期保值的盈亏情况')
plt.legend(fontsize=13)
plt.grid('True')
plt.show()


#%%9.2.2  追加保证金的风险
#第1步：从外部导入沪深300指数期货F1903合约2019年1月2日至2月28日的结算价格数据，并且输入相关的期货合约的信息，具体的代码如下
data_IF1903=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第9章\沪深300指数期货合约IF1903（2019年1月至2月）.xlsx',sheet_name='Sheet1',header=0,index_col=0)  #导入外部数据
M0=4500000         #初始保证金水平
F0=30000000        #股指期货合约的初始价值
P0=3000            #初始股指期货成交价格

#第2步：分别计算并构造出1月2日至2月末每个交易日期货合约盈亏、累积盈亏以及保证金余额的3个时间序列，具体的代码如下：
return_total_IF1903=-F0*(data_IF1903/P0-1)    #期货合约（空头头寸）累计盈亏
return_total_IF1903=return_total_IF1903.rename(columns={'IF1903结算价':'合约累计盈亏'})     #将列名进行变更

return_daily_IF1903=return_total_IF1903-return_total_IF1903.shift(1)    #计算期货合约每日的盈亏
return_daily_IF1903.iloc[0]=return_total_IF1903.iloc[0]    #第一个交易日盈亏等于第一个交易日的累计盈亏
return_daily_IF1903=return_daily_IF1903.rename(columns={'合约累积盈亏':'合约当日盈亏'})  #将列名进行变更

margin_daily_IF1903=return_total_IF1903+M0        #每日期货保证金余额（不考虑追加保证金）
margin_daily_IF1903=margin_daily_IF1903.rename(columns={'合约累积盈亏':'保证金余额'})  #将列名进行变更


#第3步：将以上3个数据框进行拼接，并且输出最终的计算结果，具体的代码如下：
future_data=pd.concat([return_daily_IF1903,return_total_IF1903,margin_daily_IF1903],axis=1)    #将3个数据框按列进行拼接

#%% 9.2.3  基差风险
#【例9-4】以沪深300指数期货IF l812合约作为分析对象，展示股指期货基差的变化情况。IF1812合约的上市首日是2018年4月23日，最后交易日是2018年12月21日，计算该期货合约在存续期内的基差，每个交易日的基差用该期货合约的每个交易日结算价格去沪深300指数收盘价的差额表示。下面，直接运用 Python进行计算并演示，分为3个步完成。
#第1步：从外部导入2018年4月23日至208年12月21日期间期货IF1812合约和沪深300指数的相关数据，并且通过绘图方式进行可视化（见图9-4），具体的代码如下：
data=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第9章\沪深300指数与期货合约IF1812的数据.xlsx',sheet_name='Sheet1',header=0,index_col=0)  #导入外部数据
data.describe()

data.plot(figsize=(9,6),grid=True,fontsize=13)

#第2步：计算期货IF1218合约的基差，并且对基差进行统计分析，具体的代码如下：
basis=data.iloc[:,0]-data.iloc[:,1]   #计算期货IF1218合约的基差
basis.describe()

#第3步：绘制期货IF1218合约的基差走势图（见图9-5）,为了方便对比，生成一个基差0的时间序列作为比较基准，具体的代码如下：
zero_basis=np.zeros_like(basis)      #生成基差等于0的数组
zero_basis=pd.DataFrame(zero_basis,index=basis.index)  #基差等于0的时间序列

plt.figure(figsize=(8,6))
plt.plot(basis,'b-',label=u'基差',lw=2.0)
plt.plot(zero_basis,'r-',label=u'基差等于0',lw=3.0)
plt.xlabel(u'日期',fontsize =13)
plt.ylabel(u'基差',rotation=0,fontsize =13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数期货IF1812合约的基差（2018/4/23至2018/12/21）',fontsize=13)
plt.legend(fontsize=14)
plt.grid('True')
plt.show()


#%% 9.2.4  交叉套期保值
#【例9-5】一家国内的E基金公司持有上证180指数ETF基金（代码50180.SH），该公司希望通过股指期货合约对该基金进行套期保值，假定可选的套期保值期货合约分别是沪深300指数期货IF1812合约、中证500指数期货IC1812合约以及上证50指数期货1H1812合约，这3个期货合约的挂牌日均是2018年4月23日，最后交易日均是2018年12月21日。因此E基金公司需要从这3个合约中选择最合适的合约，并且计算相应的最优套保比率。
#第1步：从外部导入2018年4月23日至12月21日期间每日的上证180指数ETF基金净值以及IF1812合约、IC1812合约和1H1812合约的结算价格，具体的代码如下：
data2=pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第9章\上证180ETF与期货合约的数据.xlsx',sheet_name='Sheet1',header=0,index_col=0)  #导入外部数据

#第2步：计算并生成上证180ETF基金、IF812合约、IC1812合约以及IH1812合约的日收益率序列，具体Python代码如下：
SH180ETF_return=data2.iloc[:,0]/data2.iloc[:,0].shift(1)-1  #生成上证180ETF基金的日收益率数据
SH180ETF_return=SH180ETF_return.dropna()                    #缺失数据处理

IF1812_return=data2.iloc[:,1]/data2.iloc[:,1].shift(1)-1  #生成沪深300指数期货IF1812合约的日收益率数据
IF1812_return=IF1812_return.dropna()                    #缺失数据处理

IC1812_return=data2.iloc[:,2]/data2.iloc[:,2].shift(1)-1  #生成中证500指数期货IC1812合约的日收益率数据
IC1812_return=IC1812_return.dropna()                    #缺失数据处理

IH1812_return=data2.iloc[:,3]/data2.iloc[:,3].shift(1)-1  #生成上证50指数期货IH1812合约的日收益率数据
IH1812_return=IH1812_return.dropna()                    #缺失数据处理

#第3步：以上证180ETF基金的日收益率作为被解释变量，同时依次以IF1812合约的日收益率、IC1812合约的日收益率以及IH1812合约的日收益率作为解释变量，分别构建3个
import statsmodels.api as sm           #导入StatsModels的子模块api

IF1812_return_addcons=sm.add_constant(IF1812_return)   #增加常数项的时间序列
IC1812_return_addcons=sm.add_constant(IC1812_return)   #增加常数项的时间序列
IH1812_return_addcons=sm.add_constant(IH1812_return)   #增加常数项的时间序列

model_SH180ETF_IF1812=sm.OLS(SH180ETF_return,IF1812_return_addcons).fit()    #构建上证180ETF基金日收益率与沪深300指数期货IF1812合约日收益率的线性回归模型
model_SH180ETF_IF1812.summary()

model_SH180ETF_IC1812=sm.OLS(SH180ETF_return,IC1812_return_addcons).fit()    #构建上证180ETF基金日收益率与中证500指数期货IC1812合约日收益率的线性回归模型
model_SH180ETF_IC1812.summary()

model_SH180ETF_IH1812=sm.OLS(SH180ETF_return,IH1812_return_addcons).fit()    #构建上证180ETF基金日收益率与上证50指数期货IH1812合约日收益率的线性回归模型
model_SH180ETF_IH1812.summary()


#第4步：将最终拟合得到的最优套保比率通过可视化的方式进行展示（见图9-6），具体的代码如下：
model_SH180ETF_IF1812.params    #输出线性回归结果的常数项和贝塔值

plt.figure(figsize=(9,6))
plt.scatter(IF1812_return,SH180ETF_return,marker='o')
plt.plot(IF1812_return,model_SH180ETF_IF1812.params[0]+model_SH180ETF_IF1812.params[1]*IF1812_return,'r-',lw=2.5)
plt.xlabel(u'沪深300指数期货IF1812合约',fontsize =13)
plt.xticks(fontsize=13)
plt.ylabel(u'上证180ETF',rotation=90,fontsize =13)
plt.yticks(fontsize=13)
plt.title(u'沪深300指数期货IF1812合约与上证180ETF金的日收益率散点图（2018年4至12月）',fontsize=13)
plt.grid('True')
plt.show()


#%% 3、套保的最优合约数量
def N(h,Q_A,Q_F):
    '''构建计算最优期货合约的份数
    h：代表了最优套保比率；
    Q_A：代表了被对冲资产的数量；
    Q_F：代表了1份期货合约的规模。'''
    return h*Q_A/Q_F

# 【例9-6】沿用例9-5的相关信息，假定E基金公司在2018年12月28日（周五）按照收盘净值2.768元买入了上证180指数ETF基金共计1亿元，与此同时，运用沪深300期货IF1901合约空头头寸并且用当天结算价3003.6计算套期保值的期货合约数量，该期货合约到期日是2019年1月18日到期,并且假设最优套保比率是按照例9-5计算得到的0.873536根据式子（9-4）可以得到，最优期货合约份数的计算如下：
Value_asset=100000000.0     #被套期保值资产的金额
Value_future =3003.6*300    #初始期货合约价值

N_future=N(h=model_SH180ETF_IF1812.params[1],Q_A=Value_asset,Q_F=Value_future)
print('沪深300指数最优套保份数（空头）',round(N_future,0))


#【例9-7】沿用例9-6的信息，假定当E基金公司在2018年12月28日完成了套期保值以后，随着上证180指数ETF基金净值与沪深300期货IF1901合约结算价的变化，套期保值的效果也随之发生变动，表9-9选取了2019年1月11日、1月14日和1月15日共3个交易日基金净值和期货合约结算价的情况，分别计算在这3个交易日内套期保值的效果情况（不考虑追加保证金因素）
P0_ETF=2.768       #上证180指数ETF基金在2018年12月28日的净值
P0_future=3003.6   #期货合约在2018年12月28日的结算价
N=97               #最优期货合约数量
Price_ETF_list=np.array([2.836,2.815,2.867])      #上证180指数ETF基金在3个交易日的净值
Price_future_list=np.array([3095.0,3069.4,3126.2])  #期货合约在3个交易日的结算价

profit=Value_asset*(Price_ETF_list/P0_ETF-1)-N*(Price_future_list-P0_future)*300
print('2019年1月11日套期保值组合的累计收益',round(profit[0],2))
print('2019年1月14日套期保值组合的累计收益',round(profit[1],2))
print('2019年1月15日套期保值组合的累计收益',round(profit[2],2))

#%% 9.3.1计息天数规则
def accrued_interest(par,c,m,t1,t2,t3,t4,rule):
    '''按照利息的天数计算惯例求债券的应计利息
    par：代表债券的本金；
    c：代表债券的票面利率；
    m：代表没鸟票息的支付次数
    t1：代表了非参考期间的开头日期，用元组的数据结构输入，输入格式（年，月，日）；
    t2：代表了非参考期间的结束日期，数据结构和输入形式同t1；
    t3：代表了参考期间的开头日期，数据结构和输入形式同t1；
    t4：代表了参考期间的结束日期，数据结构和输入形式同t1；
    rule：选择天数计算的3种惯例，输入'actual/actual'表示实际天数/实际天数，
    'actual/360'表示实际天数/360，'actual/365'表示实际天数/365。'''
    import datetime as dt     #导入datetime模块
    d1=dt.datetime(t2[0],t2[1],t2[2])-dt.datetime(t1[0],t1[1],t1[2])     #非参考期间
    if rule=="actual/actual":
        d2=dt.datetime(t4[0],t4[1],t4[2])-dt.datetime(t3[0],t3[1],t3[2])  #参考期间
        interest=(d1.days/d2.days)*par*c/m
    elif rule=="actual/360":
        interest=(d1.days/360)*par*c
    else:
        interest=(d1.days/365)*par*c
    return interest

R1=accrued_interest(par=1000,c=0.0385,m=2,t1=(2018,2,1),t2=(2018,6,18),t3=(2018,2,1),t4=(2018,8,1),rule='actual/actual')
R2=accrued_interest(par=1000,c=0.0385,m=2,t1=(2018,2,1),t2=(2018,6,18),t3=(2018,2,1),t4=(2018,8,1),rule='actual/360')
R3=accrued_interest(par=1000,c=0.0385,m=2,t1=(2018,2,1),t2=(2018,6,18),t3=(2018,2,1),t4=(2018,8,1),rule='actual/365')
print('按照“实际天数/实际天数”的惯例计算期间利息',round(R1,4))
print('按照“实际天数/360数”的惯例计算期间利息',round(R2,4))
print('按照“实际天数/365数”的惯例计算期间利息',round(R3,4))


#%% 9.3.2  国债的报价
#【例9-9】沿用例9-8中关于“18国债04”的相关信息，假定今天是2019年1月18日，该债券的到期收益率是3.1%，B投资者希望计算该国债的全价、应计利息以及净价。相关的计算分为3个步骤。
#第1步：输入“18国债04”的相关参数，同时运用在7.4节中通过 Python自定义的计算债券价格的函数Bond_value计算“18国债04”的全价，具体的代码如下：
import datetime as dt       #导入datetime模块

t_mature=dt.datetime(2028,2,1)    #国债最后的到期日
t_previous=dt.datetime(2018,8,1)  #国债前一次的付息日
t_pricing=dt.datetime(2019,1,18)  #国债定价日
t_next=dt.datetime(2019,2,1)      #国债下一次付息日
bond_par=100                      #国债本金
YTM=0.031                         #国债到期收益率
coupon=0.0385                     #国债票面利率
m_coupon=2                        #每年票面利率的支付次数

days_interest=(t_next-t_pricing).days        #债券定价日距离下一次付息日的天数
N=int((t_mature-t_pricing).days/182.5)+1     #剩余的付息次数并且一年按365天算
cashflow=np.ones(N)*bond_par*coupon/m_coupon #国债剩余期限内票息现金流
cashflow[-1]=100*coupon/m_coupon+bond_par    #考虑最后到期时本金支付现金流
t_list=np.arange(N)/2+days_interest/365     #债券剩余期限内每期现金流距离债券定价日的期限（按年计算）
YTM_list=np.ones_like(t_list)*YTM            #生成债券到期收益率的数组

def Bond_value(c,y,t):
    '''
    构建基于不同期限零息利率作为贴现率计算债券价格的函数
    ----------
    c : 数组，表示债券存续期内现金流
    y : 数组，表示对应于产生现金流的时刻或期限
    t : 数组，表示不同期限的零息利率
    '''
    import numpy as np
    cashflow=[];
    for i in np.arange(len(c)):
        cashflow.append(c[i]*np.exp(-y[i]*t[i]))
    return np.sum(cashflow)

dirty_price=Bond_value(c=cashflow,y=YTM_list,t=t_list)
print('18国债04债券的全价',round(dirty_price,6))


#第2步：计算从上一期的付息日2018年8月1日以后至定价日（2019年1月18日）的应计利息额，并且运用前面 Python定义的计算债券利息额的函数 accrued_interest，相关的Python代码如下：
bond_interest=accrued_interest(par=bond_par,c=coupon,m=m_coupon,t1=(2018,8,1),t2=(2019,1,18),t3=(2018,8,1),t4=(2019,2,1),rule='actual/actual')
print('18国债04债券的应计利息金额',round(bond_interest,6))

#第3步：计算今天（2019年1月18日）的债券净价，具体的代码如下：
clean_price=dirty_price-bond_interest
print('18国债04债券的净价',round(clean_price,6))


#%% 9.3.3  国债期货最终价格
def CF(r,x,n,c,m):
    '''构建计算国债期待的转换因子
    r：国债期货合约基础资产（合约标的）的票面利率；
    x：国债期货合约交割月至可交割债券下一付息月的月份数；
    n：国债期货合约到期后可交割债券的剩余付息次数；
    c：可交割债券的票面利率；
    m：可交割债券每年的付息次数。'''
    #import numpy as np
    A=1/pow(1+r/m,x*m/12)        #转换因子公式的中括号前面一项
    B=c/m+c/r+(1-c/r)/pow(1+r/m,n-1)  #转换因子公式的中括号里面一项
    D=c*(1-x*m/12)/m             #转换因子公式的最后一项
    return A*B-D                 #输出转换因子的计算结果

#【例9-10】假定分析的国债期货是在2019年3月8日到期、2019年3月13日作为最后交割日的10年期的国债期货合约T903，。用例9-8中：“18国债04”作为可交割债券，计算该债券的转换因子。
R_standard=0.03             #国债期货合约基础资产的票面利率
t_settle=dt.datetime(2019,3,13) #国债期货最后交割日
t_next2=dt.datetime(2019,8,1)   #国债期货交割日后的债券下一个付息日
months=t_next2.month-t_settle.month   #交割月到下一付息月的月份数
N2=int((t_mature-t_settle).days/182.5)+1    #国债期货交割日后的债券剩余付息次数

bond_CF=CF(r=R_standard,x=months,n=N2,c=coupon,m=m_coupon)
print('18国债04债券的转换因子',round(bond_CF,6))

#【例9-11】运用例9-8的“18国债04”作为国债期货T1903合约的可交割债券，由于该期货合约的第2个交割日是2019年3月12日，可交割债券在第2个交割日之前的上一个付息日是2019年2月1日，计算该可交割债券的应计利息。
bond_interest2=accrued_interest(par=bond_par,c=coupon,m=2,t1=(2019,2,1),t2=(2019,3,12),t3=(2019,2,1),t4=(2019,8,1),rule='actual/actual')
print('18国债04债券作为可交割债券的应计利息',round(bond_interest2,4))


#%% 9.3.4  国债期货的最廉价交割
def CTD_cost(b,f,cf):
    '''构建用于计算最廉价交割国债成本的函数
    b：输入可交割国债的净价（报价）；
    f：输入国债期货的价格；
    cf：输入可交割国债的转换因子'''
    return b-f*cf       #输出国债期货空头方的现金流出净额

bond_price_list=np.array([97.8565,101.3555,104.5917])    #3只可交割国债的价格
future_price=99.2                                      #国债期货的结算价格
CF_list=np.array([0.9908,1.028,1.0661])                  #可交割债券的转换因子

Cost=CTD_cost(b=bond_price_list,f=future_price,cf=CF_list)
print('交割16国债04的成本',round(Cost[0],4))
print('交割17国债04的成本',round(Cost[1],4))
print('交割18国债04的成本',round(Cost[2],4))


#%% 9.3.5 基于久期的套期保值策略
def N_TF(Vf,P,Df,Dp):
    '''构建计算基于久期套期保值策略的国债期货合约数量
    Vf：1手国债期货合约的价值；
    P：被套期保值的投资组合的价值；
    Df：国债期货合约基础资产在套期保值到期日的久期
    Dp：被套期保值的投资组合在套期保值到期日的久期'''
    return P*Dp/(Vf*Df)

#【例9-13】假定在2019年1月28日，一家管理市值1亿元债券投资组合的G基金公司，担心在未来1个月内市场利率会出现比较大的不利变动，进而影响到债券投资组合的价格。因此，G基金公司决定利用3月份到期的国债期货对债券投资组合进行套期保值。假定在一个月后的套期保值到期日（即2月28日），债券投资组合的久期为8.28，对于这样长久期的投资组合，需要运用10年期国债期货T1903合约进行套期保值，该期货合约在1月28日的结算价是97.725元，由于期货合约基础资产是面值为100万元的国债，因此一手T1903合约的价值是97.725方元。
#第1步：运用在7.6节通过 Python定义计算麦考利久期的函数 M_Duration，求出在套期保值到期日（2月28日）“18国债04”的久期，具体的代码如下：
bond_yield=0.032             #18国债04的到期收益率
t_hedgend=dt.datetime(2019,2,28)    #套期保值到期日
N3=int((t_mature-t_settle).days/182.5)+1    #套期保值到期日以后18国债04的剩余付息次数
cashflow_new=np.ones(N3)*bond_par*coupon/m_coupon #18国债04在套期保值到期日以后的票息现金流
cashflow_new[-1]=bond_par*(1+coupon/m_coupon)    #包括债券到期日的本金和最后一期票息的现金流
t_new=np.arange(N2)/2+(t_next2-t_hedgend).days/365   #债券剩余期限内每期现金流距套期保值到期日的期限

def M_Duration(c,y,t):
    '''
    构建一个计算麦考利久期的函数
    ----------
    c : 数组，债券存续期内的现金流.
    y : 债券的到期收益率.
    t : 数组，对应于产生现金流的时刻.
    '''
    cashflow=[]
    weight=[]
    n=len(t)
    for i in np.arange(n):
        cashflow.append(c[i]*np.exp(-y*t[i]))
    for i in np.arange(n):
        weight.append(cashflow[i]/sum(cashflow))
    duration=np.sum(t*weight)
    return duration
            
bond_duration=M_Duration(c=cashflow_new,y=bond_yield,t=t_new)
print('18国债04在套期保值到期日的久期',round(bond_duration,4))


#第2步：运用前面 Python定义的计算基于久期套期保值的国债期货合约数量的函数N_TF，求解出最终需要的国债期货T1903合约数量，具体的代码如下：
future_price=97.725          #国债期货的结算价格
future_value=future_price*1000000/100  #1手国债期货的价值
port_value=100000000         #被套期保值的债券投资组合价值
port_duration=8.28           #被套期保值的债券投资组合久期

N_future=N_TF(Vf=future_value,P=port_value,Df=bond_duration,Dp=port_duration)
print('用于套现保值的国债期货T1903的合约数量',round(N_future,0))

