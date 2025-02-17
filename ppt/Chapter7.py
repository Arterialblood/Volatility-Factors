# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:38:09 2020

@author: zw
"""

#%% 7.1.2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
LPR = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第7章\贷款基础利率（LPR）数据.xls',sheet_name="Sheet1",header=0,index_col=0) #导入外部数据
LPR.plot(figsize=(8,6),grid = True,fontsize = 13) #数据可视化


#%% 7.1.3
IBL = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第7章\银行间同业拆借利率（2018年）.xls',sheet_name="Sheet1",header=0,index_col=0) #导入外部数据
(IBL.iloc[:,0:3]).plot(figsize=(8,6),grid = True,fontsize = 13) #数据可视化

FR = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第7章\银行间回购定盘利率（2018年）.xls',sheet_name="Sheet1",header=0,index_col=0) #导入外部数据
FR.plot(figsize=(8,6),grid = True,fontsize = 13) #数据可视化

Shibor = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第7章\Shibor利率（2018年）.xls',sheet_name="Sheet1",header=0,index_col=0) #导入外部数据
(Shibor.iloc[:,3:6]).plot(figsize=(8,6),grid = True,fontsize = 13) #数据可视化


#%% 7.2
bond_GDP = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第7章\债券存量规模与GDP（2010-2018年）.xlsx',sheet_name="Sheet1",header=0,index_col=0) #导入外部数据
bond_GDP.plot(figsize=(8,6),grid = True,fontsize = 13) #数据可视化

#%% 7.2.1
bond = pd.read_excel('D:\Zhangzw\Python\Python金融数据分析\RawData\第7章\国内债券市场按照交易场所分类（2018年末）.xlsx',sheet_name="Sheet1",header=0,index_col=0) #导入外部数据
plt.figure(figsize=(9,7))
plt.pie(x=bond['债券余额(亿元)'],labels=bond.index)
plt.legend(loc=1,fontsize=13)
plt.title(u'2018年国内债券在不同市场分布情况',fontsize=13)


#%% 7.3
r=0.02                                           #一年期利率2%
M=[1,2,4,12,52,365]                             #不同复利的频次
name=['一年复利1次','每半年复利1次','每季度复利1次','每月复利1次','每周复利1次','每天复利1次'] #建立一个包含不同复利频次的字符串列表
value=[]                                         #建立一个初始的存放一年后的本息合计数的数列
i=0                                              #设置一个标量
for m in M:
    value.append(100*(1+r/m)**m)
    print(name[i],round(value[i],4))
    i=i+1

def FV(A,n,R,m):
    '''构建一个用于计算不同复利频次的投资终值函数
    A：表示初始的投资本金；
    n：表示投资期限（年）；
    R：表示年利率R是按年复利的利率；
    m：表示每年复利频次，输入Y代表1年复利1次，S代表每半年复利一次，
    Q代表每季度复利一次，M代表每月复利一次，W代表每周复利一次，
    D代表每天复利一次'''
    import numpy as np
    if m=='Y':
        return A*(1+R)**(n)
    elif m=='S':
        return A*(1+R/2)**(n*2)
    elif m=='Q':
        return  A*(1+R/4)**(n*4)
    elif m=='M':
        return  A*(1+R/12)**(n*12)
    elif m=='W':
        return A*(1+R/52)**(n*52)
    else:
        return A*(1+R/365)**(n*365)

FV_M=FV(A=100,n=1,R=0.02,m='M') #用于验证每月复利的结果
print('每月复利1次得到的本息和',round(FV_M,4))


r=0.02                                      #复利一次的年利率2%
M=np.arange(1,101)                          #生成从1到100的自然数数列
PV=100                                      #初始投资100元
FV=PV*(1+r/M)**M                            #计算投资终值

plt.figure(figsize=(8,6))
plt.plot(M,FV,'b-',lw=2.0)
plt.xlabel(u'复利频次',fontsize =13)
plt.ylabel(u'金额',fontsize =13,rotation=0)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u'复利频次与投资终值的关系图',fontsize=13)
plt.grid('True')
plt.show()


#%% 7.3.2 
def Rc(Rm,m):
    '''构建已知复利频次和对应的复利利率，计算等价连续复利利率的函数
    Rm：代表了复利频次m的复利利率
    m：代表了复利频次。'''
    import numpy as np             #导入Numpy模块
    return m*np.log(1+Rm/m)      #输出等价的连续复利利率结果

def Rm(Rc,m):
    '''构建已知复利频次和连续复利利率，计算等价的对应复利m次复利利率函数
    Rc：代表了连续复利利率
    m：代表了复利频次。'''
    import numpy as np             #导入Numpy模块
    return m*(np.exp(Rc/m)-1)      #输出等价的对应复利频次的复利利率结果

#【例7-2】假定G商业银行对外的利率报价是5%，按季度复利，计算相对应的连续复利函数。由于m=4、Rn=5%，根据等式（7-3），等价的连续复利利率等于：
R_c=Rc(Rm=0.05,m=4.)               #计算连续复利利率
print('等价的连续复利利率',round(R_c,6))

#【例7-3】假设H商业银行对外的利率报价是6%，该利率是连续复利利率，计算等价的每月复利的复利利率。由于m=12，Rn=6%，由等式（7-4）得出，与之等价的按月复利的利率等于
R_m=Rm(Rc=0.06,m=12)               #计算按月复利的复利利率
print('等价的按月复利的复利利率',round(R_m,6))


#%% 7.3.3 
#【例7-4】假如3年期连续复利的零息利率是4%，这意味着今天的100元按照该零息利率投资并且只能在3年后获得本息合计100×e^4%x3=112.7497（元）。运用 Python进行计算的代码如下：
A=100
r=0.04
T=3
FV=A*np.exp(r*T)
print('三年后到期的本息合计数',round(FV,4))


#%% 7.4.2
def Bond_price(C,M,T,m,y):
    '''构建计算债券价格的函数
    C：表示债券的票面利率；
    M：表示债券的本金；
    T：表示债券的期限，用年表示；
    m：表示债券票面利率每年的支付频次；
    y：表示贴现利率，也就是债券到期收益率。'''
    import numpy as np 
    coupon=[]                                 #建立一个初始的存放每期票息现值的列表
    for i in np.arange(1,T*m+1):
        coupon.append(np.exp(-y*i/m)*M*C/m)  #计算每期债券票息的现值并放入列表
    return np.sum(coupon)+np.exp(-y*T)*M  #输出最终的债券价格

Bond = Bond_price(C=0.0525,M=100,T=10,m=2,y=0.042)   #输入债券的相关要素
print('计算得到的债券价格',round(Bond,4))

#【例7-5】2008年6月中国国家开发银行发行了“08国开1”债券，该债券面值100元，期限为20年，票面利率5.25%，每年付息2次，起息日为2008年6月24日。假定今天是2018年6月25日，此时该债券的剩余期限为10年，贴现率假定是4.2%（连续复利）计算今天该债券的价格。

#%% 7.4.3
#【例7-6】假定有一份期限是5年、面值为100元的债券，票面利率5%，每半年付息一次，假设该债券当前的市场价格是98元。根据式子（7-5），可以得到如下的等式：
def YTM(C,M,T,m,P):
    '''构建计算债券到期收益率（连续复利）的函数
    C：债券的票面利率；
    M：债券的本金；
    T：债券的期限，用年表示；
    m：债券票面利率每年的支付频次；
    y：债券的市场价格。'''
    import scipy.optimize as so               #导入SciPy的子模块optimize
    import numpy as np 
    def f(y):
        coupon=[]                                 #建立一个初始的存放每一期票息现值的列表
        for i in np.arange(1,T*m+1):
            coupon.append(np.exp(-y*i/m)*M*C/m)  #计算每一期债券票息的现值并放入列表
        return np.sum(coupon)+np.exp(-y*T)*M-P  #相当于输出一个等于零的式子
    return so.fsolve(f,0.1)

Bond_yield=YTM(C=0.05,M=100,T=5,m=2,P=98) #得到的结果是一个列表
print('计算得到债券的到期收益率',np.round(Bond_yield,6))


#%% 7.4.4
def Bond_value(c,t,y):
    '''构建基于不同期限零息利率作为贴现率计算债券价格的函数
    c：表示债券存续期内现金流，用数组的数据结构输入；
    t：表示对应于产生现金流的时刻或期限，用数组的数据结构输入；
    y：表示不同期限的零息利率，用数组的数据结构输入。'''
    import numpy as np
    cashflow=[]                               #生成存放每期现金流现值的初始数列
    for i in np.arange(len(c)):
        cashflow.append(c[i]*np.exp(-y[i]*t[i])) #计算每期现金流现值并放入列表
    return np.sum(cashflow)

# 【例7-7】假定在2019年1月2日，债券市场上分别有剩余期限是0.25年（3个月期）、0.5年（半年期）、1年期、1.5年期和2年期的5只国债，具体债券的票面利率和债券价格信息见表7-5，基于这5只债券价格的信息运用票息剥离法计算对应期限的零息利率。
def f(R):
    R1,R2,R3,R4,R5 = R          #设置不同期限利率
    P1 =99.42                   #0.25年期国债价格
    P2 =98.83                   #0.5年期国债价格
    P3 =100.09                  #1年期国债价格
    P4 =101.32                  #1.5年期国债价格
    P5 =99.39                   #2年期国债价格
    par =100.0                  #债券面值
    C3 =0.0277                  #1年期国债的票面利率
    C4 =0.0346                  #1.5年期国债的票面利率
    C5 =0.0253                  #2年期国债的票面利率
    bond1=P1*np.exp(R1*0.25)-par     #第1只债券计算零息收益率的公式
    bond2=P2*np.exp(R2*0.5)-par     #第2只债券计算零息收益率的公式
    bond3=P3*np.exp(R3*1.0)-par*(1+C3)    #第3只债券计算零息收益率的公式
    bond4=par*(0.5*C4*np.exp(-R2*0.5)+0.5*C4*np.exp(-R3)+(1+0.5*C4)*
               np.exp(-R4*1.5))-P4    #第4只债券计算零息收益率的公式
    bond5=par*(0.5*C5*np.exp(-R2*0.5)+0.5*C5*np.exp(-R3)+0.5*C5
               *np.exp(-R4*1.5)+(1+0.5*C5)*np.exp(-R5*2))-P5    #第4只债券计算零息收益率的公式
    return np.array([bond1,bond2,bond3,bond4,bond5])

import scipy.optimize as so               #导入SciPy的子模块optimize
Zero_rates=so.fsolve(f,[0.1,0.1,0.1,0.1,0.1])
print('0.25年期零息利率（连续复利）',round(Zero_rates[0],6))
print('0.5年期零息利率（连续复利）',round(Zero_rates[1],6))
print('1年期零息利率（连续复利）',round(Zero_rates[2],6))
print('1.5年期零息利率（连续复利）',round(Zero_rates[3],6))
print('2年期零息利率（连续复利）',round(Zero_rates[4],6))

T =np.array([0.25,0.5,1.0,1.5,2.0])      #生成包含五只国债期限的数组
plt.figure(figsize=(8,6))
plt.plot(T,Zero_rates,'b-')
plt.plot(T,Zero_rates,'ro')
plt.xlabel(u'期限（年）',fontsize =13)
plt.xticks(fontsize=13)
plt.ylabel(u'利率',fontsize =13,rotation=0)
plt.yticks(fontsize=13)
plt.title(u'运用票息剥离法得到的零息曲线',fontsize=13)
plt.grid('True')
plt.show()

#2、插值处理
import scipy.interpolate as si     
func =si.interp1d(T,Zero_rates,kind='cubic')    #运用原有的数据构建一个插值函数，并运用3阶样条曲线插值法
T_new=np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0])    #生成包含0.75年、1.25年和1.75年的新数据
Zero_rates_new =func(T_new)                              #计算得到基于插值法的零息利率

plt.figure(figsize=(8,6))
plt.plot(T_new,Zero_rates_new,'o')
plt.plot(T_new,Zero_rates_new,'-')
plt.xlabel(u'期限（年）',fontsize =13)
plt.xticks(fontsize=13)
plt.ylabel(u'利率',fontsize =13,rotation=0)
plt.yticks(fontsize=13)
plt.title(u'基于插值法得到的零息曲线',fontsize=13)
plt.grid('True')
plt.show()

for i in range(len(T_new)):
    print('期限（年）',T_new[i],'零息利率',round(Zero_rates_new[i],6))


#%% 7.4.6
#【例7-8】假定在2019年1月2日，债券市场有一个债券，本金为100元，票面利率为6%，每年支付票息4次（每季度支付1.5%），剩余期限为2年，运用例7-7计算得到的零息利率曲线（见表7-6）对该债券进行定价。
coupon =0.06           #债券票面利率
m =4                   #债券票面利率的复利频次
par=100                #债券面值
bond_cashflow=np.ones_like(T_new)*par*coupon/m          #生成存放债券票息的数组
bond_cashflow[-1]=par*(1+coupon/m)                      #将债券本金和最后一期票息加入数组

bond_price=Bond_value(c=bond_cashflow,t=T_new,y=Zero_rates_new)   #输入债券的相关信息
print('计算的债券价格',round(bond_price,4))


#%% 7.5.1
zero_rate=np.array([0.025,0.028,0.032,0.037,0.045])   #生成一个零息利率的数组
def func(Rf):
    Rf2,Rf3,Rf4,Rf5=Rf          #设置不同的远期利率
    year2=100*np.exp(zero_rate[1]*2)-100*np.exp(zero_rate[0]*1.0)*np.exp(Rf2*1.0)       #计算第2年远期利率的等式
    year3=100*np.exp(zero_rate[2]*3)-100*np.exp(zero_rate[1]*2.0)*np.exp(Rf3*1.0)       #计算第3年远期利率的等式
    year4=100*np.exp(zero_rate[3]*4)-100*np.exp(zero_rate[2]*3.0)*np.exp(Rf4*1.0)       #计算第4年远期利率的等式
    year5=100*np.exp(zero_rate[4]*5)-100*np.exp(zero_rate[3]*4.0)*np.exp(Rf5*1.0)       #计算第5年远期利率的等式
    return np.array([year2,year3,year4,year5])

forward_rates =so.fsolve(func,[0.1,0.1,0.1,0.1])
print('第2年远期利率',round(forward_rates[0],6))
print('第3年远期利率',round(forward_rates[1],6))
print('第4年远期利率',round(forward_rates[2],6))
print('第5年远期利率',round(forward_rates[3],6))


def Rf(R1,R2,T1,T2):
    '''定义计算远期利率的函数
    R1：表示对应期限为T1的零息利率；
    R2：表示对应期限为T2的零息利率；
    T1：表示对应于零息利率为R1的期限长度；
    T2：表示对应于零息利率为R2的期限长度。'''
    return R2+(R2-R1)*T1/(T2-T1)     #计算远期利率的表达式

T_list = np.arange(1,6)  #快速生成一个期限的数组
Rf_result=Rf(R1=zero_rate[0:4],R2=zero_rate[1:],T1=T_list[0:4],T2=T_list[1:])
Rf_result


#%% 7.5.2
def FRA(Rk,Rm,L,T1,T2,position,when):
    '''构建计算远期利率协议现金流的函数
    Rk：表示远期利率协议中的固定利率；
    Rm：表示在T1时点观察到的[T1,T2]期间的参考利率；
    L：表示远期利率协议的本金；
    T1：表示期限的长度；
    T2：表示期限的另一个长度，T2大于T1；
    position：表示协议多头或空头，输入'long'代表多头，否则表示空头；
    when：表示需要计算的得到现金流的具体时刻，'begin'代表计算T1时刻的现金流，否则表示计算T2时刻的现金流。'''
    if position =='long':
        if when=='begin':
            return ((Rm-Rk)*(T2-T1)*L)/(1+(T2-T1)*Rm)
        else:
            return (Rm-Rk)*(T2-T1)*L
    else:
        if when=='begin':
            return ((Rk-Rm)*(T2-T1)*L)/(1+(T2-T1)*Rm)
        else:
            return (Rk-Rm)*(T2-T1)*L

#【例7-10】假定A公司预期在第1年末将向银行贷款1000万元，贷款期限是3个月。为了防范利率上涨的风险，A公司当前就与B银行签订一份远期利率协议，A公司是合约的多头，B银行是合约的空头，协议约定A公司在第1年末针对这1000万元的贷款能够获取3%的3个月期固定利率，参考利率是3个月期Shibor利率。
FRA_long_begin=FRA(Rk=0.03,Rm=0.035,L=10000000,T1=1.0,T2=1.25,position='long',when='begin')  #远期利率协议多头（A企业）在第1年末的现金流
FRA_long_end=FRA(Rk=0.03,Rm=0.035,L=10000000,T1=1.0,T2=1.25,position='long',when='end')  #远期利率协议多头（A企业）在第1年末的现金流
FRA_short_begin=FRA(Rk=0.03,Rm=0.035,L=10000000,T1=1.0,T2=1.25,position='short',when='begin')  #远期利率协议多头（A企业）在第1年末的现金流
FRA_short_end=FRA(Rk=0.03,Rm=0.035,L=10000000,T1=1.0,T2=1.25,position='short',when='end')  #远期利率协议多头（A企业）在第1年末的现金流
print('A企业在第1年末的现金流',round(FRA_long_begin,2))
print('A企业在第1.25年末的现金流',round(FRA_long_end,2))
print('B企业在第1年末的现金流',round(FRA_short_begin,2))
print('B企业在第1.25年末的现金流',round(FRA_short_end,2))


#4、远期利率协议定价
def V_FRA(Rk,Rf,R,L,T1,T2,position):
    '''构建计算远期利率协议合约价值的函数
    Rk：表示远期利率协议中的固定利率；
    Rf：表示当前观察到的未来[T1,T2]期间的远期参考利率；
    R：表示期限长度为T2的无风险利率
    L：表示远期利率协议的本金；
    T1：表示期限的长度；
    T2：表示期限的另一个长度，T2大于T1；
    position：表示协议多头或空头，输入'long'代表多头，否则表示空头；'''
    if position=='long':
        return L*(Rf-Rk)*(T2-T1)*np.exp(-R*T2)
    else:
        return L*(Rk-Rf)*(T2-T1)*np.exp(-R*T2)

#【例7-11】A公司与B银行达成一份远期利率协议，约定A公司在第2年末至第2.25年末期间将收取固定利率3.8%（每季度复利一次）。同时在第2年末至第2.25年末期间将支付3个月期的Shibor，远期利率协议的本金为1000万元。因此A公司是合约的空头，B银行是合约的多头。假定第2年末至第2.25年末期间的Shibor远期利率为4%（每季度复利一次），此外，2.25年期的无风险利率为5%（连续复利）
V_FRA_short=V_FRA(Rk=0.038,Rf=0.04,R=0.05,L=10000000,T1=2.0,T2=2.25,position='short')    #计算远期利率协议空头（A企业）的协议价值
V_FRA_long=V_FRA(Rk=0.038,Rf=0.04,R=0.05,L=10000000,T1=2.0,T2=2.25,position='long')      #计算远期利率协议多头（B企业）的协议价值
print('A企业的远期利率协议价值',round(V_FRA_short,2))
print('B企业的远期利率协议价值',round(V_FRA_long,2))


#%% 7.6.1  麦考利久期
def M_Duration(c,y,t):
    '''构建一个计算麦考利久期的函数
    c：表示债券存续期内的现金流，用数组（ndarray)的数据结构输入；
    y：表示债券的到期收益率（连续复利）；
    t：表示对应于产生现金流的时刻，用数组（ndarray）的数据结构输入。'''
    cashflow=[]      #建立存放债券每一期现金流现值的列表
    weight=[]        #建立存放在债券每一期现金流现值与债券价格比率的列表
    n = len(t)
    for i in np.arange(n):
        cashflow.append(c[i]*np.exp(-y*t[i]))   #计算得到债券每一期现金流现值的列表
    for i in np.arange(n):
        weight.append(cashflow[i]/sum(cashflow))  #计算得到每一期现金流现值与债券价格比率的列表
    duration=np.sum(t*weight)  #计算得到债券麦考利久期
    return duration            #输出债券的麦考利久期


#【例7-12】在2019年1月17日，针对发债主体是中国国家开发银行的债券“13国开09”，该债券剩余期限为4年，面值为100元，票面利率为2.95%，票息支付是每年2次（半年1次），到期收益率为3.8%（连续复利）利用式子（7-16），计算该债券的麦考利久期。
coupon =0.0295                      #债券票面利率
par=100                             #债券面值
bond_yield=0.038                    #债券到期收益率（连续复利）
t_list=np.arange(1,9)/2             #快速生成现金流期限的数组
cashflow=np.ones_like(t_list)*coupon*0.5*par    #生成一个不包括到期本金的现金流数组
cashflow[-1]=par*(1+coupon*0.5)    #将本金纳入现金流的数组

Duration=M_Duration(c=cashflow,y=bond_yield,t=t_list)
print('13国开09债券的麦考利久期',round(Duration,4))


#【例7-13】沿用前面例7-12的“13国开09”债券，假定2019年1月17日当天的债券到期收益率从3.8%增加至3.9%（连续复利），也就是增加10个基点（△y=0.1%），分别运用麦考利久期和债券定价公式两种方法计算债券价格的变动金额。
Bond=Bond_price(C=coupon,M=par,T=4,m=2,y=(bond_yield+0.001))   #输入13国开09债券的要素
print('运用债券定价公式计算得到13国开09债券的最新债券价格',round(Bond,4))


#%% 7.6.2  修正久期
def Modi_Duration(c,y,m,t):
    '''构建一个计算修正久期的函数
    c：表示债券存续期内现金流，用数组（ndarray）的数据结构输入；
    y：表示债券的到期收益率，复利频次是m次；
    m：表示复利频次；
    t：表示对应于产生现金流的时刻，用数组的数据结构输入。'''
    cashflow=[]      #建立存放债券每一期现金流现值的列表
    weight=[]        #建立存放在债券每一期现金流现值与债券价格比率的列表
    n = len(t)
    Rc=m*np.log(1+y/m)      #计算对应的连续复利的债券到期收益率
    for i in np.arange(n):
        cashflow.append(c[i]*np.exp(-Rc*t[i]))   #计算得到债券每一期现金流现值的列表
    for i in np.arange(n):
        weight.append(cashflow[i]/sum(cashflow))  #计算得到每一期现金流现值与债券价格比率的列表
    duration=np.sum(t*weight)  #计算得到债券麦考利久期
    return duration/(1+y/m)            #输出债券的麦考利久期

R2=Rm(Rc=bond_yield,m=2)    #将连续复利的债券到期收益率转化为每年复利2次的收益率
print('每年复利2次的债券到期收益率',round(R2,6))

Modified_Duration=Modi_Duration(c=cashflow,y=R2,m=2,t=t_list)     #计算债券修正久期
print('13国开09债券的修正久期',round(Modified_Duration,4))

# 债券价格的变化金额:
deltaB = -96.7421*Modified_Duration*0.001
# 债券价格下降至:
96.7421+deltaB

# 运用债券定价公式计算精确的结果:
y_continous=Rc(Rm=(R2+0.001),m=2)   #将最新的每年复利2次的债券收益率转化为连续复利
print('连续复利的债券到期收益率',round(y_continous,6))

bondNewPrice = Bond_price(C=coupon,M=par,T=4,m=2,y=y_continous)   #输入13国开09债券的要素
print('运用债券定价公式计算得到13国开09债券的最新债券价格',round(Bond,4))


#%% 7.6.3  美元久期
def Dollar_Duration(c,y,m,t):
    '''构建一个计算美元久期的函数
    c：表示债券存续期内现金流，用数组（ndarray）的数据结构输入；
    y：表示债券的到期收益率，复利频次是m次；
    m：表示复利频次；
    t：表示对应于产生现金流的时刻，用数组的数据结构输入。'''
    cashflow=[]      #建立存放债券每一期现金流现值的列表
    weight=[]        #建立存放在债券每一期现金流现值与债券价格比率的列表
    n = len(t)
    Rc=m*np.log(1+y/m)      #计算对应的连续复利的债券到期收益率
    for i in np.arange(n):
        cashflow.append(c[i]*np.exp(-Rc*t[i]))   #计算得到债券每一期现金流现值的列表
    bond_price=sum(cashflow)
    for i in np.arange(n):
        weight.append(cashflow[i]/bond_price)  #计算得到每一期现金流现值与债券价格比率的列表
    duration=np.sum(t*weight)/(1+y/m)  #计算得到债券修正久期
    return bond_price*duration           #输出债券的美元久期


#【例7-15】沿用前面例7-12的“13国开09”债养，根据在例7-14中已丝计算得到修正久期3.7268以及表7-9中债券价格的96.7421，运用式子（7-23）就可以计算得到该债券美元久期。
D_Duration=Dollar_Duration(c=cashflow,y=R2,m=2,t=t_list)    #输入13国开债券要素
print('13国开09债券的美元久期',round(D_Duration,2))


#%% 7.7  衡量债券利率风险的非线性指标——凸性
#【例7-16】沿用前面例7-12的“13国开09”债券，假定连续复利的债券到期收益变动100个基点，即从3.8%增加至4.8%，需要计算债券的最新价格。
# Bond_newprice=Bond_price(C=coupon,M=par,T=4,m=2,y=0.048)   #输入13国开09债券的要素
# print('运用债券定价公式计算得到13国开09债券的最新债券价格',round(Bond_newprice,4))

#Python自定义计算债券凸性的函数
def Convexity(c,y,t):
    '''构建一个计算债券凸性的函数
    c：表示债券存续期内现金流，用数组（ndarray）的数据结构输入；
    y：表示债券的到期收益率（连续复利）；
    t：表示对应于产生现金流的时刻，用数组的数据结构输入。'''
    cashflow=[]      #建立存放债券每一期现金流现值的列表
    weight=[]        #建立存放在债券每一期现金流现值与债券价格比率的列表
    n = len(t)
    for i in np.arange(n):
        cashflow.append(c[i]*np.exp(-y*t[i]))   #计算债券每一期现金流现值的列表
    bond_price=sum(cashflow)
    for i in np.arange(n):
        weight.append(cashflow[i]/bond_price)  #计算每一期现金流现值与债券价格比率的列表
    convexity=np.sum(weight*t**2)        #计算得到债券凸性
    return convexity          #输出债券的凸性

#【例7-17】依然沿用例7-12关于“13国开09”债券的信息，计算该债券的凸性。下表7-11显示计算“13国开09”债券凸性的完整计算过程。
Bond_conv=Convexity(c=cashflow,y=bond_yield,t=t_list)  #输入债券要素
print('13国开09债券的凸性',round(Bond_conv,4))

# 债券价格的变化金额:
deltaB = -96.7421*Modified_Duration*0.001+0.5*96.7421*Bond_conv*0.001**2
# 债券价格下降至:
96.7421+deltaB

#这一价格与利用债券定价公式（例7-14）计算得到的最新价格 bondNewPrice 元之间仅仅相差0.0001元
