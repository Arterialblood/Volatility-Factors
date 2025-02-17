# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 07:17:29 2020

@author: zw
"""
#%% 3.1 
import numpy as np
np.__version__


#%% 3.2
#【例3-2】根据例3-1中的信息，将4只股票的配置比例以一维数组方式直接在Python中进行输入，具体的代码如下：
weight = np.array([0.15,0.2,0.25,0.4])
type(weight)
weight.shape #显示数组的结构维度

#【例3-3】根据例3-1中的信息，将这4只股票涨跌幅以数组方式在python中进行输入，具体的代码如下：
stock_return = np.array([[0.003731,	0.021066,	-0.004854,	0.006098,	-0.006060,],
[-0.001838,	0.001842,	-0.016544,	-0.003738,	0.003752,],
[-0.003087,	-0.000344,	-0.033391,	0.007123,	0.004597,],
[-0.024112,	0.011704,	-0.029563,	-0.014570,	0.016129]])

stock_return.shape

#【例3-4】根据例3-1中的信息，将4只股票的配置比例先以列表的方式在 Python中输入，然后用array函数将列表转为一维数组，具体的代码如下：
weight_list = [0.15,0.2,0.25,0.4]
weight_array = np.array(weight_list)

#【例3-5】根据例3-1中的信息，将这4只股票涨跌幅先以列表方式在 Python中输入，然后用array和reshape函数将列表变为二维数组，具体的代码如下：
return_list = [0.003731,	0.021066,	-0.004854,	0.006098,	-0.006060, 0.001838,	0.001842,	-0.016544,	-0.003738,	0.003752, -0.003087,	-0.000344,	-0.033391,	0.007123,	0.004597, -0.024112,	0.011704,	-0.029563,	-0.014570,	0.016129]
return_array = np.array(return_list)
return_array = return_array.reshape(4,5)

return_array1 = return_array.ravel()

# 表2-3
weight_array.ndim
return_array.ndim
weight_array.size
return_array.size
weight_array.dtype
return_array.dtype

# 【例3-6】通过 NumPy快速生成包含0-9整数的数组以及1-14且步长为3的数组，具体的代码如下：
a = np.arange(20)
b = np.arange(1,15,3)

# 【例3-7】通过NumPy生成一个0-100并且元素个数为51的等差序列并且以数组形式存放，具体的代码如下：
c = np.linspace(0,100,10000)

#【例3-8】创建一个一维的零数组，数组的元素个数为5，具体的代码如下：
zeros_array1 = np.zeros(5)

#【例3-9】创建一个二维的零数组，并且是3×4形式的数组，具体的代码如下：
zeros_array2 = np.zeros((3,4))

#【例3-10】创建与前面例3-4、例3-5中已生成的weight_array，return_array同维度、同形状的零数组，具体的代码如下：
zero_weight = np.zeros_like(weight_array)
zero_return = np.zeros_like(return_array)

#【例3-11】创建与前面例3-4、例3-5中已生成的 weight_array、 return_array同维度同形状并且元素均为1的数组，具体的代码如下：
one_weight = np.ones_like(weight_array)
one_return = np.ones_like(return_array)

#【例3-12】在 NumPy中，快速创建一个5×5的单位矩阵，具体的代码如下：
I = np.eye(5)

#%% 3.3
#【例3-13】沿用例3-1中的信息，投资者希望找到工商银行这只股票在2018年9月5日的涨跌幅，对应于数组中第2行第3列，具体的代码如下：
return_array[1,2]

#【例3-14】沿用例3-1中的信息，投资者希望找出涨跌幅低于-1%的数据所在数组中的索引值，具体的代码如下：
np.where(return_array<0)

#【例3-15】沿用例3-1中的信息，投资者希望提取上汽集团、宝钢股份在2018年9月4日至9月6日的涨跌幅数据，也就是提取第3行、第4行中第2~4列的数据，具体的代码如下：
return_array[2:,1:4]

#【例3-16】沿用例3-1中的信息，投资者希望分别提取第2行的全部数据和第3列的全部数据，相关的操作如下
return_array[1]
return_array[:,2]

#【例3-17】沿用例3-1中的信息，投资者希望针对股票按照日涨跌幅进行排序，具体的代码如下：
np.sort(return_array,axis=0)
np.sort(return_array,axis=1)
np.sort(return_array)

#%% 3.4
#【例3-18】沿用例3-1中的信息，投资者按照 return_ array数组中的列、行分别求和，具体的代码如下：
return_array.sum(axis=0)
return_array.sum(axis=1)
return_array.sum()

#【例3-19】沿用例3-1中的信息，投资者按照return_array数组中的列、行分别求乘积，具体的代码如下：
return_array.prod(axis=0)
return_array.prod(axis=1)
return_array.prod()

#【例3-20】沿用例3-1中的信息，投资者按照return数组中的列、行分别求最大值和最小值，具体的代码如下：
return_array.max(axis=0)
return_array.max(axis=1)
return_array.max()

#【例3-21】沿用例3-1中的信息，投资者按照 return_array数组中的列、行分别求算术平均值，具体的代码如下：
return_array.mean(axis=0)
return_array.mean(axis=1)
return_array.mean()

#【例3-22】沿用例3-1中的信息，投资者按照 return_array数组中的列、行分别求方差、标准差，具体的代码如下：
return_array.var(axis=0)
return_array.var(axis=1)
return_array.var()

#【例3-23】沿用例3-1中的信息，投资者对return_array数组中每个元素分别计算开平方、平方，以及以e为底的指数次方，具体的代码如下：
np.sqrt(return_array)
np.square(return_array)  #乘方
np.exp(return_array)  #以e为整数次方

x = np.array([0,1,2,3,4])
x = x.reshape(1,5)
y = np.power(x, return_array)

# 【例3-24】沿用例3-1中的信息，投资者对return_array数组中每个元素分别计算自然对数，底数10的对数，底数2的对数，具体的代码如下：
np.log(return_array)
np.log10(return_array)
np.log2(return_array)

#%% 3.4
#【例3-25】沿用例3-1和例3-11的信息，将数组return_array以及与该数组具有相同的行数与列数且元素等于1的数组one_ return进行数组间的加、减运算，再将新生成的数组进行乘，除和幂运算，具体的代码如下：
new_array1 = return_array+one_return #数组相加
new_array2 = return_array-one_return #数组相减
new_array3 = new_array1*new_array2 #数组相乘
new_array4 = new_array1/new_array2 #数组相除
new_array5 = new_array1**new_array2 #数组幂运算

#【例3-26】沿用例3-1中的信息，对 return_array数组的每个元素依次加上1、减去1、乘上2、除以2以及平方，具体的代码如下：
new_array6 = return_array+1
new_array7 = return_array-1
new_array8 = return_array*2
new_array9 = return_array/2
new_array10 = return_array**2

#【例3-27】沿用例3-5中生成的数组return_array以及例3-10生成的数组zero_return，分别生成由这两个数组之间对应元素的最大值、最小值作为元素的新数组，具体的代码如下：
return_max = np.maximum(return_array,zero_return)
return_min = np.minimum(return_array,zero_return)

#【例3-28】沿用例3-1中的信息，计算4只股票涨跌幅的相关系数矩阵，运用corrcoef函数可以直接得到计算结果，corrcoef是相关系数英文名correlation coefficient的缩写，具体的代码如下：
corrcoef_return = np.corrcoef(return_array)

#表3-3  计算矩阵的性质
np.diag(corrcoef_return)
np.triu(corrcoef_return)
np.tril(corrcoef_return)
np.trace(corrcoef_return)
np.transpose(return_array)

#【例3-29】沿用例3-1中的信息，按照每只股票在投资组合中的配置比例（权重）求出相应每个交易日投资组合的平均收益率，也就相当于求矩阵之间的内积，是运用到函数dot,具体的代码如下：
average_return = np.dot(weight_array,return_array)

average_return = np.dot(return_array.T,weight_array)


import numpy.linalg as la

#表3-4
la.det(corrcoef_return)
la.inv(corrcoef_return)
la.eig(corrcoef_return)
la.svd(corrcoef_return)


#%% 3.5
#【例3-30】假定从均值为1、标准差为2的正态分布中抽取随机数，同时设定抽取随数的次数为1万次，具体的代码如下：
import numpy.random as npr
x_norm = npr.normal(loc=1.0,scale=2.0,size=10000)
print('从正态分布中抽取的平均值',x_norm.mean())
print('从正态分布中抽取的标准差',x_norm.std())


#【例3-31】假定从标准正态分布中抽取随机数，并且抽取随机数的次数依然是1万次，有3个函数可供选择，分别是 randn、 standard_ normal以及 normal函数，具体的代码如下：
x_snorm1 = npr.randn(10000)
x_snorm2 = npr.standard_normal(size=10000)
x_snorm3 = npr.normal(loc=0,scale=1.0,size=10000)
print('运用randn函数从正态分布中抽取的样本平均值',x_snorm1.mean())
print('运用randn函数从正态分布中抽取的样本标准差',x_snorm1.std())
print('运用standard_normal函数从正态分布中抽取的样本平均值',x_snorm2.mean())
print('运用standard_normal函数从正态分布中抽取的样本标准差',x_snorm2.std())
print('运用normal函数从正态分布中抽取的样本平均值',x_snorm3.mean())
print('运用normal函数从正态分布中抽取的样本标准差',x_snorm3.std())

#【例3-32】假定随机变量X的对数服从均值为0.5，标准差为1.0的正态分布，对变量进行随机抽样，并且抽取随机数的次数依然是1万次，具体的代码如下：
x_logn = npr.lognormal(mean=0.5,sigma=1.0,size=100000)
print('从对数正态分布中抽样的平均值',x_logn.mean())
print('从对数正态分布中抽样的标准差',x_logn.std())
print('对数正态分布总体的数学期望：',np.exp(0.5+1**2/2))
print('对数正态分布总体的标准差：',np.sqrt(np.exp(2*0.5+1**2)*(np.exp(1**2)-1)))

import matplotlib.pyplot as plt  #导入matplotlib子模块pyplot
from pylab import mpl  #导入子模块mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #以黑体字显示中文
mpl.rcParams['axes.unicode_minus']=False #解决保存图像是负号‘_’为显示方块的问题
plt.figure(figsize=(8,5))
plt.hist(x_logn,label=[u'对数正态分布的抽样组'],stacked=True,bins=30,edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=0)
plt.title(u'对数正态分布随机抽取的两组样本值堆叠的直方图')
plt.legend(loc=0,fontsize=13)
plt.grid(True)




#【例3-33】假定分别从自由度是4和100的卡方分布中抽取随机数，并且抽取随的次数依然是1万次，具体的代码如下：
x_chi1 = npr.chisquare(df=4,size=10000)
x_chi2 = npr.chisquare(df=100,size=10000)
print('从自由度为4的卡方分布中抽样的平均值',x_chi1.mean())
print('从自由度为4的卡方分布中抽样的标准差',x_chi1.std())
print('从自由度为100的卡方分布中抽样的平均值',x_chi2.mean())
print('从自由度为100的卡方分布中抽样的标准差',x_chi2.std())

plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
plt.hist(x_chi1,label=[u'自由度为4的卡方分布的抽样组'],stacked=True,bins=30,edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=0)
plt.title(u'自由度为4的卡方分布随机抽取的两组样本值堆叠的直方图')
plt.legend(loc=0,fontsize=13)
plt.grid(True)

plt.subplot(2,1,2)
plt.hist(x_chi2,label=[u'自由度为100的卡方分布的抽样组'],stacked=True,bins=30,edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=0)
plt.title(u'自由度为100的卡方分布随机抽取的两组样本值堆叠的直方图')
plt.legend(loc=0,fontsize=13)
plt.grid(True)

#【例3-34】假定分别从自由度是2和120的学生t分布中抽取随机数，并且抽取随机数的次数依然是1万次，具体的代码如下：
x_t1 = npr.standard_t(df=2,size=10000)
x_t2 = npr.standard_t(df=120,size=10000)
print('从自由度为2的t分布中抽样的平均值',x_t1.mean())
print('从自由度为2的t分布中抽样的标准差',x_t1.std())
print('从自由度为120的t分布中抽样的平均值',x_t2.mean())
print('从自由度为120的t分布中抽样的标准差',x_t2.std())

plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
plt.hist(x_t1,label=[u'自由度为2的t分布的抽样组'],stacked=True,bins=30,edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=0)
plt.title(u'自由度为2的t分布随机抽取的两组样本值堆叠的直方图')
plt.legend(loc=0,fontsize=13)
plt.grid(True)

plt.subplot(2,1,2)
plt.hist(x_t2,label=[u'自由度为120的t分布的抽样组'],stacked=True,bins=30,edgecolor='k')
plt.xticks(fontsize=13)
plt.xlabel(u'样本值',fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u'频数',fontsize=13,rotation=0)
plt.title(u'自由度为120的t分布随机抽取的两组样本值堆叠的直方图')
plt.legend(loc=0,fontsize=13)
plt.grid(True)

#【例3-35】假定从自由度n1=6和n2=8的F分布中抽取随机数，并且抽取随机数的数依然是1万次，具体的代码如下：
x_f = npr.f(dfnum=6,dfden=8,size=10000)
print('从自由度n1=6,n2=8的F分布中抽样的平均值',x_f.mean())
print('从自由度n1=6,n2=8的F分布中抽样的标准差',x_f.std())

#【例3-36】假定从a=2、β=4的贝塔分布中抽取随机数，并且抽取随机数的次数依然是1万次，具体的代码如下：
x_beta = npr.beta(a=2,b=4,size=10000)
print('从贝塔分布中抽样的平均值',x_beta.mean())
print('从贝塔分布中抽样的标准差',x_beta.std())

#【例3-37】假定从α=1、β=3的伽玛分布中抽取随机数，并且抽取随机数的次数依然是1万次，具体的代码如下：
x_gamma = npr.gamma(shape=1.0,scale=3.0,size=10000)
print('从伽马分布中抽样的平均值',x_gamma.mean())
print('从伽马分布中抽样的标准差',x_gamma.std())







