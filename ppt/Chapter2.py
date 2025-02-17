# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:55:24 2020

@author: zw
"""
#%%
#【例2-1】假定利率是8%，可以采用如下代码对利率变量进行赋值：
rate=0.08  #利率等于8%

#%%
#【例2-2】假定A投资者持有一家上市公司100股股票，通过 Python进行赋值，并且通过type函数判断数据类型，具体的代码如下：
share=100	
type(share)

#【例2-3】运用前面例22中的相关信息，在 Python中输入数字时，用户在100后面增加了一个小数点，则数据类型结果就是浮点型而非整型，具体的代码如下:
Share=100.
type(Share)  


#【例2-4】一个变量a等于复数2+3i，在 Python中进行输入，并且判断数据类型，具体的代码如下：
a=2+3j 
type(a)

#【例2-5】在 Python中，以字符串的数据类型依次输入 finance、 risk management、金融风险管理、888、1+1，具体的代码如下:
a='finance'  
type(a)  
 
b="risk management"  
type(b)  
#注意，risk和management之间的空格也是占据一个字符。
c="金融风险管理"  
type(c)  

d="888"  
type(d)  

e="1+1"  
type(e)  

#【例2-6】运用例2-5中输入的字符串 finance，依次索引 finance这个单词的首字母f、单词的第4个字母a以及单词最后一个字母e，具体的代码如下：
a="finance"       #索引首字母
a[0]  
 
a[3]               #索引第四个字母
 
a[-2]              #索引最后一个字母
 
#【例2-7】假定用户在 Python中输入字符串" I love risk”，并且在该字符串中截取子字符串“love”，具体的代码如下
x="I love risk"   
x[2:6]        #截取字符串love

#【例2-8】沿用例2-7的信息，假定用户希望从” I love risk"的字符串中截取” love risk”，具体的代码如下：
x[1:]  

#【例2-9】沿用例2-7的信息，假定用户希望从“ I love risk”的字符串中截取“ I love"，具体的 Python代码如下
x[:6]  

#【例2-10】假定用户在 Python中输入字符串“I love risk management”，并且希望从该字符串中截取从第3位至第9位并且步长为3（也就是跳2格选取）的子字符串，具体的代码如下:
x="I love risk management"  
x[2:9:3]     #截取从第3位至第9位并且步长为3的子字符串
 
#【例2-11】假定用户在事后检查中发现，原本希望在 Python中输入”I love finance”，但是却误写成了”I love management”，此时运用 replace函数进行修改，具体的代码如下:
y="I love management"                 #错误的更正
y.replace("management","finance")  
##% 

#%% 
#【例2-12】在 Python中，创建一个空元组，具体的代码如下：
tup1=()  	
type(tup1)  

#【例2-13】在 Python中，创建一个仅包含一个元素3的元组，具体的代码如下：
tup2=(3,)     #在元素后面加上一个逗号
type(tup2)  

tup3=(3.)       #没有在元素后面加一个逗号
type(tup3)  

#【例2-14】在 Python中创建一个元组，该元组包含的元素包括 finance、风险管理2019、8.88，相关的代码如下：
tup4=('finance','风险管理',2019,8.88)  
type(tup4)  

#【例2-15】沿用例2-14中的信息，分别访问元组的第1个元素、最后一个元素以及同时访问第2个和第3个元素，相关的代码如下：
tup4[0]           #访问元祖的第一个元素
 
tup4[-1]           #访问元祖最后一个元素   
 
tup4[1:3]           #访问元祖第2个至第3个元素  
   

#【例2-16】在 Python中创建一个空列表，相关的代码如下：
list1=[]  
type(list1)  

#【例2-17】在 Python中创建一个列表，该列表包含的元素包括 finance、 risk management、金融风险管理、2020、8.88，具体的代码如下：
list2=['finance','risk management',"金融风险管理",2020,8.88]  
type(list2)  

#【例2-18】对例2-17中创建的列表进行访问，分别访问该列表中的第一个元素、最后一个元素以及从第3个至第4个元素，相关的代码如下：
list2[0]  

list2[-1]  
 
list2[2:4]  

#【例2-19】对例2-17中创建的列表中，得到元素2020在列表中的索引值，需要运用index函数，具体的代码如下：
list2.index(2020)  

#【例2-20】在例2-16创建的空列表中添加新的元素，新的元素是2018年1月15日至19日这一个交易周沪深300指数（本书第8章的8.1节将介绍主要的股票指数）每日的涨跌幅，分别是-0.54%、0.77%、0.24%、0.87%以及0.38%，具体的代码如下:
list1.append(-0.0054)  
list1.append(0.0077)  
list1.append(0.0024)  
list1.append(0.0087)  
list1.append(0.0038)  
list1  

#【例2-21】 对例2-20的列表list1，删除列表中第3个元素0.024，具体的代码如下： 
list1.remove(0.0024)  
list1  

#【例2-22】针对包括2、4、6、8、10、2、4、2元素的列表，删除列表中数值是2的第1个元素，具体的代码如下：
list3=[2,4,6,8,10,2,4,2]  
list3.remove(2)  
list3  

#【例2-23】对例2-22中的列表list3，将列表中的全部元素进行清空处理，返回一个空列表，具体的代码如下：
list3.clear()  
list3  

#【例2-24】针对例2-21中的列表list1，在列表第3个元素的位置重新插入元素0.0024，具体的代码如下:
list1.insert(2,0.0024)  
list1  

#【例2-25】针对例2-24的列表list1，分别按照由小到大、由大到小进行排序，具体操的代码如下：
list1.sort()                    #从小到大排序 
list1  

list1.reverse()                  #由大到小排序
list1  

#【例2-26】假定有一个列表:[1,2,3,1,5,1,6,9,1,2,7]，需要计算该列表中，数字1和2出现的次数，具体的代码如下：
list4=[1,2,3,1,5,1,6,2,9,1,2,7]  
list4.count(1)        #计算列表中数字1出现的次数

list4.count(2)        #计算列表中数字2出现的次数


#【例2-27】分别创建两个集合，一个集合包含上证综指、深圳成指、恒生指数、日经225指数、道琼斯指数等元素，另一个集合则包含标普500指数、道琼斯指数、沪深300指数、日经225指数、法国CAC40指数、德国DAX指数等元素，具体的代码如下：
set1={"上证综指","深圳成指","恒生指数","日经225指数","道琼斯指数"}  
type(set1)  

set2={"标普500指数","道琼斯指数","沪深300指数","日经225指数","法国CAC指数","德国DAX指数"}  
type(set2)  

#【例2-28】针对例2-27中创建的两个集合，求这两个集合的并集，具体的代码如下：
set1|set2  

#【例2-29】针对例227中创建的两个集合，求这两个集合的交集，具体的代码如下：
set1&set2  

#【例2-30】针对例227中创建的两个集合，分别求set1对set2的差集、set2对set1差集，具体的代码如下：高亮 
set1-set2                 #set1对set2的差集

set2-set1                 #set2对set1的差集

#【例2-31】针对例2-27中创建的集合set1，在集合中增加元素“德国DAX指数”，具体的代码如下：
set1.add('德国DAX指数')  
set1  

#【例2-32】针对例2-27中创建的集合set1，。删除集合中的元素“日经225指数”，具体的代码如下：
set1.discard('日经225指数')  
set1  

#%%
#【例2-33】将表2-3中的信息以字典的形式在 Python中输入，并且分别运用直接法和间接法进行输入，具体的代码如下：
dict1={'指数名称':'沪深300','证券代码':'000300','交易日期':'2019-01-08','涨跌幅':-0.0022}  #直接法创建
dict1  

type(dict1)  

dict2={}             #间接法创建 
dict2["指数名称"]="沪深300"  
dict2["证券代码"]="000300"  
dict2["交易日期"]="2019-01-08"  
dict2["涨跌幅"]=-0.0022  
dict2  
type(dict2)  

#【例2-34】针对例2-33中创建的字典，访问并输出字典中的全部键和值，具体的代码如下：
dict1.keys()  
dict1.values()  

#【例2-35】针对例2-33中创建的字典，遍历字典的全部元素，具体的代码如下：
dict1.items()  

#【例2-36】针对例2-33中创建的字典，查找并输出涨趺幅对应的具体金额，具体的代码如下:
dict1["涨跌幅"]      #注意是用中括号

#【例2-37】针对例2-33中创建的字典，用户希望将字典中的“交易日期”对应的“2019-01-08”修改为“2019-01-07”，涨跌幅对应的-0.22%相应调整为0.61%，具体的代码如下：
dict1["交易日期"]="2019-01-07"  
dict1["涨跌幅"]="0.0061"  
dict1  

#【例2-38】针对例2-33中创建的字典，增加当日的收盘价3054.3以及成交额1057.04亿元的信息，具体的代码如下:
dict1.update({"收盘价":3054.3,"成交额(亿元)":1057.04})  
dict1  

#【例2-39】针对例2-33中创建的字典，用户希望将字典中的证券代码删除，具体的代码如下：
del dict1['证券代码']  
dict1  


#%% 2.4
#【例2-40】在Python中输入整形和浮点型，并且进行相应的加法运算，具体的代码如下：
a=2  
b=5  
c=1.  
d=3.8  
e=6.95  
a+b  
a+c  
d+e  

#【例2-41】对例2-40中输入的整型和浮点型，进行相应的减法运算，具体的代码如下
a-b                #整形与整形相减
a-c               #整形与浮点型相减
a-e              #整形与浮点型相减 
d-e  

#【例2-42】针对例2-41中，d-e输出结果需要保留小数点后2位，具体的代码如下：
round(d-e,2)  

#【例2-43】对例2-40中输入的整型和浮点型，进行相应的乘法运算，具体的代码如下：
a*b  
a*c  
d*e  

#【例2-44】对例2-40中输入的整型和浮点型，进行相应的幂运算，具体的代码如下：
a**b  
b**a  
a**c  
d**e  

#【例2-45】对例2-40中输入的整型和浮点型，进行相应的除法运算，具体的代码如下：
f=4  
f/a               #整形与整形相除
b/a               #整形与整形相除
a/c              #整形与浮点型相除  
e/d               #浮点型与浮点型相除

#【例2-46】对例2-40、例2-45中输入的整型和浮点型，进行相应的模运算，具体的代码如下：
f%a  
b%a  
d%a  
d%e  
e%d  

#【例2-47】对例2-40中输入的整型和浮点型，进行相应的整除运算，具体的代码如下： 
b//a                      #均是整形
b//d                    #一个整形一个浮点型
e//a                    #一个整形一个浮点型
e//d                    #均是浮点型

#【例2-48】在 Python中考察相应的数字是否在一个列表中，相关的代码如下：
a=1  
b=3  
c=[1,2,4,8,16]  
a in c  
b in c  

#【例2-49】在 Python中考察相应的字符串是否在一个列表中，相关的代码如下：
d='金融'  
e='风险管理'  
f=['finance','风险管理','波动率']  
d in f  
e in f  

#%% 内置函数
#Python的内置函数有许多，可以运用命令dir（__builtins__）查看，具体的代码输入和输出结果如下:
dir(__builtins__)  

#针对上面没有具体介绍的Python内置函数，如果大家感兴趣，可以运用help函数查询各种函数的具体用法，help函数的运用非常简单，输入的格式就是help()的括号内输入需要查询的函数。下面,以查询函数bin的用法作为例子进行演示,具体的代码如下：
help(zip)  

#%% 
#【例2-50】通过 Python自定义一个计算算术平均收益率的函数,具体的代码如下:
def mean_a(r):  
   '''定义一个求解算术平均收益率的函数  
   r:代表收益率的一个列表'''
   total=sum(r)
   aver = total/len(r)  
   return aver

#【例2-51】运用在例2-50中自定义的函数 mean_a求解2018年1月15日至1月19日沪深300指数每日涨趺幅(收益率〕的算数平均值,具体的代码如下:
list1=[-0.0054,0.0077,0.0024,0.0087,0.0038]  
mean_a(r=list1)
#通过以上的计算得到算术平均收益率是0.344%.此外,可以通过另一种方法验算该结果是否正确：
sum(list1)/5 

#【例2-52】用 lambda定义计算算术平均收益率的函数,并且依然用新定义的函数求解2018年1月15日至1月19日沪深300指数每日涨跌幅的算数平均值,具体的代码如下：
mean_A=lambda r:sum(r)/len(r)                      #用lambda定义函数
mean_A(r=list1)

#%%
#【例2-53】2019年1月4日,沪深300指数当天收盘上涨了2.4%,需要通过 Python的条件语句并设定一个判断条件,用于判断是正收益还是非正收益,具体的代码如下：
r1=0.024  
if r1>0:
    print("正收益",r1)  
else:  
    print("非正收益",r1)   

#【例2-54】2019年1月14日，沪深300指数当天收盘下跌了0.74%，需要设定两个判断条件，用于区分是正收益、零收益或负收益，具体的代码如下:
r2=-0.0074  
if r2>0:  
    print("正收益:",r2)  
elif r2==0:  
    print("零收益:",r2)  
else: 
    print("负收益:",r2)  

#【例2-55】通过 Python定义一个计算几何平均收益率的函数,具体的代码如下：
def mean_g(r): 
    '''定义一个计算几何平均收益率的函数 
    r:代表收益率的一个列表'''  
    total=1  
    for i in r:  
        total=total*(1+i)  
    return pow(total,1/len(r))-1  

#【例2-56】运用在例2-55中定义的函数 mean_g计算2018年1月15日至1月19日沪深300指数每日涨跌幅(收益率)的几何平均值,代码如下:
mean_g(list1) 

#【例2-57】假定需要依次输出0～10的数字，并且运用 while语句编写，具体的代码如下:
n=0  
while n<=10:  
    print("输出数字是:",n)  
    n+=1  
print("完")  

#【例2-58】在 Python中，可以通过range函数生成整数数列，同时，也可以通过 while语句生成同样的整数数列，通过while语句生成0～9的整数数列，具体的代码如下:
n=0  
list1=[]  
while n<10:  
    list1.append(n)
    n+=1  
print(list1)  

#【例2-59】2019年1月2日至1月18日，沪深300指数的每日涨跌幅如下：-1.37%、2.4%、0.61%、-0.22%、1.01%、-0.19%、0.72%、-0.87%、1.96%、0.02%、-0.55%、1.82%。对此，分别完成若干个编程任务。
#任务1:在依次访问以上这些涨跌幅数据时，一旦访问到涨幅大于2%时，就终止整个程序，并且输出已经访问的数据。就需要运用到for、if和 break搭配的语句，具体的代码如下：
r_list=[-0.0137,-0.0016,0.024,0.0061,-0.0022,0.0101,-0.0019,0.0072,-0.0087,0.0196,0.0002,-0.0055,0.0182]  
for i in r_list: 
    if i>0.02:  
        break  
    print("收益率数据:",i)  

#任务2：在依次访问以上这些涨跌幅数据时,一旦访问到涨跌幅为负数时,就跳过这些负数的数据,并且输出全部非负数的数据。这时,可以有两种不同的代码：
#第1种:运用到for、if和 continue搭配的语句结构,具体的代码如下：
for i in r_list:  
    if i<0:  
        continue
    print("非负收益率数据:",i)  
    
#第2种：运用到for、if、pass和else搭配的语句结构，具体的代码如下：
for i in r_list: 
    if i<0:  
        pass  
    else:  
        print("非负收益率数据:",i)  
        
for i in r_list: 
    if i>=0:  
        print("非负收益率数据:",i)  

#任务3:在依次访问以上这些涨跌幅数据时,按照以下3类标准:(1)涨跌幅超过1%;(2)跌幅超过-1%;(3)涨跌幅处于-1%~1%区间,将相应的数据找出来并且以数列的方式出.就需要运用for、if、elif和else搭配的语句结构,以下是具体的代码:
r3=[]  
r4=[]  
r5=[]  
for i in r_list:
    if i>0.01:  
        r3.append(i)  
    elif i<-0.01: 
        r4.append(i)  
    else:  
        r5.append(i)  
print("涨幅超过1%的数据列表:",r3)  
print("跌幅超过-1%的数据列表:",r4)  
print("涨跌幅处于[-1%,1%]区间的数据列表:",r5)  

import math
dir(math)  

#%% 2.7.5 异常捕捉处理语句
def func():
    import numpy as np
    try:
        road_num = np.random.randint(1,4)
        print("road_num == " , road_num)
        if road_num == 1:
            a=1  
            b=2
            assert a == b
        elif road_num == 2:
            a=1  
            b=0
            a/b
        else:
            print(故意输出错误)
    except AssertionError:
        print("断言错误")
    except SyntaxError:
        print("语法错误")
    except ArithmeticError:
        print("数学错误")
    except BaseException:
        print("异常基类")
    else:
        print("我在摸鱼")
    finally:
        print("我一定会露脸的")
func()


