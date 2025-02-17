# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:05:34 2020

@author: zw
"""
#%% 
#【例题1.1】一个函数，执行著名的python程序“hello world“：
# Demo file for Spyder Tutorial
# Hans Fangohr, University of Southampton, UK
def hello():
    """Print "Hello World" and return None"""
    print("Hello World")
    # main program starts here

#%% 
#【例题1.2】执行hello()函数，在console中写入hello()，然后按下Enter键：
hello()


#%%
#【例题1.3】Python提供了一个函数，它可以显示console中当前命名空间中所有已知的对象。它就是dir()：当你在console中写入dir()，将得到一个对象列表，你能看见hello在列表中：
dir()

#%% 一、下载哔哩哔哩视频

import subprocess

def download_video(video_url):
    command = f"you-get -o D:/Zhangzw/Career/投资组合/研究生教学/专题8_行业分析/ {video_url}"
    subprocess.call(command, shell=True)

# 调用函数并传入视频链接
video_url = "https://www.bilibili.com/video/BV1iN4y1X7Lu/?spm_id_from=333.337.search-card.all.click&vd_source=1f4dd51e8cbc1b4295e534754ba8f4ae"
download_video(video_url)


#%% 
#【例题1.4】一旦一个对象在当前命名空间中可见（例如本例中的hello），我们可以使用help函数了解这个对象：在console提示区写入help(hello)，你应该可以看到如下的输出：
help(hello)

#%% 
#【例题1.5】控制台（在任何给定时间内在console中定义的对象集合）在IPython中可以使用%reset命令清除。输入%reset然后按下enter键，用y确认：
%reset

#%%
#【例题1.6】使用%reset命令之后，在当前会话中只有少部分对象在命名空间中。我们可以使用dir()把他们都列出来：
dir()

#%% 这是一个程序cell
def hello():
    """Print "Hello World" and return None"""
    print("Hello World")
    # main program starts here
    
hello()
#%%

abc = 20
efg = abc+20
hij = str(efg)

#%% 【例题1.7】
mylongvariablename = 4200
myshortvariablename = 42

#%% 
from WindPy import w
import pandas as pd
import datetime

w.start()
error_code, wsd_data=w.wsd("000001.SZ", "open,high,low,close", "2020-10-9", "2020-10-10", "Fill=Previous", usedf=True)

if error_code == 0:
  print(wsd_data)
else:
  print("Error Code:", error_code)
  print("Error Message:", wsd_data.iloc[0, 0])



#%%


