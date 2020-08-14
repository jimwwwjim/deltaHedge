
'''
delta hedging trial

backtest framework

author: JIMWWWJIM
'''


from WindPy import *
w.start()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, sqrt, log
from random import seed, gauss
import datetime
from time import clock
from scipy.stats.distributions import norm
import scipy



dt = 0     #相邻节点之间的距离
Niter = 0   #总结点contract_enddate = '2021-01-31'
contract_startdate = '2020-07-31'
contract_enddate = '2020-08-31'

def time_remain(contract_enddate):
    t_end = datetime.datetime.strptime(contract_enddate,'%Y-%m-%d')
    date_end = t_end.date()
    date_now = datetime.datetime.now().date()
    time_delta = date_end - date_now
    re_days = time_delta.days
    T = re_days/365
    return re_days,T
remain_days, remain_T = time_remain(contract_enddate)
itertype = '1day'
t_start = datetime.datetime.strptime(contract_startdate,'%Y-%m-%d')
t_end = datetime.datetime.strptime(contract_enddate,'%Y-%m-%d')
t_delta = t_end - t_start
print(t_delta)

def get_dt_Niter(itertype):
	if itertype == '1day':
		dt = 1/360
		Niter = t_delta.days
	elif itertype == '30min':
		dt = 1/3000        # waiting for modification
		Niter = t_delta.days*12
	elif itertype == '1hour':
		dt = 1/1500       # waiting for modification
		Niter = t_delta.days*6
	return dt,Niter

dt, Niter = get_dt_Niter(itertype)


