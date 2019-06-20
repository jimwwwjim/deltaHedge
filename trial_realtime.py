


'''
delta hedging trial

real time hedging framework

author: JIMWWWJIM
'''


#----packages input---

#for the historical data
from WindPy import *
w.start()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, sqrt, log
from random import seed, gauss
import datetime
import time
from scipy.stats.distributions import norm
import scipy
# pd setting
pd.set_option('display.width',320)
pd.set_option('display.max_rows',100)
# historical data collection and management
# minutes data based on wsi api function

wsi_data = w.wsi('RU1909.SHF','close','2019-05-24 09:00:00','2019-05-24 10:30:57','')
prices_data = wsi_data.Data
times_data = wsi_data.Times
#print(wsi_data)
#print(wsi_data.Data)
#print(wsi_data.Times)

def MonteCarlo(reTime, rf, S, K, sigma):
	siTi = 10000
	# siTi times of simulations, could be 1,000,000
	#reTime  remaining Time, would be given in the class()
	list_1 = []   #asian call option value list
	list_2 = []   #asian put option value list
	dt = 1/3000
	for si in range(siTi):
		path = []
		w = gauss(0,1)
		totalNodes = reTime*3000
		for node in range(int(totalNodes)):
			if node == 0:
				path.append(S)
			else:
				St = path[-1]*exp((rf-0.5*sigma**2)*dt+(sigma*sqrt(dt)*w))
				path.append(St)
		ave_close = 500
		asian_put_value = max(K-ave_close,0)
		asian_call_value = max(ave_close-K,0)
		list_2.append(asian_put_value)
		list_1.append(asian_call_value)
	p = sum(list_2)/siTi
	c = sum(list_1)/siTi
	return {'asianput_MC':p,'asiancall_MC':c}


def asian_delta(reTime, rf, S, K, sigma):
	S1=S+0.01*S
	S2=S-0.01*S
	MC_1 = MonteCarlo(reTime, rf, S1, K, sigma)
	MC_2 = MonteCarlo(reTime, rf, S2, K, sigma)
	putvalue_1 = MC_1['asianput_MC']
	putvalue_2 = MC_2['asianput_MC']
	delta = (putvalue_1-putvalue_2)/(0.02*S)
	return delta

def asian_gamma(reTime, rf, S, K, sigma):
	S1 = S + 0.01*S
	S2 = S - 0.01*S
	ad_1 = asian_delta(reTime, rf, S1, K, sigma)
	ad_2 = asian_delta(reTime, rf, S2, K, sigma)
	gamma = (ad_1 - ad_2)/(0.02*S)
	return gamma

def asian_theta(reTime,rf,S,K,sigma):
	global dt
	dt = 1/3000
	tau_1 = reTime-dt
	MC_1 = MonteCarlo(tau_1, rf, S, K, sigma)
	MC_2 = MonteCarlo(reTime, rf, S, K, sigma)
	putvalue_1 = MC_1['asianput_MC']
	putvalue_2 = MC_2['asianput_MC']
	theta = (putvalue_2-putvalue_1)/dt
	return theta

class MCAPut(object):
	def __init__(self,start,T,K,N):
		self.T=T
		self.K=K
		self.start=start  #time to sell option
		self.N=N
		
	def calc(self,today,vol,S,rf):
		if today<self.start:
			return {'asian_delta':0,'asian_put':0,'asian_gamma':0,'asian_theta':0,'theta':0}
		if today>self.T:
			return {'asian_delta':0,'asian_put':0,'asian_gamma':0,'asian_theta':0,'theta':0}
		if today == self.T:
			return {'asian_delta':0,'asian_put':0,'asian_gamma':0,'asian_theta':0,'theta':0}
		#reTime=(self.T-today)/250.
		#print('class MCAPut self.T',self.T)
		#print('class MCAPut today',today)
		reTime=(self.T-today)/3000
		#print('class MCAPut reTime',reTime)
		asian_put = MonteCarlo(reTime, rf, S, self.K, vol)['asianput_MC']
		delta = asian_delta(reTime, rf, S, self.K, vol)
		#print(type(delta))
		gamma = asian_gamma(reTime, rf, S, self.K, vol)
		#print(type(gamma))
		theta = asian_theta(reTime, rf, S, self.K, vol)
		#print(type(theta))
		return{'asian_delta':self.N*delta,'asian_put':self.N*asian_put,'asian_gamma':self.N*gamma,'asian_theta':self.N*theta}

def price_required():
	wsq_data = w.wsq("RU1909.SHF", "rt_last,rt_latest")
	print(wsq_data)
	price_rt_last = wsq_data.Data[0][0]
	price_rt_latest = wsq_data.Data[1][0]
	print('price_rt_last',price_rt_last)
	print('price_rt_latest',price_rt_latest)
	return price_rt_last,price_rt_last

def european_price_required():
	wsq_data = w.wsq("RU1909P12000.SHF", "rt_last,rt_imp_volatility,rt_theta,rt_delta,rt_gamma")
	print(wsq_data)
	price_ = wsq_data.Data[0][0]
	IV_ = wsq_data.Data[1][0]
	theta_ = wsq_data.Data[2][0]
	delta_ = wsq_data.Data[3][0]
	gamma_ = wsq_data.Data[4][0]
	print('price of european option',price_)
	print('implied volatility', IV_)
	print('theta',theta_)
	print('delta',delta_)
	print('gamma',gamma_)
	return price_,IV_,theta_,delta_,gamma_

def time_remain():
	t_end = '2019-06-30'
	
def main():
	columns = ['标的资产现价','标的资产最新成交价','RU1909P12000 现价','RU1909P12000 隐含波动率','RU1909P12000 theta','RU1909P12000 delta','RU1909P12000 gamma','t']
	list_ua_price1 = []
	list_ua_price2 = []
	list_eo_price = []
	list_eo_iv = []
	list_eo_theta = []
	list_eo_delta = []
	list_eo_gamma = []
	t_list = []
	
	while True:
		now = datetime.datetime.now()
		#print(now.month,now.day)
		stamp = '2019/0'+str(now.month)+'/'+str(now.day)
		print(stamp)

		print(now.hour,now.minute)
		h_list = [9,10,11,13,14,15,21,23]
		if now.hour in h_list:
			if now.minute == 0 or now.minute == 30:
				print(now)
				ua_price_1, ua_price_2 = price_required()
				eo_price, eo_iv, eo_theta, eo_delta, eo_gamma = european_price_required()
				print('time:',now)
				print(u'标的资产现价：',ua_price_1)
				print(u'标的资产最新成交价：',ua_price_2)
				print(u'RU1909P12000 现价：',eo_price)
				print(u'RU1909P12000 隐含波动率：',eo_iv)
				print(u'RU1909P12000 theta',eo_theta)
				print(u'RU1909P12000 delta',eo_delta)
				print(u'RU1909P12000 gamma',eo_gamma)
				list_ua_price1.append(ua_price_1)
				list_ua_price2.append(ua_price_2)
				list_eo_price.append(eo_price)
				list_eo_iv.append(eo_iv)
				list_eo_theta.append(eo_theta)
				list_eo_delta.append(eo_delta)
				list_eo_gamma.append(eo_gamma)
				t_list.append(now)
		time.sleep(60)
		if now.hour == 15 and now.minute == 0:
			print(list_ua_price1)
			print(llist_ua_price2)
			print(list_eo_price)
			print(list_eo_iv)
			print(list_eo_theta)
			print(list_eo_delta)
			print(list_eo_gamma)
			print(t_list)
			df = pd.DataFrame([list_ua_price1,list_ua_price2,list_eo_price,list_eo_iv,list_eo_theta,list_eo_delta,list_eo_gamma,t_list],columns=columns)
			print(df)

main()

