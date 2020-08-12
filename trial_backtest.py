


'''
delta hedging trial

backtest framework

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
from time import clock
from scipy.stats.distributions import norm
import scipy
# pd setting
pd.set_option('display.width',320)
pd.set_option('display.max_rows',100)
# historical data collection and management
# minutes data based on wsi api function


#time structure
contract_enddate = '2021-01-31'
contract_startdate = '2020-07-31'
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

dt = 0     #相邻节点之间的距离
Niter = 0   #总结点的选择
def get_dt_Niter(itertype):
	if itertype == '1day':
    	dt = 1/360
    	Niter = t_delta.days
    	print(Niter)
	elif itertype == '30min':
    	dt = 1/3000        # waiting for modification
    	Niter = t_delta.days*12
	elif itertype == '1hour':
    	dt = 1/1500       # waiting for modification
    	Niter = t_delta.days*6
	return dt,Niter




'''
first part 

several kind of function setting
'''

#---BS formula for greeks and option prices

def BlackScholes(reTime, rf, S, K, sigma):
	d1=(log(S/K)+(rf+sigma**2/2)*reTime)/sigma*sqrt(reTime)
	d2=d1-sigma*sqrt(reTime)
	call_BS = (S*norm.cdf(d1,0,1)-K*exp(-rf*reTime)*norm.cdf(d2,0,1))
	put_BS = K*exp(-rf*reTime)*norm.cdf(-d2,0,1)-S*norm.cdf(-d1,0,1)
	delta=norm.cdf(d1,0,1)
	gamma=norm.pdf(d1,0,1)/(S*sigma*sqrt(reTime))
	vega=S*norm.pdf(d1)*np.sqrt(reTime)
	theta=-.5*S*norm.pdf(d1)*sigma/np.sqrt(reTime)
	return {'call_BS':call_BS,'put_BS':put_BS,'delta':delta,'gamma':gamma,'vega':vega,'theta':theta}

#---Monte Carlo simulations for option prices and maybe for greeks?
def MonteCarlo(reTime, rf, S, K, sigma):
	reTime = Niter
	siTi = 100000
	list_1 = []   #asian call option value list
	list_2 = []   #asian put option value list
	dt = 1/250
	totalNodes = reTime
	for si in range(siTi):
		path = [S]
		for node in range(int(totalNodes)-1):
			path.append(path[-1]*exp((rf-0.5*sigma**2)*dt+(sigma*sqrt(dt)*gauss(0,1))))
		ave_close = average(path)
		asian_put_value = max(K-ave_close,0)
		asian_call_value = max(ave_close-K,0)
		list_2.append(asian_put_value)
		list_1.append(asian_call_value)
	p = sum(list_2)/siTi
	c = sum(list_1)/siTi
	#return {'asianput_MC':p,'asiancall_MC':c,'asiandelta':delta,'asian_gamma':gamma}
	return {'asianput_MC':p,'asiancall_MC':c}

'''

def asianOption_delta(reTime, rf, S, K, sigma):
	MC_1 = MonteCarlo(reTime, rf, S*1.01, K, sigma)   #和对冲仓处理的方式不同
	MC_2 = MonteCarlo(reTime, rf, S*0.99, K, sigma)   #对方用sigma确定 delta S
	putvalue_1 = MC_1['asianput_MC']
	putvalue_2 = MC_2['asianput_MC']
	delta = (putvalue_1-putvalue_2)/(0.02*S)
	return delta

def asian_gamma(reTime, rf, S, K, sigma):
	ad_1 = asian_delta(reTime, rf, S*1.01, K, sigma)
	ad_2 = asian_delta(reTime, rf, S*0.99, K, sigma)
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
'''

def BS_delta(reTime,rf,S,K,sigma):
    return norm.cdf((log(S/K)+(rf+sigma**2/2)*reTime*dt)/sigma*sqrt(reTime*dt),0,1)

def DBA_BS(reTime,rf,S,K,sigma,min_pay):
    list_=[]
    for i in range(int(reTime)):
        list_.append(BlackScholes(reTime-i,rf,S,K,sigma)['put_BS'])
    print(np.average(list_))
    print(np.average(list_)/S)
    return np.average(list_)

def DBA_Delta(reTime,rf,S,K,sigma,min_pay):
    return (DBA_BS(reTime,rf,S+0.01,K,sigma,min_pay)-DBA_BS(reTime,rf,S-0.01,K,sigma,min_pay))/0.02

class BSCall(object):
	def __init__(self,start,T,K,N):
		self.T=T
		self.K=K
		self.start=start  #day to sell   option
		self.N=N

	def calc(self,today,vol,S,rf):
		if today<self.start:
			return {'delta':0,'call_BS':0,'vega':0,'gamma':0,'theta':0,'intrinsic':0}
		if today>self.T:
			return {'delta':0,'call_BS':0,'vega':0,'gamma':0,'theta':0,'intrinsic':0}
		if today==self.T:
			return {'delta':0,'call_BS':0,'vega':0,'gamma':0,'theta':0,'intrinsic':self.N*max(0,S-self.K)}
		#reTime=(self.T-today)/250.
		reTime=(self.T-today)/3000
		call=BlackScholes(reTime, rf, S, self.K, vol)
		return {'delta':self.N*call['delta'],'call_BS':self.N*call['call_BS'],'vega':self.N*call['vega'],'gamma':self.N*call['gamma'],'theta':self.N*call['theta'],'intrinsic':self.N*max(0,S-self.K)}

class BSPut(object):
	def __init__(self,start,T,K,N):
		self.T=T
		self.K=K
		self.start=start #day to sell option
		self.N=N
	def calc(self,today,vol,S,rf):
		if today<self.start:
			return {'delta':0,'put_BS':0,'vega':0,'gamma':0,'theta':0,'intrinsic':0}
		if today>self.T:
			return {'delta':0,'put_BS':0,'vega':0,'gamma':0,'theta':0,'intrinsic':0}
		if today == self.T:
			return {'delta':0,'put_BS':0,'vega':0,'gamma':0,'theta':0,'intrinsic':self.N*max(0,s-self.K)}
		#reTime=(self.T-today)/250.
		reTime=(self.T-today)/3000
		put=BlackScholes(reTime, rf, S, self.K, vol)
		return {'delta':self.N*put['delta'],'put_BS':self.N*put['put_BS'],'vega':self.N*put['vega'],'gamma':self.N*put['gamma'],'theta':self.N*put['theta'],'intrinsic':self.N*max(0,self.K-S)}


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




'''
second part

backtest based on the prices simulated

'''


def deltahedge_simulation_1(Niter,Sdynamics='S*=(1.0+vol*sqrt(dt)*gauss(0,1))',sigmaDynamics='sigma=.3'):
	#optiontype = 'asian'     # optiontype could be: asian, european
	#pcflag = 'put'           # pcflag could be call or put
	global dt
	dt = 1/250
	S = 8500
	Strike = 8500
	rf = .03
	sigma = .28
	cash = 0
	#original code is dt = 1/250, in a daily basis, but in this case, each iteration means 30 mins)
	# dt is defintly required
	iterToSell = 1
	iterMaturity = Niter-1
	asianput = MCAPut(iterToSell,iterMaturity,Strike,-10)
	# just for now setting
	S_list = []
	'''
	-----------------------------------------------------------------
	only if the other type of option is using this framework
	-----------------------------------------------------------------
	
	if optiontype == 'asian':
		columns = {'spot','vol','ZSS','cash','asian_put','totalValue','delta','gamma','ww','comm','pnlPredict'}
		df = pd.DataFrame([[S,vol,0,0,0,0,0,0,0,0,0]],columns=columns)
		if pcflag == 'put':
			put = MCAPut(iterToSell,iterMaturity,Strike,-10)
		elif pcflag == 'call':
			print('this is not good yet')
		else:
			print('pcflag is needed')
	elif optiontype == 'european':
		columns=('spot','vol','ZSS','cash','option','call_BS','vega','gamma','theta','comm','pnlPredict')
		df = pd.DataFrame([[S,vol,0,0,0,0,0,0,0,0,0]],columns=columns)
		if pcflag == 'call':
			call = BSCall(iterToSell,iterMaturity,Strike,-10)
		elif pcflag == 'put':
			print('this function is not available yet')
		else:
			print('pcfalg is needed')
	'''
	columns = {'spot','vol','ZSS','cash','asian_put','totalValue','delta','gamma','ww','comm','pnlPredict'}
	df = pd.DataFrame([[S,sigma,0,0,0,0,0,0,0,0,0]],columns=columns)
	put = MCAPut(iterToSell,iterMaturity,Strike,-10)

	for iter in range(1,Niter+1):
		#exec(Sdynamics)
		#exec(sigmaDynamics)
		
		
		S*=exp(rf*dt+sigma*sqrt(dt)*gauss(0,1))
		S_list.append(S)
		asianPutValue = asianput.calc(iter,sigma,S,rf)
		
		if iter == iterToSell:     # sell the put
			#asianPutValue = asianput.calc(iter,vol,S,rf)
			cash-=asianPutValue['asian_put']
			#print(asianPutValue['asian_put'])
		
		#delta hedging
		
		asianPutValue=asianput.calc(iter,sigma,S,rf)
		#print(asianPutValue)
		delta = asianPutValue['asian_delta']
		#delta = 0
		#print(delta)
		currentNumberContracts=df.iloc[iter-1].ZSS
		contractsBuy=-currentNumberContracts-delta
		cash-=contractsBuy*S
		
		#comm could be ZSS*rate or contratsBuy*rate
		
		if iter==iterMaturity:
			cash+=max(Strike-(sum(S_list)/len(S_list)),0)     #settle asian put, directly calculate the intrincsic value
		
		gamma = asianPutValue['asian_gamma']
		print('gamma:',gamma)
		theta = asianPutValue['asian_theta']
		print('theta:',theta)
		lambda_1 = 0.03                #trading cost at percentage
		comm = contractsBuy*S*lambda_1      #commision fee
		print('comm:',comm)
		ww = (1.5*exp(-rf*dt)*S*lambda_1*gamma**2)**(1/3)
		print('ww',ww)
		# based on the formula of whalley wilmott, 风险厌恶系数取1
		
		
		
		dS=S-df.iloc[iter-1].spot
		pnlPredict=0.5*gamma*dS*dS+theta*dt    # 是否成立还需要再论证
		dfnew = pd.DataFrame([[S,sigma,-delta,cash,-asianPutValue['asian_put'],cash+asianPutValue['asian_put']-delta*S,delta,gamma,ww,comm,pnlPredict]],columns=columns)
		df=df.append(dfnew,ignore_index=True)
		
	df['pnl'] = df['asian_put'] - df['asian_put'].shift(1)
	df['vol'] = 100.0*df['vol']
	df['error'] = df['pnl'] - df['pnlPredict']
	df.set_value(iterToSell,'error',0)
	
	
	#data visualization
	df.loc[:,['vol','spot']].plot(title='spot and implied volatility')
	df.loc[:,['asian_put','spot','option']].plot(title='i dont know what it is')
	df.loc[:,['delta']].plot(title='delta {0} {1}'.format(Sdynamics,sigmaDynamics))
	df.loc[:,['gamma']].plot(title='Gamma {0} {1}'.format(Sdynamics,sigmaDynamics))
	df.loc[:,['ww']].plot(title='ww value {0} {1}'.format(Sdynamics,sigmaDynamics))
	df.loc[:,['pnl']].hist(bins=50)
	print(df.loc[:,['pnl']].describe())
	print(df.head())

'''
third part

backtest based on the prices generated from Wind Terminals

'''

#data generating

itertype = '1day'     #frequency of detecting and hedging
if itertype == '30min':
    startdate = '2019-05-31 09:00:00'
    enddate = '2019-06-06 15:00:00'
    expired_data = '2019-06-31 15:00:00'
    wdata = w.wsi("RU1909.SHF", "close", startdate, enddate, "BarSize=30")
elif itertype == '1hour':
    startdate = '2019-05-31 09:00:00'
    enddate = '2019-06-06 15:00:00'
    expired_data = '2019-06-31 15:00:00'
    wdata = w.wsi("RU1909.SHF", "close", startdate, enddate, "BarSize=60")
elif itertype == '1day':
    startdate = '2019-03-31'
    enddate = '2019-06-06'
    expired_data = '2019-06-31'
    wdata = w.wsd("RU1909.SHF", "close", startdate, enddate, "")
    
prices_data = wdata.Data[0]
times_data = wdata.Times
Niter = len(prices_data)
original_data = pd.DataFrame({'prices':prices_data,'timestamp':times_data})
print(original_data)
#print(wsi_data)
#print(wsi_data.Data)
#print(wsi_data.Times)

# backtest 
def deltahedge_2(Niter):
    timer_1 = clock()
    Strike = 12000
    rf = .03
    vol = .3
    Strike = 12250
    iterToSell = 1
    iterMaturity = Niter-1
    for i in range(Niter):
        S = original_data['prices'][i]
        print(S)
            
        #asianput = MCAPut(iterToSell,iterMaturity,Strike,-10)
        #asianPutValue = asianput.calc(iter,vol,S,rf)
    timer_2 = clock()
    print('deltahedge_2 time spent:',timer_2 - timer_1)

deltahedge_2(Niter)





