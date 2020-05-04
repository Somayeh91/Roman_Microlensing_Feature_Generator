import numpy as np
from Common_functions import *
import matplotlib.pyplot as plt
import matplotlib
from numpy import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
import pandas as pd
from lmfit import minimize, Parameters, Parameter, report_fit
import matplotlib.patches as patches
from sympy.solvers import solve
from sympy import Symbol
import sympy


class Cauchy_fit(source):


	def __init__(self, filename):
		super().__init__(self, filename)
		

	def initial_guess_finder(self):

		self.A_max = self.df['A'][np.argmin(self.m[self.m>np.percentile(self.m, [0.0,100])[0]])] 
	    self.t0_ini =  self.df['t'][np.argmin(self.m[self.m>np.percentile(self.m, [0.0,100])[0]])] 
	    self.amp_ini =  self.df.A[it0] - np.median(self.df.A[self.baseline])
	    self.tE_ini = [0.01, 0.1, 1.0, 10.0, 100]
	    
	    self.fs_ini = 0.5
	    if A_max > 1:
	        u0 = Symbol('u0')
	        solve_u0 = (solve((2+u0**2)/(u0*sympy.sqrt(4+u0**2))-((A_max-(1-self.fs_ini))/self.fs_ini), u0))

	        self.u0_ini = np.array(solve_u0[0], dtype=np.float64)
	    else:
	        self.u0_ini = np.sqrt( ( (1+np.sqrt(1+16*(A_max**2)))/(2* A_max) ) - 2 )

	    if u0_ini <0.5 :
	        A_lim = 1.34
	    else:
	        A_lim = 1.06

	    interpol = interp1d(self.df['t'],self.df['A'], kind='cubic')
	    dt = np.abs(t[np.argmin(np.abs(interpol(self.df['t'])-A_lim))]-t0_true)
	    self.tE_ini_PSPL = dt

	def PSPL_fitter(self):

	    params2 = Parameters()
	    params2.add('t0', value= self.t0_ini, min=min(df['t']), max=max(df['t']))
	    params2.add('tE', value= self.tE_ini_PSPL, min=0.001, max=1000)
	    params2.add('u0', value= self.u0_ini, min=0, max=5)
	    params2.add('fs', value= self.fs_ini, min=0, max=1)


	    res2 = minimize(PSPL_data, params2, args=(self.df['t'].values, self.df['A'].values))

	    temp_PSPL[0].append(res2.params)
	    temp_PSPL[1].append(res2.chisqr)

	    result2 = temp_PSPL[0][np.argmin(temp_PSPL[1])]

	    self.PSPL_params = [result2['t0'], result2['tE'], result2['u0'], result2['fs']]

	    self.PSPL_model = PSPL(result2['t0'], result2['tE'], result2['u0'], result2['fs'] , self.df['t'])
	    
	    self.top_interval = 1*result2['tE']*result2['u0']

	    A_top = self.df['A'][(self.df.t < result2['t0'] + self.top_interval) & (self.df.t > result2['t0'] - self.top_interval)]
	    model_top = self.PSPL_model[(self.df.t < result2['t0'] + self.top_interval) & (self.df.t > result2['t0'] - self.top_interval)]
	    A_err_top = self.df['A_err'][(self.df.t < result2['t0'] + self.top_interval) & (self.df.t > result2['t0'] - self.top_interval)]

	    self.chisqr_PSPL = cal_chisqr(self.PSPL_model, self.df['A'].values, self.df['A_err'].values)
	    self.chisqr_PSPL_top = cal_chisqr(model_top, A_top.values, A_err_top.values)
	    self.chisqr_PSPL_top_reduced = self.chisqr_PSPL_top/len(model_top)
	    self.top_number = len(model_top)



	def Cauchy_fitter(self):

		temp_bell = [[], []]
	    for tE in self.tE_ini:
	        params = Parameters()
	        params.add('t0', value= self.PSPL_params[0], min=self.PSPL_params[0]-100, max=self.PSPL_params[0]+100)
	        params.add('tE', value= tE, min=0.0001, max=1000)
	        params.add('b', value= 1, min=0, max=20)
	        params.add('amp', value= self.A_max, min=0, max=1000)


	        res1 = minimize(bell_curve_data, params, args=(self.df['t'].values, self.df['A'] ))
	        temp_bell[0].append(res1.params)
	        temp_bell[1].append(res1.chisqr)

	    result1 = temp_bell[0][np.argmin(temp_bell[1])]
	    self.Cauchy_model = bell_curve(self.df.t, result1['t0'], result1['tE'] ,result1['b'], result1['amp'])
	    self.Cauchy_params = [result1['t0'], result1['tE'] ,result1['b'], result1['amp']]




	    A_top = self.df['A'][(self.df.t < self.PSPL_params[0] + self.top_interval) & (self.df.t > self.PSPL_params[0] - self.top_interval)]
	    model_top = self.Cauchy_model[(self.df.t < self.PSPL_params[0] + self.top_interval) & (self.df.t > self.PSPL_params[0] - self.top_interval)]
	    A_err_top = self.df['A_err'][(self.df.t < self.PSPL_params[0] + self.top_interval) & (self.df.t > self.PSPL_params[0] - self.top_interval)]

	    self.chisqr_Cauchy = cal_chisqr(self.Cauchy_model, self.df['A'].values, self.df['A_err'].values)
	    self.chisqr_Cauchy_top = cal_chisqr(model_top, A_top.values, A_err_top.values)
	    self.chisqr_Cauchy_top_reduced = self.chisqr_bell_top/(len(model_top))

	def feature_producer (self):

		self.b_Cauchy = self.Cauchy_params[2]
		self.psi = self.chisqr_PSPL_top_reduced-self.chisqr_Cauchy_top_reduced
		self.delta_chisqr_total = self.chisqr_PSPL - self.chisqr_Cauchy

	def plotter(self):

		if self.t.min() > 2458234:

	    	t_shift = -2458234
	    else: 
	    	t_shift = 0

	    xdata = self.df['t'] + t_shift
	    ydata = self.df['A']
	    model1 = self.PSPL_model
	    model2 = self.Cauchy_model
	    t0 = self.PSPL_params[0]
	    dt = 10*self.PSPL_params[1]*self.PSPL_params[2]


		plt.figure()
		ax = plt.gca()
		peak = result2['t0']
		plt.title('Fitting Cauchy and PSPL Functions to'+str(self._filename), size=25)
		plt.xlabel('$t - 2458234 $'+' (days)',size=29)
		plt.ylabel('Magnification',size=29)
		plt.tick_params(axis='y',labelsize=20)
		plt.tick_params(axis='x',labelsize=20)
		plt.plot(xdata, ydata, '.', color='#1f78b4', markersize=15, alpha=0.8)
		plt.plot(xdata, model1, '-',color='#e31a1c', linewidth=3, label = 'PSPL Fit')
		plt.plot(xdata, model2,'-', color='#ff7f00', linewidth=3, label = 'Cauchy Fit')

		plt.xlim(t0-dt, t0+dt)
		plt.legend(loc = 'upper right' ,prop={'size': 25})

		fig = plt.gcf()
		fig.set_size_inches(12.0,10.0)




