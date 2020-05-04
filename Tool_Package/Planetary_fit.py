from lc import lc
from Common_functions import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import scipy.stats as st
import scipy.optimize as op
import gzip
from scipy.interpolate import interp1d
import pandas as pd
from lmfit import minimize, Parameters, Parameter, report_fit
from sympy.solvers import solve
from sympy import Symbol
import sympy


class Planetary_fit(lc):

	def __init__(self, filename):
		lc.__init__(self, filename)

	# def initial_guess_finder (self):



	# 	if np.abs(np.mean(self.df['A']) - np.median(self.df['A']))< 0.01:
	# 		up_lim = 99.4
	# 	else:
	# 		up_lim = 100

	# 	A_max = self.df['A'][ ( self.df['A'][self.df['A'] < np.percentile( self.df['A'], [0.0,up_lim] )[1]]).idxmax] 
	# 	self.u0_true = np.sqrt( ( ( 1 + np.sqrt( 1 + 16 *( A_max ** 2 )))/( 2 * A_max ) ) - 2 )
	# 	self.t0_true =  self.df['t'][( self.df['A'][self.df['A'] < np.percentile( self.df['A'], [0.0,up_lim] )[1]]).idxmax] 

	def PSPL_fitter_scipy (self):


		nll = lambda *args: -lnlike(*args)
		res_scipy = op.minimize(nll, [self.t0_true,self.tE_true, self.u0_true, 0.5], args=(self.df['t'],self.df['A'], self.df['A_err']),method = 'Nelder-Mead')
		t0_ml, tE_ml, u0_ml,fs_ml = res_scipy['x']
		self.PSPL_params = [t0_ml, tE_ml, u0_ml,fs_ml]

		self.PSPL_chisqr = cal_chisqr(PSPL(self.PSPL_params[0],self.PSPL_params[1],self.PSPL_params[2],self.PSPL_params[3], self.df['t']), self.df['A'], self.df['A_err'])



	def PSPL_residual (self):

		self.df['A_residual'] = self.df['A'] - PSPL(self.PSPL_params[0],self.PSPL_params[1],self.PSPL_params[2],self.PSPL_params[3], self.df['t'])

		n_peaks, self.peaks = deviation_finder (self.df['t'], self.df['A_residual'] , self.PSPL_params)


	def double_horn_fitter_scipy (self):

		if len(self.peaks)==0 or len(self.peaks)>2:

			self.peaks = []
			self.peaks.append(self.df.t[(self.df['A_residual'].idxmax)])


		if len(self.peaks) == 1:

			s_init = 5
			self.tp = self.peaks[0]
			a_init = max(self.df['A_residual'])

			initials = [0.01*s_init,  1, 1, a_init, 1*s_init, s_init]
			nll = lambda *args: -lnlike_erfs(*args)
			result = op.minimize(nll, initials, args=(self.df['t']-self.tp,self.df['A_residual'], self.df['A_err']),method = 'Nelder-Mead')
			xe, b1,b2, a, w, s = result['x']
			chisqr_double_horn = (-2*lnlike_erfs([xe, b1,b2, a, w, s],self.df['t']-self.tp,self.df['A_residual'], self.df['A_err']))
			self.fp_double_horn = ([xe,0, b1,b2, a, 0, w, 0, s])

			self.dh_chisqr = cal_chisqr(erfs(xe, b1, b2, a, w, s, self.df['t']-self.tp), self.df['A_residual'], self.df['A_err'])

			self.dev_counter = 1

		if len(self.peaks) == 2:

			s_init = 5
			width = np.abs(self.peaks[1]-self.peaks[0])
			self.tp = min(self.peaks) + width/2.

			a_init = np.median(self.df['A_residual'][(self.df['t']>min(self.peaks)) & (self.df['t']<max(self.peaks))])

			if a_init<0:
				a_init = max(self.df['A_residual'])/100.
			else:
				pass

			initials = [0.01*s_init, 0.01*s_init, 1, 1, a_init, 10, (width/2.)*s_init,0.02, s_init]
			nll = lambda *args: -lnlike_double_horn(*args)
			result = op.minimize(nll, initials, args=(self.df['t']-self.tp,self.df['A_residual'], self.df['A_err']),method = 'Nelder-Mead')
			xe, xp, b1, b2, a, n, w, c, s = result['x']
			chisqr_double_horn = (-2*lnlike_double_horn([xe,xp, b1,b2, a, n, w, c, s], self.df['t']-self.tp,self.df['A_residual'], self.df['A_err']))
			self.fp_double_horn = ([xe, xp, b1, b2, a, n, w, c, s])

			self.dh_chisqr = cal_chisqr(Double_horn(xe, xp, b1, b2, a, n, w, c, s, self.df['t']-self.tp), self.df['A_residual'], self.df['A_err'])

			self.dev_counter = 2


	def PSPL_double_horn_fitter_lmfit (self):

		

		if self.tp > self.PSPL_params[0]:
			xe_init = +1 * np.abs(self.tp-self.PSPL_params[0]) + (self.fp_double_horn[0]/float(self.fp_double_horn[8]))
			xp_init = +1 * np.abs(self.tp-self.PSPL_params[0]) + (self.fp_double_horn[1]/float(self.fp_double_horn[8]))
		else: 
			xe_init = -1 * np.abs(self.tp-self.PSPL_params[0]) + (self.fp_double_horn[0]/float(self.fp_double_horn[8]))
			xp_init = -1 * np.abs(self.tp-self.PSPL_params[0]) + (self.fp_double_horn[1]/float(self.fp_double_horn[8]))

		b1, b2, a, n, w, c, s = self.fp_double_horn[2], self.fp_double_horn[3], self.fp_double_horn[4],\
								self.fp_double_horn[5], self.fp_double_horn[6], self.fp_double_horn[7],\
								self.fp_double_horn[8]

		if c != 0 :

			params2 = Parameters()
			params2.add('t0', value= 0, min=-10, max=10)
			params2.add('tE', value= self.PSPL_params[1], min=self.PSPL_params[1]/2., max=self.PSPL_params[1]*2)
			params2.add('u0', value= self.PSPL_params[2], min=0, max=5)
			params2.add('fs', value= self.PSPL_params[3], min=self.PSPL_params[3]/1.1, max=self.PSPL_params[3]*1.1)
			params2.add('xe', value= xe_init*s, min=xe_init*s-10, max=xe_init*s+10)
			params2.add('xp', value= xp_init*s, min=xp_init*s-10, max=xp_init*s+10)
			params2.add('b1', value= b1,  min=b1/1.1, max=b1*1.1)
			params2.add('b2', value= b2 , min=b2/1.1, max=b2*1.1)
			params2.add('a', value= a, min=a/10., max=a*10.)
			params2.add('n', value= n,  min=n/1.1, max=n*1.1)
			params2.add('w', value= w, min=w/1.1, max=w*1.1)
			params2.add('c', value= c  , min=c/1.01, max=c*1.01)
			params2.add('s', value= s, min=s/1.5, max=s*1.5)
			result2 = minimize(Double_horn_PSPL_data, params2, args=(self.df['t']-self.PSPL_params[0], self.df['A']))
			self.final_params = ([result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value,result2.params['fs'].value,\
										 result2.params['xe'].value,result2.params['xp'].value, result2.params['b1'].value,\
										 result2.params['b2'].value, result2.params['a'].value,\
										 result2.params['n'].value,result2.params['w'].value,result2.params['c'].value, result2.params['s'].value])
			self.final_chisqr = cal_chisqr(Double_horn_PSPL(result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value,result2.params['fs'].value,\
												 result2.params['xe'].value,result2.params['xp'].value, result2.params['b1'].value,\
												 result2.params['b2'].value, result2.params['a'].value, result2.params['n'].value,\
												 result2.params['w'].value,result2.params['c'].value, result2.params['s'].value, self.df['t']-self.PSPL_params[0]), self.df['A'], self.df['A_err'])
			
		else :

			params2 = Parameters()
			params2.add('t0', value= 0, min=-10, max=10)
			params2.add('tE', value= self.PSPL_params[1], min=self.PSPL_params[1]/2., max=self.PSPL_params[1]*2)
			params2.add('u0', value= self.PSPL_params[2], min=0, max=5)
			params2.add('fs', value= self.PSPL_params[3], min=self.PSPL_params[3]/1.1, max=self.PSPL_params[3]*1.1)
			params2.add('xe', value= xe_init*s, min=xe_init*s-10, max=xe_init*s+10)
			params2.add('xp', value= xp_init*s, min=xp_init*s-10, max=xp_init*s+10)
			params2.add('b1', value= b1,  min=b1/1.1, max=b1*1.1)
			params2.add('b2', value= b2 , min=b2/1.1, max=b2*1.1)
			params2.add('a', value= a, min=a/10., max=a*10.)
			params2.add('n', value= n,  min=1, max=10)
			params2.add('w', value= w, min=w/1.1, max=w*1.1)
			params2.add('c', value= c  , min=0.001, max=1)
			params2.add('s', value= s, min=s/1.5, max=s*1.5)
			result2 = minimize(Erfs_PSPL_data, params2, args=(self.df['t']-self.PSPL_params[0], self.df['A']))
			self.final_params = ([result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value,result2.params['fs'].value,\
										 result2.params['xe'].value,0, result2.params['b1'].value,\
										 result2.params['b2'].value, result2.params['a'].value,\
										 0,result2.params['w'].value,0, result2.params['s'].value])
			self.final_chisqr = cal_chisqr(Erfs_PSPL(result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value,result2.params['fs'].value,\
												 result2.params['xe'].value, result2.params['b1'].value,\
												 result2.params['b2'].value, result2.params['a'].value,\
												 result2.params['w'].value, result2.params['s'].value, self.df['t']-self.PSPL_params[0]), self.df['A'], self.df['A_err'])


	def calculate_s_q (self):


		t0, tE, u0, fs, xe,xp, b1,b2, a, n, w, c, s = self.final_params[0], self.final_params[1], self.final_params[2], self.final_params[3],\
													  self.final_params[4], self.final_params[5], self.final_params[6], self.final_params[7],\
													  self.final_params[8], self.final_params[9], self.final_params[10], self.final_params[11],\
													  self.final_params[12]

		model = Double_horn(xe,xp, b1,b2, a, n, w, c, s, self.df['t']-self.PSPL_params[0])

		if np.mean(model[model!= 0.0])>0:
			
			min_model = 0.0001
		else:
			min_model = -0.0001

		cc = 'None'

		if (c != 0):
			if (min_model > 0):
				max1 = self.df['t'][(model[self.df['t']>(np.median(self.df['t'][model>min_model]))]).idxmax]
				max2 = self.df['t'][(model[self.df['t']<(np.median(self.df['t'][model>min_model]))]).idxmax]
				self.tEp = (max(self.df['t'][model > min_model]) - min(self.df['t'][model > min_model]))/2
				cc = 'Major'
				
				tp_dd = max1+(max2-max1)/2
				t_new = self.df['t'][ (self.df['t'] > self.tp-self.tEp-5) & (self.df['t'] < self.tp+self.tEp+5)]
				model_new = Double_horn(xe,xp, b1,b2, a, n, w, c, s, t_new-self.PSPL_params[0])
				residual_new = self.df['A_residual'][ (self.df['t'] > self.tp-self.tEp-5) & (self.df['t'] < self.tp+self.tEp+5)]
				double_check =  np.abs(np.sum((residual_new-model_new)[(residual_new-model_new)<0])/np.sum((residual_new-model_new)[(residual_new-model_new)>0]))

			else:
				max1 = self.df['t'][(model[self.df['t']>(np.median(self.df['t'][model<min_model]))]).idxmin]
				max2 = self.df['t'][(model[self.df['t']<(np.median(self.df['t'][model<min_model]))]).idxmin]
				self.tEp = (max(self.df['t'][model < min_model]) - min(self.df['t'][model < min_model]))/2
				cc = 'Minor'

			t1 = max1
			t2 = max2



			u1 = np.sqrt( ((t1-self.PSPL_params[0])/tE)**2 + (u0)**2 )
			u2 = np.sqrt( ((t2-self.PSPL_params[0])/tE)**2 + (u0)**2 )

			s0s1 = np.sqrt( (u1)**2 - (u0)**2 )
			s0s2 = np.sqrt( (u2)**2 - (u0)**2 )
			s1s2 = s0s1 - s0s2

				
			if cc == 'Major':
				
				xs1 = (s1s2 * s0s1) / u1
				Lx = u1 - xs1
				
				s0 = Symbol('s0')
				s_final = (solve(s0-(1/s0)-Lx, s0))[1]
				if (xs1 > u1):
					q_final = ((max2 - max1)/tE)**2
				else:
					q_final = ( xs1*float(s_final)*np.sqrt(float(s_final**2 -1))/2. )**2

				q_final2 = (self.tEp/tE)**2
				q_final = (q_final2+q_final)/2
				
				if double_check > 1:
					cc = 'Minor'

					
			if cc == 'Minor':
				xs1 = s1s2/2.
				xs0 = s0s2 + xs1
				Lx = np.sqrt( (xs0 ** 2) + (u0 **2))
				
				s0 = Symbol('s0')
				s_final = (solve((1/s0)-s0-Lx, s0))[1]
				
				q_final = (16 * (s1s2**2))/((256./s_final**2)+27*s_final**6)
				


			if np.abs(max1-max2)<0.2:
				c = 0

			
		else:
			if (min_model > 0) :
				tp = self.df['t'][model.idxmax]
				cc = 'Major'
				self.tEp = (max(self.df['t'][model > min_model]) - min(self.df['t'][model > min_model]))/2
				t_new = self.df['t'] [ (self.df['t'] > tp-10) & (self.df['t'] < tp+10)]
				residual_new = self.df['A_residual'][ (self.df['t'] > tp-10) & (self.df['t'] < tp+10)]
				model_new = Double_horn(xe,xp, b1,b2, a, n, w, c, s, t_new-self.PSPL_params[0])
				double_check =  np.abs(np.sum((residual_new-model_new)[(residual_new-model_new)<0])/np.sum((residual_new-model_new)[(residual_new-model_new)>0]))
				if double_check > 1:
					# tp = self.df['t'][(model).idxmin]
					cc = 'Minor'

			else:
				tp = self.df['t'][(model).idxmin]
				cc = 'Minor'
				self.tEp = (max(self.df['t'][model < min_model]) - min(self.df['t'][model < min_model]))/2
			
			u = np.sqrt( ((tp-self.PSPL_params[0])/tE)**2 + (u0)**2 )
			s0 = Symbol('s0')
			s_final = (solve(s0-(1/s0)-u, s0))[1]

			if cc == 'Minor':
				s_final = 1./s_final

			q_final = (self.tEp/tE)**2

		self.s_estimated = s_final
		self.q_estimated = q_final

	def Plotter (self, model_name,xlim=1):

		

		if self.t.min() > 2458234:

			t_shift = -2458234
		else: 
			t_shift = 0

		if model_name == 'PSPL':


			ydata = self.df['A']
			model = PSPL(self.PSPL_params[0],self.PSPL_params[1],self.PSPL_params[2],self.PSPL_params[3], self.df['t'])
			xdata = self.df['t'] + t_shift
			t0 = self.PSPL_params[0] + t_shift
			dt = 10 * self.PSPL_params[1]
			labelname = 'PSPL'
			mainlabel = 'Lightcurve Data'

		elif model_name == 'PSPL_Plus_Busy':

			

			t0, tE, u0, fs, xe,xp, b1,b2, a, n, w, c, s = self.final_params[0], self.final_params[1], self.final_params[2], self.final_params[3],\
													  self.final_params[4], self.final_params[5], self.final_params[6], self.final_params[7],\
													  self.final_params[8], self.final_params[9], self.final_params[10], self.final_params[11],\
													  self.final_params[12]

			# t0 = self.final_params[0] + t_shift

			if self.dev_counter == 2 :
				model = Double_horn_PSPL(0, tE, u0, fs, xe,xp, b1,b2, a, n, w, c, s, self.df['t']-self.PSPL_params[0])
			else: 
				model = Erfs_PSPL(0, tE, u0, fs, xe, b1, b2, a, w, s, self.df['t']-self.PSPL_params[0])
				
			ydata = self.df['A']
			xdata = self.df['t'] - self.PSPL_params[0]
			t0 = 0
			dt = 10 * self.final_params[1]
			labelname = 'PSPL_Plus_Busy'
			mainlabel = 'Lightcurve Data'

		elif model_name == 'Busy_Fit_Residual':


			xe, xp, b1, b2, a, n, w, c, s = self.fp_double_horn[0],self.fp_double_horn[1], self.fp_double_horn[2],\
											self.fp_double_horn[3],self.fp_double_horn[4], self.fp_double_horn[5],\
											self.fp_double_horn[6],self.fp_double_horn[7], self.fp_double_horn[8]

			if self.dev_counter == 2 :
				model = Double_horn(xe, xp, b1, b2, a, n, w, c, s, self.df['t']-self.tp)

			else:
				model = erfs(xe, b1, b2, a, w, s, self.df['t']-self.tp)

			

			ydata = self.df['A_residual']
			xdata = self.df['t'] - self.tp
			t0 = 0
			dt = 5 * (w)
			labelname = 'Busy_Fit_Residual'	
			mainlabel = 'Residual'	







		plt.plot(xdata, ydata, '.', color='gray', markersize=20, label=mainlabel)
		plt.plot(xdata, model,'g-', Label= str(labelname) +' Fit')
		plt.ylabel('Magnification', size=25)
		plt.xlabel('Time - 2458234', size=25)
		plt.xlim(t0 - xlim*dt, t0 + xlim*dt)
		plt.legend(loc=2 , fontsize=20)

		fig = plt.gcf()
		fig.set_size_inches(15.0,12.0)

