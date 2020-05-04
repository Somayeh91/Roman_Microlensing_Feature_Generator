from Common_functions import *
import scipy.optimize as opt
import pandas as pd
import numpy as np




class Trapezoidal_fit(source):


	def __init__(self, filename):
		super().__init__(self, filename)
		

	def initial_guess_finder(self):

		self.df['magnitude_modified'] = np.median(self.df['magnitude'][self.baseline]) - self.df['magnitude']
			
		self.t0_ini = self.df['t'][np.argmax(self.df['magnitude_modified'][self.df['magnitude_modified'] <np.percentile(self.df['magnitude_modified'],[0,99.95] )[1]])]
		self.tE_ini = [10., 1., 0.1, 0.01] #0.5*(self.df['t'].values[1]-self.df['t'].values[0])*len(np.where(abs(self.df['magnitude_modified']-np.median(self.df['magnitude_modified']))>2*np.std(self.df['magnitude_modified']))[0])
		self.am_ini = np.max(self.df['magnitude_modified'][self.df['magnitude_modified'] <np.percentile(self.df['magnitude_modified'],[0,99.95] )[1]])

		a, b, tau1, tau2, tau3, tau4 = 0, self.am_ini, self.t0_ini-1*self.tE_ini, self.t0_ini-0.5*self.tE_ini, self.t0_ini+0.5*self.tE_ini, self.t0_ini+1*self.tE_ini

		self.initial_params = [a, b, tau1, tau2, tau3, tau4]

		# self.self.self.t0_ini2 = self.df['t'][np.argmax(self.df['magnitude_modified'][self.df['magnitude_modified'] <np.percentile(self.df['magnitude_modified'],[0,100] )[1]])]
		# self.self.tE_ini2 = 0.5
		# self.am_ini2 = np.max(self.df['magnitude_modified'][self.df['magnitude_modified'] <np.percentile(self.df['magnitude_modified'],[0,100] )[1]]) 

		# a, b, tau1, tau2, tau3, tau4 = 0, self.am_ini2, self.t0_ini2-1*self.tE_ini2, self.t0_ini2-0.5*self.tE_ini2, self.t0_ini2+0.5*self.tE_ini2, self.t0_ini2+1*self.tE_ini2

		# self.initial_params2 = [a, b, tau1, tau2, tau3, tau4]

	def fitter(self):

		# popt1, pcov1 = opt.curve_fit(trapezoid, self.df['t'], self.df['magnitude_modified'], p0=self.initial_params1)
		# popt2, pcov2 = opt.curve_fit(trapezoid, self.df['t'], self.df['magnitude_modified'], p0=self.initial_params2)

		# if cal_chisqr(self.df['magnitude_modified'],trapezoid(self.df['t'], *popt2),self.e) < calc_chisq2(self.df['magnitude_modified'],trapezoid(self.df['t'], *popt1),self.e):
				
	 #        self.popt = popt2
	 #        self.initial_params = self.initial_params2
	 #        self.second = 1
	 #    else:

	 #    	self.popt = popt1
	 #    	self.initial_params = self.initial_params1
	 #    	self.second = 0
		chisqr = []
		popt_final = []

		for tE_i in self.tE_ini:

			# optmizing
			popt, pcov = opt.curve_fit(trapezoid, self.df['t'], self.df['magnitude_modified'], p0=self.initial_params)
			
			chisqr.append(cal_chisqr(ydata,trapezoid(xdata, *popt),e))
			popt_final.append(popt)


			self.chisqr = min(chisqr)
			self.popt = popt_final[np.argmin(chisqr)]

	def feature_producer (self):

		self.t0 = self.df['t'][np.argmin(self.df['t']-((self.popt[5]-self.popt[2])/2))]
		self.tE_total = (self.popt[5]-self.popt[2])
		self.tE_flat_part = (self.popt[4]-self.popt[3])
		self.tE_ratio = tE_flat_part/tE_total
		self.tau1 = self.popt[2]
		self.tau2 = self.popt[3]
		self.tau3 = self.popt[4]
		self.tau4 = self.popt[5]
		self.normalized_baseline = self.popt[0] # Theoretically, this should be zero.
		self.normalized_maximum = self.popt[1] # This is maximum of the modified magnitude (median of the baseline is subtracted from it.)
		self.magnitude_median = np.median(self.df['magnitude'][self.baseline])
		self.ydata_magnitude = 


