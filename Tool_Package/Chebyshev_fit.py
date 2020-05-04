import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lc import lc
from scipy.interpolate import interp1d




class Chebyshev_fit(lc):
	"""
	Chebyshev_fit(lc)
	Given a function func, lower and upper limits of the interval [xmin,xmax],
	and maximum degree n, this class computes a Chebyshev approximation
	of the function.
	Method eval(x) yields the approximated function value.
	"""

	def __init__(self, filename, delta_t = 10):
		lc.__init__(self, filename, delta_t = 10)
		

	def Chebyhev_coefficients (self, degree):
		self.n = degree
		self.xmin = min(self.df['t'][self.event])
		self.xmax = max(self.df['t'][self.event])
		bma = 0.5 * (self.xmax - self.xmin)
		bpa = 0.5 * (self.xmax + self.xmin)
		interpoll = interp1d(self.df['t'],self.df['A'], kind='cubic')
		f = [interpoll(math.cos(math.pi * (k + 0.5) / self.n) * bma + bpa) for k in range(self.n)]
		fac = 2.0 / self.n
		self.cheby_coefficients = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / self.n) for k in range(self.n)]) for j in range(self.n)]




	def eval (self):

		self.Cheby_func = []

		for t_i in np.sort(self.df['t'][self.event].values):

			y = (2.0 * t_i - self.xmin - self.xmax) * (1.0 / (self.xmax - self.xmin))
			y2 = 2.0 * y
			(d, dd) = (self.cheby_coefficients[-1], 0)             # Special case first step for efficiency
			
			for cj in self.cheby_coefficients[-2:0:-1]:            # Clenshaw's recurrence
				(d, dd) = (y2 * d - dd + cj, d)
			self.Cheby_func.append(y * d - dd + 0.5 * self.cheby_coefficients[0])

		self.Cheby_func = np.asarray(self.Cheby_func)


	def feature_producer (self):
			
		self.Cheby_a0 = (self.cheby_coefficients[0])/(self.cheby_coefficients[0])
		self.Cheby_a2 = (self.cheby_coefficients[2])/(self.cheby_coefficients[0])
		self.Cheby_a4 = (self.cheby_coefficients[4])/(self.cheby_coefficients[0])
		self.Cheby_a6 = (self.cheby_coefficients[6])/(self.cheby_coefficients[0])
		self.Cheby_a8 = (self.cheby_coefficients[8])/(self.cheby_coefficients[0])
		self.Cheby_a10 = (self.cheby_coefficients[10])/(self.cheby_coefficients[0])

		n_ = np.linspace(0,(self.n/2.)-1,self.n/2., dtype=int)


		self.Cheby_cj_sqr = np.sum((np.asarray(self.cheby_coefficients)/(self.cheby_coefficients[0]))**2)
		self.Cheby_cj_sqr_odd = np.sum((np.asarray([self.cheby_coefficients[2*n+1] for n in n_])/(self.cheby_coefficients[0]))**2)
		self.Cheby_cj_sqr_even = np.sum((np.asarray([self.cheby_coefficients[2*n] for n in n_])/(self.cheby_coefficients[0]))**2)
		self.log10_Cheby_cj_sqr_minus_one = np.log10(self.Cheby_cj_sqr - 1)
		self.log10_Cheby_cj_sqr_even_minus_one = np.log10(self.Cheby_cj_sqr_even - 1)
		self.log10_Cheby_cj_sqr_odd_minus_one = np.log10(self.Cheby_cj_sqr_odd - 1)
		self.pos_log10_Cheby_cj_sqr_minus_one = -1*np.log10(self.Cheby_cj_sqr - 1)
		self.pos_log10_Cheby_cj_sqr_even_minus_one = -1*np.log10(self.Cheby_cj_sqr_even - 1)
		self.pos_log10_Cheby_cj_sqr_odd_minus_one = -1*np.log10(self.Cheby_cj_sqr_odd - 1)
	def plotter (self):

		if self.t.min() > 2458234:

			t_shift = -2458234
		else: 
			t_shift = 0

		xdata = np.sort(self.df['t'][self.event]) + t_shift
		ydata = self.df['A'][self.event]
		model = self.Cheby_func

		plt.plot(xdata, ydata,'b.', markersize=25, label='Lightcurve Data')
		plt.plot(xdata, model,'r-', linewidth=3, label='Chebyshev Fit')
		plt.ylabel('Magnification', size=25)
		plt.xlabel('Time - 2458520', size=25)
		plt.title('Chebyshev approximation of a Binary-lens Microlensing Lightcurve',size=25)
		plt.legend(loc=2,fontsize=20)

		plt.xticks(fontsize=20)
		plt.yticks(fontsize=20)


		fig = plt.gcf()
		fig.set_size_inches(15.0,12.0)





