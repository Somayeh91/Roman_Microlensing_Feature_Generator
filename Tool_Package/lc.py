import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import *
import pandas as pd
from scipy.interpolate import interp1d


class lc():
	'''Astronomical source object for NOAO formatted light curve
	
	Parameters
	----------
	filename : str
		path to space delimited file including each time, 
		filter, flux, and flux_uncertainty brightness 
		measurement on each line. A header line must be 
		included in the file with 't pb flux dflux'
	
	Attributes
	----------
	_filename : str
		the filename parameter
	
	_lc_df : pandas DataFrame
		light curve data read into a pandas dataframe
	
	_filters : arr-like
		numpy array storing each of the unique passbands 
		in the light curve file
	'''
	
	def __init__(self, filename, delta_t = 50):

		t,m,e = np.loadtxt(filename,unpack=True, usecols=(0,1,2))


		self.t = t
		self.m = m
		self.e = e
		self._filename = filename

	#def localize_event(self):
		
		self.t_max = self.t[np.argmin(self.m)]
		# delta_t = 50
		idx1 = np.where(self.t >= self.t_max-delta_t)[0]
		idx2 = np.where(self.t <= self.t_max+delta_t)[0]
		event = list(set(idx1).intersection(set(idx2)))
		
		baseline = np.arange(0,len(self.t),1)
		baseline = np.delete(baseline,event)
		
		it0 = np.where(self.t == self.t_max)[0][0]
		

		self.event = event
		self.baseline = baseline

		self.t_idx = it0
		
	#def pd_maker(self):

		# self.localize_event()

		df = pd.DataFrame({'t': self.t, 'magnitude': self.m, 'm_err': self.e})
		
		base_mag = np.median(df['magnitude'][self.baseline])
		df['A'] = 10 ** (0.4*(base_mag - df['magnitude']))
		

		A_max = 10 ** (0.4*(base_mag - (df['magnitude']-df['m_err'])))
		A_min = 10 ** (0.4*(base_mag - (df['magnitude']+df['m_err'])))
		df['A_err'] = (A_max - A_min)/2
		
		if np.abs(np.mean(self.df['A']) - np.median(self.df['A']))< 0.01:
			up_lim = 99.4
		else:
			up_lim = 100

		A_max = self.df['A'][ ( self.df['A'][self.df['A'] < np.percentile( self.df['A'], [0.0,up_lim] )[1]]).idxmax]
		self.u0_true = np.sqrt( ( ( 1 + np.sqrt( 1 + 16 *( A_max ** 2 )))/( 2 * A_max ) ) - 2 )
		self.t0_true =  self.df['t'][( self.df['A'][self.df['A'] < np.percentile( self.df['A'], [0.0,up_lim] )[1]]).idxmax] 


		if self.u0_true <0.5 :
			A_lim = 1.34
		else:
			A_lim = 1.06

		interpol = interp1d(self.df['t'],self.df['A'], kind='cubic')
		dt = np.abs(self.t[np.argmin(np.abs(interpol(self.df['t'])-A_lim))]-self.t0_true)
		self.tE_true = dt

		self.df = df
		# self.tE = dt
		self.interpol_A = interpol

		# Three-parameter PSPL Fit
		
		params = Parameters()
		params.add('t0', value= self.t0_true, min=min(df['t']), max=max(df['t']))
		params.add('tE', value= dt, min=0.001, max=1000)
		params.add('u0', value= self.u0_true, min=0, max=5)

		result = minimize(PSPL_data, params, args=(df['t'].values, df['A'].values))

		self.PSPL1_t0, self.PSPL1_tE, self.PSPL1_u0 = result.params['t0'],result.params['tE'],result.params['u0']
		self.PSPL1_chisqr = result.chisqr




	def plot(self):

		# self.pd_maker()

		'''Plot the 4 band light curve'''
		fig, axs = plt.subplots(figsize=(15, 12))

		if self.t.min() > 2458234:
				t2 = self.t - 2458234
		else: 
				t2 = self.t

		plt.plot(t2[self.event], self.df['A'][self.event], '.', color='gray', markersize=20)
		plt.ylabel('Magnification', size=25)
		plt.xlabel('Time - 2458234', size=25)
		# plt.xlim(self.t_idx - 2 * self.tE, self.t_idx + 2 * self.tE)
		# plt.legend(loc=2 , fontsize=20)

		fig.tight_layout()