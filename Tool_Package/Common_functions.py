# -*- coding: utf-8 -*-

import glob,os,sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import *
import re
from tqdm import tqdm
import scipy.stats as st
from os.path import expanduser
import cmath
import scipy.optimize as op
import time
import gzip
from scipy.interpolate import interp1d
import pandas as pd
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset,inset_axes,InsetPosition) 
import traceback
import scipy.special as sp





def localize_event(lightcurve,t0,tE):
    """Function to estimate roughly the area around the peak of an event, 
    and identify which timestamps in the array belong to the event versus
    the baseline
    """
    
    idx1 = np.where(lightcurve >= t0-tE)[0]
    idx2 = np.where(lightcurve <= t0+tE)[0]
    event = list(set(idx1).intersection(set(idx2)))
    
    baseline = np.arange(0,len(lightcurve),1)
    baseline = np.delete(baseline,event)
    
    it0 = np.where(lightcurve == t0)[0][0]
    
    #print min(lightcurve)
    #print it0
    return baseline, event, it0

def prepare(t,m,err):
    
    df = pd.DataFrame({'t': t, 'magnitude': m, 'm_err': err})
    peaks = np.array([t[np.argmin(m)]])
    baseline, event, it0 = localize_event(df['t'], peaks[0],50)
    
    base_mag = np.median(df['magnitude'][baseline])
    df['A'] = 10 ** (0.4*(base_mag - df['magnitude']))
    
    interpol = interp1d(df['t'],df['A'], kind='cubic')
    dt = np.abs(df['t'][np.argmin(np.abs(interpol(df['t'])-1.06))]-peaks[0])
    #print dt
    
    if dt==0.0:
        dt = 50


        
    
    #dt = 50
    # baseline, event, it0 = localize_event(df['t'], peaks[0],dt)


    A_max = 10 ** (0.4*(base_mag - (df['magnitude']-df['m_err'])))
    A_min = 10 ** (0.4*(base_mag - (df['magnitude']+df['m_err'])))
    df['A_err'] = (A_max - A_min)/2
    

    while (np.abs((df['t'][event]).diff())).max() > 0.1:
        
        if dt>20:
            dt = dt - 10
            baseline, event, it0 = localize_event(df['t'], peaks[0],dt)
        else:
            break
    #print dt    
    return df,baseline, event, it0, dt




def empty(df):
    return len(df.index) == 0

def fun (t,t0,tE, u0, fs):
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = ((u**2)+2)/(u*np.sqrt(u**2+4))
    F = fs*A +(1-fs)
    return F
        
def fun2 (t, mean, sigma,amp, t0,tE, u0, fs):
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = (((amp/np.sqrt(2*pi*(sigma**2)))*np.exp(-((t-mean)**2)/(2*(sigma**2)))))+((u**2)+2)/(u*np.sqrt((u**2)+4))
    F = fs*A +(1-fs)
    return F

def lnlike(theta, t, f, f_err):
    t0, tE, u0, fs = theta
    model = fun(t, t0, tE, u0, fs)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def lnlike2(theta, t, f, f_err):
    mean, sigma,amp, t0, tE, u0, fs = theta
    model = fun2(t,mean, sigma,amp, t0, tE, u0, fs)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def max_finder (t, A, f_err, it0, dt, event):
    
    t_event = t
    A_event = A
    f_err_event = f_err
    
    A_max = max(A_event[A_event <np.percentile(A_event,[0,100] )[1]]) #1.0/(float(f_s_true)/(max(df['f'])-1+float(f_s_true)))
    u0_true = np.sqrt( ( (1+np.sqrt(1+16*(A_max**2)))/(2* A_max) ) - 2 )
    t0_true =  t_event[np.argmax(A_event[A_event <np.percentile(A_event,[0,100] )[1]])]
    
    return t0_true

def planetary_fit (t, A, f_err, it0, dt, event):

	t_event = t
	A_event = A
	f_err_event = f_err

	A_max = max(A_event[A_event <np.percentile(A_event,[0,100] )[1]]) #1.0/(float(f_s_true)/(max(df['f'])-1+float(f_s_true)))
	u0_true = np.sqrt( ( (1+np.sqrt(1+16*(A_max**2)))/(2* A_max) ) - 2 )
	t0_true =  t_event[np.argmax(A_event[A_event <np.percentile(A_event,[0,100] )[1]])] #it0 #t[A.argmax()] #float(t0_theo)
	#ind1, ind2 = fwhm(A,A.argmax(),1)
	#tE_true = t[ind2]-t[ind1]
	tE_true = dt #[tE_finder (t,A),t[ind2]-t[ind1]]
	#print 'tE_true = '+ str( tE_true)


	if t0_true> max(t) or t0_true<min(t):
	    
	    t0_true
	tE_ = [[],[]]  

	#for i in tE_true:

	nll = lambda *args: -lnlike(*args)
	result = op.minimize(nll, [t0_true, u0_true, tE_true], args=(t_event,A_event, f_err_event),method = 'Nelder-Mead')
	t0_ml, u0_ml, tE_ml = result['x']
	tE_[0].append(lnlike([t0_ml, u0_ml, tE_ml],t_event,A_event, f_err_event))
	tE_[1].append([t0_ml, u0_ml, tE_ml])
	    
	#if t0_ml > max(t_event) or t0_ml < min(t_event):

	            
	#print tE_[1]
	mm = np.asarray( tE_[0] ) #tE_[0])
	tE__ = tE_[1] #[mm.argmax()]

	#print tE__

	t0_ml, u0_ml, tE_ml = tE__[0][0],tE__[0][1],tE__[0][2]

	f_ris = A_event - fun(t_event, t0_ml, u0_ml, tE_ml)

	duration = [0.01,0.1,1]  

	f_res = f_ris
	f_res = smooth(f_ris,10)

	t_ = np.asarray(t_event.values)


	f_ris__ = [f_res.max(),f_res.min()]
	t_ris__ = [t_[f_res.argmax()],t_[f_res.argmin()]]


	min_model_ = [[],[]]

	for sigma in duration:

	    for a in range(0,2):        
	        amp_ = f_ris__[a]
	            #print amp_
	        t_mean_ = t_ris__[a]


	        amp_ = amp_ * sigma * np.sqrt(2*pi)

	        nll = lambda *args: -lnlike2(*args)
	        result = op.minimize(nll, [t_mean_,sigma,amp_,t0_ml, u0_ml, tE_ml], args=(t_event, A_event, f_err_event),method = 'Nelder-Mead')
	        mean_mll, sigma_mll,amp_mll,t0_mll, u0_mll, tE_mll = result['x']
	            #print result['x']
	        min_model_[0].append(lnlike2([mean_mll, sigma_mll,amp_mll,t0_mll, u0_mll, tE_mll],t_event, A_event, f_err_event))
	        min_model_[1].append([mean_mll, sigma_mll,amp_mll,t0_mll, u0_mll, tE_mll])


	mmm_ = np.asarray( min_model_[0])
	final_param = min_model_[1][mmm_.argmax()] 


	chi_tot_1 = lnlike([t0_ml, u0_ml, tE_ml],t,A, f_err)
	chi_tot_2 = lnlike2([mean_mll, sigma_mll,amp_mll,t0_mll, u0_mll, tE_mll],t, A, f_err)

	inv_sigma2 = 1.0/(f_err**2)
	chi_base = np.sum((A-np.ones(len(A)))**2*inv_sigma2)

	return A, tE__, final_param, chi_tot_1, chi_tot_2, chi_base


def cal_chisqr(model, f, ferr):
    
    
    return np.sum(((f-model)**2)/((ferr)**2))

def F_t (t, t0, t_eff, f_1, f_0):

	Q = 1 + ((t-t0)/t_eff)**2

	F = f_1 *(Q**(-1.0/2) + (1 - (1 + Q/2)**-2)**(-1.0/2)) +f_0

	return F

def Gould_2_par_PSPL (t, m, t0, t_eff, f1, f0):
    
    t0_ini = t0
    t_eff_ini = t_eff
    
    paramt = [t0_ini, t_eff_ini, f1, f0]
    
    popt, pcov = scipy.optimize.curve_fit(F_t, t, m, p0=paramt)
    
    
    
    return popt




def cal_chisqr_modes(lightcurve,fx,ftype = 'm'):
    """Function to calculate the chi squared of the fit of the lightcurve
    data to the function provided"""
    if ftype=='m':
        chisq = ((lightcurve[:,1] - fx)**2 / fx).sum()
        
    if ftype=='A':
        chisq = ((lightcurve[:,3] - fx)**2 / fx).sum()
    
    return chisq
    
def bell_curve(x,c,a,b,d):
    """Function describing a bell curve of the form:
    f(x; a,b,c,d) = d / [1 + |(x-c)/a|^(2b)]
    
    Inputs:
    :param  np.array x: Series of intervals at which the function should
                        be evaluated
    :param float a,b,c: Coefficients of the bell curve function
    
    Returns:
    :param np.array f(x): Series of function values at the intervals in x
    """
    
    fx = d / ( 1.0 + (abs( (x-c)/a ))**(2.0*b) )
    
    return fx+1

def bell_curve_data(params, t, A_data):
    """Function describing a bell curve of the form:
    f(x; a,b,c,d) = d / [1 + |(x-c)/a|^(2b)]
    
    Inputs:
    :param  np.array x: Series of intervals at which the function should
                        be evaluated
    :param float a,b,c: Coefficients of the bell curve function
    
    Returns:
    :param np.array f(x): Series of function values at the intervals in x
    """
    c = params['t0'].value
    a = params['tE'].value
    b = params['b'].value
    d = params['amp'].value
    
    fx = (d / ( 1.0 + (abs( (t-c)/a ))**(2.0*b) ))
    
    return fx+1 - A_data

# def gaussian(x,a,b,c):
#     """Function describing a Gaussian of the form:
#     f(x; a,b,c) = a * exp(-(x-b)**2/2c*2)
    
#     Inputs:
#     :param  np.array x: Series of intervals at which the function should
#                         be evaluated
#     :param float a,b,c: Coefficients of the bell curve function
    
#     Returns:
#     :param np.array f(x): Series of function values at the intervals in x
#     """
    
#     fx = a * np.exp(-( (x-b)**2 / (2*c*c) ))
    
#     return fx

def PSPL_data (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    fs = params['fs'].value
#     fb = params['fb'].value

    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = ((u**2)+2)/(u*np.sqrt(u**2+4))
    F = (fs * (A-1)) +1
    return F - A_data

def PSPL (t0, tE, u0,fs, t):
    
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    A = ((u**2)+2)/(u*np.sqrt(u**2+4))
    F = (fs * (A-1)) +1
    return F

# def localize_event(lightcurve,t0, dt=10):
#     """Function to estimate roughly the area around the peak of an event, 
#     and identify which timestamps in the array belong to the event versus
#     the baseline
#     """
    
#     idx1 = np.where(lightcurve[:,0] >= t0-dt)[0]
#     idx2 = np.where(lightcurve[:,0] <= t0+dt)[0]
#     event = list(set(idx1).intersection(set(idx2)))
    
#     baseline = np.arange(0,len(lightcurve),1)
#     baseline = np.delete(baseline,event)
    
#     it0 = np.where(lightcurve[:,0] == t0)[0][0]
    
#     return baseline, event, it0
    
def estimate_init_param(lightcurve, t0, model, baseline,event,it0):
    """Function to provide rough estimates of the model parameters
    prior to fitting.
    
    Model parameter indicates which order in the coefficients array
    should be used, appropriate to model = { gaussian, bellcurve }
    options.
    """
    
    a = np.median(lightcurve[baseline,1]) - lightcurve[it0,1]

    half_mag = np.median(lightcurve[baseline,1]) - (a/1.2)
    dm = abs(lightcurve[:,1] - half_mag)
    idx = dm.argsort()
#     base = np.round(np.median(lightcurve[baseline,1]),4)
#     mag_peak = min(lightcurve[:,1][event])
#     A_max = 10 ** (0.4*(base - mag_peak))
    u0_ini = np.sqrt( ( (1+np.sqrt(1+16*(max(lightcurve[:,3])**2)))/(2* max(lightcurve[:,3])) ) - 2 )
    #print u0_ini
    
    
    dt = abs(lightcurve[idx[0],0] - lightcurve[idx[1],0])
    
    if model == 'gaussian':
        g_init_par = [ a, 0.0, dt ]
    
    elif model == 'bellcurve':
        g_init_par = [ dt, 1.0, 0.0, a]
        
    elif model == 'PSPL':
        g_init_par = [ 0.0, u0_ini, dt]
      
        
    else:
        raise RuntimeError('Unrecognised model type '+model)
        
    return g_init_par



# Smoothing the data

def low_pass_filter(y, box_pts, mode='same', base=1):
    box = base*(np.ones(box_pts)/box_pts)
    y_filtered = np.convolve(y, box, mode=mode)
    if mode=='same':
        y_filtered[0:int(box_pts/2)]=y_filtered[int(box_pts/2)]
        y_filtered[len(y_filtered)-int(box_pts/2):len(y_filtered)]=y_filtered[len(y_filtered)-int(box_pts/2)]
    return y_filtered

def count_peaks (t, m, smooth='yes', bin_size = 30, threshold = 3):
    
    if smooth == 'yes':
        m = low_pass_filter(m,8)
    else:
        pass
    df_ = pd.DataFrame({'t':t, 'm':m})

    bins = np.linspace(df_['t'].min(),df_['t'].max(),int((df_['t'].max()-df_['t'].min())/bin_size))
    # print bins
    groups = df_.groupby(np.digitize(df_['t'], bins))
    
    std_ = np.std(m)
    delta_m = []
    t__ = []
    c=0
    for i in groups.indices:
        #print c
        c = c+1
        #print i
        m_ = df_['m'][groups.indices[i]]
        t_ = df_['t'][groups.indices[i]]
        # std_ = np.std(df_['m'][groups.indices[i]])
        #print t,m
        del_m = np.asarray((np.abs(m_- m_.mean())/std_))
        delta_m.append(del_m)
        t__.append(np.asarray(t_))
    peaks = []    
    n_outliers = []
    for j in range(len(delta_m)):
        n_temp = len(np.where(delta_m[j]>threshold)[0])
        n_outliers.append(n_temp) 
        if n_temp > 5:
            peaks.append(t__[j][np.argmax(delta_m[j])])
    return n_outliers, peaks

def Double_horn_data (params, t, A_data):
    
    xe = params['xe'].value
    xp = params['xp'].value
    b1 = params['b1'].value
    b2 = params['b2'].value
    a = params['a'].value
    n = params['n'].value
    w = params['w'].value
    c = params['c'].value
    s = params['s'].value
    
    
    A = (a/4.)* (sp.erf(b1*(w+(s*t)-xe))+1) * (sp.erf(b2*(w-(s*t)+xe))+1) * (c*(np.abs((s*t)-xp)**n)+1)
                 
    return (A - A_data)

def Double_horn (xe,xp, b1,b2, a, n, w, c, s, t):
    
    A = (a/4.)* (sp.erf(b1*(w+(s*t)-xe))+1) * (sp.erf(b2*(w-(s*t)+xe))+1) * (c*(np.abs((s*t)-xp)**n)+1)

    return A


def Double_horn_PSPL_data (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    fs = params['fs'].value
    xe = params['xe'].value
    xp = params['xp'].value
    b1 = params['b1'].value
    b2 = params['b2'].value
    a = params['a'].value
    n = params['n'].value
    w = params['w'].value
    c = params['c'].value
    s = params['s'].value
    
    
    F = Double_horn (xe,xp, b1,b2, a, n, w, c, s, t)+PSPL (t0, tE, u0,fs, t)
                 
    return (F - A_data)

def Double_horn_PSPL (t0, tE, u0,fs, xe,xp, b1,b2, a, n, w, c, s, t):
    
    F = Double_horn (xe,xp, b1,b2, a, n, w, c, s, t)+PSPL (t0, tE, u0,fs, t)

    return F

def erfs (xe, b1,b2, a, w, s, t):
    A = (a/4.)* (sp.erf(b1*(w+(s*t)-xe))+1) * (sp.erf(b2*(w-(s*t)+xe))+1)
    
    return A

def Erfs_PSPL_data (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    fs = params['fs'].value
    xe = params['xe'].value
    xp = params['xp'].value
    b1 = params['b1'].value
    b2 = params['b2'].value
    a = params['a'].value
    n = params['n'].value
    w = params['w'].value
    c = params['c'].value
    s = params['s'].value
    
    
    F = erfs (xe, b1,b2, a, w, s, t)+PSPL (t0, tE, u0,fs, t)
                 
    return (F - A_data)

def Erfs_PSPL (t0, tE, u0,fs, xe, b1,b2, a, w, s, t):
    
    F = erfs (xe, b1,b2, a, w, s, t)+PSPL (t0, tE, u0,fs, t)

    return F
    

def lnlike_double_horn(theta, t, f, f_err):
    xe,xp, b1,b2, a, n, w, c, s= theta
    model = Double_horn (xe,xp, b1,b2, a, n, w, c, s, t)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def lnlike_double_horn_PSPL(theta, t, f, f_err):
    t0, tE, u0,fs, xe,xp, b1,b2, a, n, w, c, s, t = theta
    model = Double_horn_PSPL (t0, tE, u0,fs, xe,xp, b1,b2, a, n, w, c, s, t)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))

def lnlike_erfs(theta, t, f, f_err):
    xe, b1,b2, a, w, s = theta
    model = erfs (xe, b1,b2, a, w, s, t)
    inv_sigma2 = 1.0/(f_err**2)
    return -0.5*(np.sum((f-model)**2*inv_sigma2))
    



def PSPL_Gaussian (t0,tE, u0,fs,tp, tEp,amp, t):
#     u = np.sqrt(u0**2+((t-t0)/tE)**2)
    F = PSPL (t0, tE, u0,fs, t) + Gaussian (tp, tEp,amp, t)
    #= (((amp/np.sqrt(2*pi*(sigma**2)))*np.exp(-((t-mean)**2)/(2*(sigma**2)))))+((u**2)+2)/(u*np.sqrt((u**2)+4))
    #F = (fs * (A-1)) +1
    return F

def PSPL_Gaussian_data (params, t, A_data):
    
    t0 = params['t0'].value
    tE = params['tE'].value
    u0 = params['u0'].value
    tp = params['tp'].value
    tEp = params['tEp'].value
    amp = params['amp'].value
    fs = params['fs'].value
#     fb = params['fb'].value
    
    u = np.sqrt(u0**2+((t-t0)/tE)**2)
    #A = (((amp/np.sqrt(2*pi*(tEp**2)))*np.exp(-((t-tp)**2)/(2*(tEp**2)))))+((u**2)+2)/(u*np.sqrt((u**2)+4))
    #F = (fs * (A-1)) +1
    F = PSPL (t0, tE, u0,fs, t) + Gaussian (tp, tEp,amp, t)

    return F - A_data

def Gaussian (tp, tEp,amp, t):
    A = amp*np.exp(-1*((t-tp)**2)/(2*(tEp**2)))
    #(((amp/np.sqrt(2*pi*(sigma**2)))
#     F = (fs * (A-1)) +1
    return A

def Gaussian_data (params, t, A_data):
    

    tp = params['tp'].value
    tEp = params['tEp'].value
    amp = params['amp'].value
#     fs = params['fs'].value
#     fb = params['fb'].value
    
    A = amp*np.exp(-1*((t-tp)**2)/(2*(tEp**2)))
    #(((amp/np.sqrt(2*pi*(sigma**2)))

    return A- A_data
    
def trapezoid(x, a, b, tau1, tau2, tau3, tau4):
    # a and c are slopes
    #tau1 and tau2 mark the beginning and end of the flat top
#     y = np.zeros(len(x))
#     c = -np.abs(c)
#     a = np.abs(a)
#     #(tau1,tau2) = (min(tau1,tau2),max(tau1,tau2))
#     y[:int(tau1)] = base
#     y[int(tau1):int(tau2)] =  a*x[:int(tau1)] + b
#     y[int(tau2):int(tau3)] =  a*tau1 + b 
#     y[int(tau2):int(tau4)] = c*(x[int(tau2):]-tau2) + (a*tau1 + b)
#     y[int(tau4):] = base

    y = np.zeros(len(x))
    df_trap = pd.DataFrame({'x': x, 'y': y})
    
    c1 = np.abs((b-a)/(tau2-tau1))
    c2 = -1 * np.abs((a-b)/(tau4-tau3))
    
    df_trap['y'][df_trap['x']<tau1] = a
    df_trap['y'][(df_trap['x']>tau1) & (df_trap['x']<tau2)] =  c1*df_trap['x'][(df_trap['x']>tau1) & (df_trap['x']<tau2)] + (a- c1 * tau1)
    df_trap['y'][(df_trap['x']>tau2) & (df_trap['x']<tau3)] =  b
    df_trap['y'][(df_trap['x']>tau3) & (df_trap['x']<tau4)] = c2*df_trap['x'][(df_trap['x']>tau3) & (df_trap['x']<tau4)] + (a- c2 * tau4)
    df_trap['y'][df_trap['x']>tau4] = a

    return df_trap['y']
    
def med_med (true,fitted):
    temp = fitted - true
    return (np.median(np.abs(temp-np.median(temp))))

def deviation_finder (t, A_residual, PSPL_params,  binsize_initial = 600, threshold_default = 3):
    
    std_base = np.std(A_residual[(t > PSPL_params[0]+10*PSPL_params[1]) | (t < PSPL_params[0]-10*PSPL_params[1]) ])
    std_all = np.std(A_residual)
    percent_diff = (np.abs(std_base-std_all)/float(std_all))*100
    
    if percent_diff < 50:
        smoothie ='yes'
#         print 'Smooth'
    else:
        smoothie ='no'
#         print 'Do not smooth'
        
    b_s = binsize_initial
    
    n_out, peaks = count_peaks (t, A_residual, smooth=smoothie, bin_size =b_s, threshold = threshold_default)

    n_peaks = len(peaks)
    temp_peaks = peaks
#     print n_peaks
    
    c = 0
    
    if (c != 2) & (percent_diff > 5):
    
        while c != 2 :
            if b_s < 0.2:
#                 print 'No two peaks were found!'
                break

            b_s = b_s/2.
#             print b_s
            n_out2, temp_peaks = count_peaks (t, A_residual, smooth=smoothie, bin_size =b_s, threshold = 3)
            c = len(temp_peaks)

    if len(temp_peaks) == 2 :
        if np.abs(temp_peaks[0]-temp_peaks[1])<10:
            peaks = temp_peaks
            n_peaks = len(temp_peaks)
            # print '<100'

        # if (np.abs(temp_peaks[1]-temp_peaks[0]) <0.1) or (np.abs(temp_peaks[1]-temp_peaks[0]) > 10):
        #     b_s_ = [4,3,2, 1, 0.5]
        #     # print '<0.1 or >10'
        #     for b in b_s_:
                
        #         n_out, temp_peaks = count_peaks (t, A_residual, smooth='yes', bin_size =b, threshold = 3)

        #         if (len(temp_peaks) == 2):
        #             if (np.abs(temp_peaks[1]-temp_peaks[0]) >0.1) and (np.abs(temp_peaks[1]-temp_peaks[0]) < 10):
        #                 # print '>0.1 and <10'
        #                 peaks = temp_peaks
        #                 n_peaks = len(temp_peaks)

        if (np.abs(temp_peaks[1]-temp_peaks[0]) <0.2):
        
            peaks = [temp_peaks[0]]
            n_peaks = 1    
    
    return n_peaks, peaks

