# Functions include: deviation_finder, initial_guess_finder, PSPL_fitter_scipy, PSPL_plotter,
#                    PSPL_residual, Gaussian_fitter_lmfit, PSPL_Gaussian_fitter_lmfit 

from Common_functions import fun, fun2, lnlike, lnlike2, smooth, \
                             max_finder, localize_event, prepare, \
                             planetary_fit, count_peaks, PSPL, Gaussian_data,\
                             Gaussian, PSPL_Gaussian, PSPL_Gaussian_data, cal_chisqr,\
                             Double_horn_data, Double_horn, Double_horn_PSPL_data,\
                             Double_horn_PSPL, erfs, lnlike_double_horn, lnlike_double_horn_PSPL,\
                             lnlike_erfs, Erfs_PSPL_data, Erfs_PSPL


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
from lmfit import minimize, Parameters, Parameter, report_fit
import math
from sympy.solvers import solve
from sympy import Symbol
import sympy


home = os.path.expanduser("~")

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
        if np.abs(temp_peaks[0]-temp_peaks[1])<100:
            peaks = temp_peaks
            n_peaks = len(temp_peaks)
            # print '<100'

        if (np.abs(temp_peaks[1]-temp_peaks[0]) <0.1) or (np.abs(temp_peaks[1]-temp_peaks[0]) > 10):
            b_s_ = [4,3,2, 1, 0.5]
            # print '<0.1 or >10'
            for b in b_s_:
                
                n_out, temp_peaks = count_peaks (t, A_residual, smooth='yes', bin_size =b, threshold = 3)

                if (len(temp_peaks) == 2):
                    if (np.abs(temp_peaks[1]-temp_peaks[0]) >0.1) and (np.abs(temp_peaks[1]-temp_peaks[0]) < 10):
                        # print '>0.1 and <10'
                        peaks = temp_peaks
                        n_peaks = len(temp_peaks)    
    
    return n_peaks, peaks

def initial_guess_finder (t, A):

    # t = args['t']
    # A = args['A']

    if np.abs(np.mean(A) - np.median(A))< 0.01:
        up_lim = 99.4
    else:
        up_lim = 100

    A_max = A[ ( A[A < np.percentile( A, [0.0,up_lim] )[1]]).idxmax] 
    u0_true = np.sqrt( ( ( 1 + np.sqrt( 1 + 16 *( A_max ** 2 )))/( 2 * A_max ) ) - 2 )
    t0_true =  t[( A[A < np.percentile( A, [0.0,up_lim] )[1]]).idxmax] 

    if u0_true <0.5 :
        A_lim = 1.34
    else:
        A_lim = 1.06

    interpol = interp1d(t,A, kind='cubic')
    dt = np.abs(t[np.argmin(np.abs(interpol(t)-A_lim))]-t0_true)
    tE_true = dt

    return t0_true, tE_true, u0_true

def PSPL_fitter_scipy (t, A, A_err, initial_guesses):

    t0_true, tE_true, u0_true = initial_guesses[0], initial_guesses[1], initial_guesses[2]

    nll = lambda *args: -lnlike(*args)
    res_scipy = op.minimize(nll, [t0_true,tE_true, u0_true, 0.5], args=(t,A, A_err),method = 'Nelder-Mead')
    t0_ml, tE_ml, u0_ml,fs_ml = res_scipy['x']
    PSPL_params = [t0_ml, tE_ml, u0_ml,fs_ml]

    chisqr1 = cal_chisqr(PSPL(PSPL_params[0],PSPL_params[1],PSPL_params[2],PSPL_params[3], t), A, A_err)

    return PSPL_params, chisqr1

def PSPL_plotter (t, A, model_params):

    if t.min() > 2458234:
        t2 = t - 2458234
        t0 = model_params[0] - 2458234
    else: 
        t2 = t
        t0 = model_params[0]

    plt.plot(t2, A, '.', color='gray', markersize=20, label='Lightcurve Data')
    plt.plot(t2, PSPL(model_params[0],model_params[1],model_params[2],model_params[3], t),'g.', Label='PSPL Fit')
    plt.ylabel('Magnification', size=25)
    plt.xlabel('Time - 2458234', size=25)
    plt.xlim(t0 - 5 * model_params[1], t0 + 5 * model_params[1])
    plt.legend(loc=2 , fontsize=20)

    fig = plt.gcf()
    fig.set_size_inches(15.0,12.0)

def PSPL_residual (t, A, model_params):

    A_residual = A - PSPL(model_params[0],model_params[1],model_params[2],model_params[3], t)

    n_peaks, peaks = deviation_finder (t, A_residual , model_params)

    return A_residual, peaks

def Gaussian_fitter_lmfit (t, A, model_params):

    A_residual, peaks = PSPL_residual(t, A, model_params)

    duration = [0.01, 0.1, 1]
    temp = [[], []]


    for tEp in duration:

        params2 = Parameters()
        params2.add('tp', value= peaks[0], min=min(t), max=max(t))
        params2.add('tEp', value= tEp, min=0.001, max=10)
        params2.add('amp', value= A_residual[(np.abs(t-peaks[0])).idxmin],  min=-20, max=20)
        result2 = minimize(Gaussian_data, params2, args=(t, A_residual))
        temp[0].append(result2.params)
        temp[1].append(result2.chisqr)

    Gaussian_params = temp[0][np.argmin(temp[1])]['tp'].value, temp[0][np.argmin(temp[1])]['tEp'].value, \
                   temp[0][np.argmin(temp[1])]['amp'].value

    return Gaussian_params

def PSPL_Gaussian_fitter_lmfit (t, A, A_err, PSPL_params, Gaussian_params):

    params2 = Parameters()
    params2.add('t0', value= PSPL_params[0], min=min(t), max=max(t))
    params2.add('tE', value= PSPL_params[1], min=0.001, max=1000)
    params2.add('u0', value= PSPL_params[2], min=0, max=5)
    params2.add('fs', value= PSPL_params[3], min=0, max=1)
    params2.add('tp', value= Gaussian_params[0], min=Gaussian_params[0]-10, max=Gaussian_params[0]+10)
    params2.add('tEp', value= Gaussian_params[1], min=0, max=20)
    params2.add('amp', value= Gaussian_params[2], min=0.00001, max=100)
    result2 = minimize(PSPL_Gaussian_data, params2, args=(t, A))

    final_params = ([result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value, 
                     result2.params['fs'].value,
                      result2.params['tp'].value,result2.params['tEp'].value,result2.params['amp'].value])
    chisqr2 = cal_chisqr(PSPL_Gaussian(result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value, 
                                       result2.params['fs'].value, result2.params['tp'].value,
                                       result2.params['tEp'].value,result2.params['amp'].value, 
                                       t), A, A_err)

    return final_params, chisqr2



def calculate_s_q (t, A_residual, peaks, final_params,tp):

    t0, tE, u0, fs, xe,xp, b1,b2, a, n, w, c, s = final_params[0], final_params[1], final_params[2], final_params[3],\
                                                  final_params[4], final_params[5], final_params[6], final_params[7],\
                                                  final_params[8], final_params[9], final_params[10], final_params[11],\
                                                  final_params[12]

    model = Double_horn(xe,xp, b1,b2, a, n, w, c, s, t)

    if np.mean(model[model!= 0.0])>0:
        
        min_model = 0.0001
    else:
        min_model = -0.0001

    cc = 'None'

    if (c != 0):
        if (min_model > 0):
            max1 = t[(model[t>(np.median(t[model>min_model]))]).idxmax]
            max2 = t[(model[t<(np.median(t[model>min_model]))]).idxmax]
            tEp = (max(t[model > min_model]) - min(t[model > min_model]))/2
            cc = 'Major'
            
            tp_dd = max1+(max2-max1)/2
            t_new = t[ (t > tp-tEp-5) & (t < tp+tEp+5)]
            model_new = Double_horn(xe,xp, b1,b2, a, n, w, c, s, t_new-t0)
            residual_new = A_residual[ (t > tp-tEp-5) & (t < tp+tEp+5)]
            double_check =  np.abs(np.sum((residual_new-model_new)[(residual_new-model_new)<0])/np.sum((residual_new-model_new)[(residual_new-model_new)>0]))

        else:
            max1 = t[(model[t>(np.median(t[model<min_model]))]).idxmin]
            max2 = t[(model[t<(np.median(t[model<min_model]))]).idxmin]
            tEp = (max(t[model < min_model]) - min(t[model < min_model]))/2
            cc = 'Minor'

        t1 = max1
        t2 = max2



        u1 = np.sqrt( ((t1-t0)/tE)**2 + (u0)**2 )
        u2 = np.sqrt( ((t2-t0)/tE)**2 + (u0)**2 )

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

            q_final2 = (tEp/tE)**2
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
            tp = t[model.idxmax]-t0
            cc = 'Major'
            tEp = (max(t[model > min_model]) - min(t[model > min_model]))/2
            t_new = t [ (t > tp-10) & (t < tp+10)]
            residual_new = A_residual[ (t > tp-10) & (t < tp+10)]
            model_new = Double_horn(xe,xp, b1,b2, a, n, w, c, s, t_new-t0)
            double_check =  np.abs(np.sum((residual_new-model_new)[(residual_new-model_new)<0])/np.sum((residual_new-model_new)[(residual_new-model_new)>0]))
            if double_check > 1:
                tp = t[(model).idxmin]-t0
                cc = 'Minor'

        else:
            tp = t[(model).idxmin]-t0
            cc = 'Minor'
            tEp = (max(t[model < min_model]) - min(t[model < min_model]))/2
        
        u = np.sqrt( ((tp)/tE)**2 + (u0)**2 )
        s0 = Symbol('s0')
        s_final = (solve(s0-(1/s0)-u, s0))[1]

        if cc == 'Minor':
            s_final = 1./s_final

        q_final = (tEp/tE)**2


    return s_final, q_final , tEp

def double_horn_fitter_scipy (t, A, A_err, A_residual, peaks, model_params):

    #A_residual, peaks = PSPL_residual(t, A, model_params)

    if len(peaks) == 1:

        s_init = 5
        tp = peaks[0]
        a_init = max(A_residual)

        initials = [0.01*s_init,  1, 1, a_init, 1*s_init, s_init]
        nll = lambda *args: -lnlike_erfs(*args)
        result = op.minimize(nll, initials, args=(t-tp,A_residual, A_err),method = 'Nelder-Mead')
        xe, b1,b2, a, w, s = result['x']
        chisqr_double_horn = (-2*lnlike_erfs([xe, b1,b2, a, w, s],t-tp,A_residual, A_err))
        fp_double_horn = ([xe,0, b1,b2, a, 0, w, 0, s])

        chisqr = cal_chisqr(erfs(xe, b1, b2, a, w, s, t), A, A_err)

    if len(peaks) == 2:

        s_init = 5
        width = np.abs(peaks[1]-peaks[0])
        tp = min(peaks) + width/2.

        a_init = np.median(A_residual[(t>min(peaks)) & (t<max(peaks))])

        if a_init<0:
            a_init = max(A_residual)/100.
        else:
            pass

        initials = [0.01*s_init, 0.01*s_init, 1, 1, a_init, 10, (width/2.)*s_init,0.02, s_init]
        nll = lambda *args: -lnlike_double_horn(*args)
        result = op.minimize(nll, initials, args=(t-tp,A_residual, A_err),method = 'Nelder-Mead')
        xe, xp, b1, b2, a, n, w, c, s = result['x']
        chisqr_double_horn = (-2*lnlike_double_horn([xe,xp, b1,b2, a, n, w, c, s],t-tp,A_residual, A_err))
        fp_double_horn = ([xe, xp, b1, b2, a, n, w, c, s])

        chisqr = cal_chisqr(Double_horn(xe, xp, b1, b2, a, n, w, c, s, t), A, A_err)

    return fp_double_horn, tp, chisqr

def PSPL_double_horn_fitter_lmfit (t, A, A_err, PSPL_params, Double_horn_params, tp):

    dev_counter = 0

    if tp > PSPL_params[0]:
        xe_init = +1 * np.abs(tp-PSPL_params[0]) + (Double_horn_params[0]/float(Double_horn_params[8]))
        xp_init = +1 * np.abs(tp-PSPL_params[0]) + (Double_horn_params[1]/float(Double_horn_params[8]))
    else: 
        xe_init = -1 * np.abs(tp-PSPL_params[0]) + (Double_horn_params[0]/float(Double_horn_params[8]))
        xp_init = -1 * np.abs(tp-PSPL_params[0]) + (Double_horn_params[1]/float(Double_horn_params[8]))

    b1, b2, a, n, w, c, s = Double_horn_params[2], Double_horn_params[3], Double_horn_params[4],\
                            Double_horn_params[5], Double_horn_params[6], Double_horn_params[7],\
                            Double_horn_params[8]

    if c != 0 :

        params2 = Parameters()
        params2.add('t0', value= 0, min=-10, max=10)
        params2.add('tE', value= PSPL_params[1], min=PSPL_params[1]/2., max=PSPL_params[1]*2)
        params2.add('u0', value= PSPL_params[2], min=0, max=5)
        params2.add('fs', value= PSPL_params[3], min=PSPL_params[3]/1.1, max=PSPL_params[3]*1.1)
        params2.add('xe', value= xe_init*s, min=xe_init*s-10, max=xe_init*s+10)
        params2.add('xp', value= xp_init*s, min=xp_init*s-10, max=xp_init*s+10)
        params2.add('b1', value= b1,  min=b1/1.1, max=b1*1.1)
        params2.add('b2', value= b2 , min=b2/1.1, max=b2*1.1)
        params2.add('a', value= a, min=a/10., max=a*10.)
        params2.add('n', value= n,  min=n/1.1, max=n*1.1)
        params2.add('w', value= w, min=w/1.1, max=w*1.1)
        params2.add('c', value= c  , min=c/1.01, max=c*1.01)
        params2.add('s', value= s, min=s/1.5, max=s*1.5)
        result2 = minimize(Double_horn_PSPL_data, params2, args=(t-PSPL_params[0], A))
        final_params = ([result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value,result2.params['fs'].value,\
                                     result2.params['xe'].value,result2.params['xp'].value, result2.params['b1'].value,\
                                     result2.params['b2'].value, result2.params['a'].value,\
                                     result2.params['n'].value,result2.params['w'].value,result2.params['c'].value, result2.params['s'].value])
        chisqr = cal_chisqr(Double_horn_PSPL(result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value,result2.params['fs'].value,\
                                             result2.params['xe'].value,result2.params['xp'].value, result2.params['b1'].value,\
                                             result2.params['b2'].value, result2.params['a'].value, result2.params['n'].value,\
                                             result2.params['w'].value,result2.params['c'].value, result2.params['s'].value, t), A, A_err)
        dev_counter = 2
    else :

        params2 = Parameters()
        params2.add('t0', value= 0, min=-10, max=10)
        params2.add('tE', value= PSPL_params[1], min=PSPL_params[1]/2., max=PSPL_params[1]*2)
        params2.add('u0', value= PSPL_params[2], min=0, max=5)
        params2.add('fs', value= PSPL_params[3], min=PSPL_params[3]/1.1, max=PSPL_params[3]*1.1)
        params2.add('xe', value= xe_init*s, min=xe_init*s-10, max=xe_init*s+10)
        params2.add('xp', value= xp_init*s, min=xp_init*s-10, max=xp_init*s+10)
        params2.add('b1', value= b1,  min=b1/1.1, max=b1*1.1)
        params2.add('b2', value= b2 , min=b2/1.1, max=b2*1.1)
        params2.add('a', value= a, min=a/10., max=a*10.)
        params2.add('n', value= n,  min=1, max=10)
        params2.add('w', value= w, min=w/1.1, max=w*1.1)
        params2.add('c', value= c  , min=0.001, max=1)
        params2.add('s', value= s, min=s/1.5, max=s*1.5)
        result2 = minimize(Erfs_PSPL_data, params2, args=(t-PSPL_params[0], A))
        final_params = ([result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value,result2.params['fs'].value,\
                                     result2.params['xe'].value,0, result2.params['b1'].value,\
                                     result2.params['b2'].value, result2.params['a'].value,\
                                     0,result2.params['w'].value,0, result2.params['s'].value])
        chisqr = cal_chisqr(Erfs_PSPL(result2.params['t0'].value, result2.params['tE'].value, result2.params['u0'].value,result2.params['fs'].value,\
                                             result2.params['xe'].value, result2.params['b1'].value,\
                                             result2.params['b2'].value, result2.params['a'].value,\
                                             result2.params['w'].value, result2.params['s'].value, t), A, A_err)
        dev_counter = 1

    return final_params, dev_counter, chisqr


def planetary_fitter (path):

    t,m,e = np.loadtxt(path,unpack=True, usecols=(0,1, 2))

    if min(t)>2458234:
        t = t-2458234


    df, baseline, event, it0, dt = prepare(t,m,e)

    init = initial_guess_finder (df['t'], df['A'])

    PSPL_final_params, PSPL_chisqr = PSPL_fitter_scipy (df['t'], df['A'], df['A_err'], init)


    A_residual, peaks = PSPL_residual (df['t'], df['A'], PSPL_final_params)

    n_peaks, peaks = deviation_finder (df['t'], A_residual , PSPL_final_params)
        
    fp_double_horn, tp, double_horn_chisqr = double_horn_fitter_scipy (df['t'], df['A'], df['A_err'], A_residual, peaks, PSPL_final_params)

    final_params, dev_counter, chisqr_final = PSPL_double_horn_fitter_lmfit (df['t'], df['A'], df['A_err'], PSPL_final_params, fp_double_horn, tp)

    s_final, q_final, tEp = calculate_s_q (df['t']-PSPL_final_params[0], A_residual,peaks, final_params, tp)

    return df, PSPL_final_params, PSPL_chisqr, A_residual, peaks, fp_double_horn,\
             tp, double_horn_chisqr, final_params, dev_counter, chisqr_final,\
             s_final, q_final, tEp

