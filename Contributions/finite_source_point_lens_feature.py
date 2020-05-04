# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:20 2019

@author: rstreet
"""

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

def finite_source_point_lens_feature(lightcurve,peaks,diagnostics=False):
    """Function to calculate a feature estimating the degree to which 
    finite source effects may be impacting the shape of a microlensing 
    lightcurve.
    
    Feature(k) = delta chisq = chisq(gaussian,k) - chisq(bell curve,k)
    
    where k is the index of a peak in the lightcurve.  The peaks array provides
    the central timestamps of the peaks. 
    
    The chisq of a gaussian and a bell curve fit to each peak is calculated, 
    and the feature is represented as the delta chisq between the fits. 
    
    Where more than one peak is in the lightcurve, the feature returned is
    an array of the feature values for each peak, k.
    
    Inputs:
    :param np.array lightcurve: Timeseries photmetry 3-column array
                                (timestamp, mag/flux, mag error/flux error)
                                for N datapoints
    :param np.array peaks:      Timestamps of the centers of peaks in the 
                                lightcurves.
    :param bool diagnostics:    Switch for plotting output
    
    Returns:
    :param np.array feature:    Two feature values for k peaks : Feature1 = Gaussian_chisq - Bellcurve_chisq
                                                                 Feature2 = PSPL_chisq - Bellcurve_chisq
    """
    
    feature1 = np.zeros(len(peaks))
    feature2 = np.zeros(len(peaks))
    
    for k, t0 in enumerate(peaks):
        
        (baseline, event, it0) = localize_event(lightcurve,t0, 10)
        
        g_init_par = estimate_init_param(lightcurve, t0, 'gaussian',
                                         baseline, event, it0)

        
        try: 
            (gfit,gcov) = optimize.curve_fit(gaussian,lightcurve[:,0]-t0,
                                          lightcurve[:,1],
                                         p0=g_init_par)
        except RuntimeError:
            (baseline, event, it0) = localize_event(lightcurve, peaks[0],60)
            g_init_par = estimate_init_param(lightcurve, t0, 'gaussian',
                                         baseline, event, it0)
            (gfit,gcov) = optimize.curve_fit(gaussian,lightcurve[:,0]-t0,
                                          lightcurve[:,1],
                                         p0=g_init_par)
            #(gfit,gcov) = optimize.curve_fit(gaussian,lightcurve[event,0]-t0,
            #                             -lightcurve[event,1]+np.median(lightcurve[baseline,1]),
            #                             p0=g_init_par)
            
        gfx = gaussian(lightcurve[:,0]-t0,g_init_par[0],g_init_par[1],g_init_par[2])
        gfx = np.median(lightcurve[baseline,1]) - gfx
        
        gfx2 = gaussian(lightcurve[:,0]-t0,gfit[0],gfit[1],gfit[2])
        gfx2 = np.median(lightcurve[baseline,1]) - gfx2
        
        g_chisq = calc_chisq(lightcurve,gfx2)
        
        b_init_par = estimate_init_param(lightcurve, t0, 'bellcurve',
                                         baseline, event, it0)
        
        #(bfit,bcov) = optimize.curve_fit(bell_curve,lightcurve[event,0]-t0,
        #                                        -lightcurve[event,1]+np.median(lightcurve[baseline,1]),
        #                                        p0=b_init_par)
        
        try: 
            (bfit,bcov) = optimize.curve_fit(bell_curve,lightcurve[:,0]-t0,
                                                lightcurve[:,1],
                                                 p0=b_init_par)
            print b_init_par
            
        except RuntimeError:
            (baseline, event, it0) = localize_event(lightcurve, t0,60)
            b_init_par = estimate_init_param(lightcurve, t0, 'bellcurve', baseline, event, it0)
            (bfit,bcov) = optimize.curve_fit(bell_curve,lightcurve[:,0]-t0,
                                                lightcurve[:,1],
                                                 p0=b_init_par)
        
        bfx = bell_curve(lightcurve[:,0]-t0,bfit[0],bfit[1],bfit[2],bfit[3])
        bfx = np.median(lightcurve[baseline,1]) - bfx
        
        b_chisq = calc_chisq(lightcurve,bfx)
        
        feature1[k] = g_chisq - b_chisq

        PSPL_init_par = estimate_init_param(lightcurve, t0, 'PSPL',
                                         baseline, event, it0)

        #(PSPLfit,PSPLcov) = optimize.curve_fit(PSPL,lightcurve[event,0]-peaks[0],
        #                                        lightcurve[event,1],
        #                                         p0=PSPL_init_par)
        
        try: 
            (PSPLfit,PSPLcov) = optimize.curve_fit(PSPL,lightcurve[:,0]-t0,
                                                lightcurve[:,1],
                                                 p0=PSPL_init_par)
        except RuntimeError:
            (baseline, event, it0) = localize_event(lightcurve, t0,60)
            PSPL_init_par = estimate_init_param(lightcurve, t0, 'PSPL',
                                         baseline, event, it0)
            (PSPLfit,PSPLcov) = optimize.curve_fit(PSPL,lightcurve[:,0]-t0,
                                                lightcurve[:,1],
                                                 p0=PSPL_init_par)
        
        PSPLfx = PSPL(lightcurve[:,0]-t0,PSPLfit[0],PSPLfit[1],PSPLfit[2],PSPLfit[3])
        #PSPLfx = np.median(lightcurve[baseline,1]) - PSPLfx
        
        PSPL_chisq = calc_chisq(lightcurve,PSPLfx)

        
        if diagnostics:
            fig = plt.figure(1,(10,10))
            plt.plot(lightcurve[:,0],lightcurve[:,1],'m.')
            #plt.plot(lightcurve[:,0],gfx,'g-.')
            plt.plot(lightcurve[:,0],gfx2,'g-',label='Gaussian')
            plt.plot(lightcurve[:,0],bfx,'b-',label='Bellcurve')
            plt.plot(lightcurve[:,0],PSPLfx,'k-',label='PSPL')
            plt.xlabel('HJD')
            plt.ylabel('Mag')
            [xmin,xmax,ymin,ymax] = plt.axis()
            plt.axis([xmin,xmax,ymax,ymin])
            plt.legend()
            plt.show()
            print 'PSPL_chisq: '+str(PSPL_chisq)
            print 'Bellcurve_chisq: '+str(b_chisq)
            print 'Gaussian_chisq: '+str(g_chisq)
        
        feature2[k] = PSPL_chisq - b_chisq

    

    return feature1, feature2
    
def calc_chisq(lightcurve,fx):
    """Function to calculate the chi squared of the fit of the lightcurve
    data to the function provided"""
    
    chisq = ((lightcurve[:,1] - fx)**2 / fx).sum()
    
    return chisq
    
def bell_curve(x,a,b,c,d):
    """Function describing a bell curve of the form:
    f(x; a,b,c,d) = d / [1 + |(x-c)/a|^(2b)]
    
    Inputs:
    :param  np.array x: Series of intervals at which the function should
                        be evaluated
    :param float a,b,c,d: Coefficients of the bell curve function
    
    Returns:
    :param np.array f(x): Series of function values at the intervals in x
    """
    
    fx = d / ( 1.0 + (abs( (x-c)/a ))**(2.0*b) )
    
    return fx

def gaussian(x,a,b,c):
    """Function describing a Gaussian of the form:
    f(x; a,b,c) = a * exp(-(x-b)**2/2c*2)
    
    Inputs:
    :param  np.array x: Series of intervals at which the function should
                        be evaluated
    :param float a,b,c: Coefficients of the bell curve function
    
    Returns:
    :param np.array f(x): Series of function values at the intervals in x
    """
    
    fx = a * np.exp(-( (x-b)**2 / (2*c*c) ))
    
    return fx

def PSPL (x,t0,u0,tE,base):

    u = np.sqrt(u0**2+((x-t0)/tE)**2)
    A = ((u**2)+2)/(u*np.sqrt(u**2+4))
    fx = base*np.ones(len(x)) + (-2.5 * np.log10(A))

    """Function describing a point-source point-lens of the form:
    f(x; t0,u0,tE,base) = -2.5*log10(((u^2)+2)/(u*sqrt(4+u^2))) + base
        Where:
                u = sqrt(u0^2 + ((x-t0)/tE)^2)
    
    Inputs:
    :param  np.array x: Series of intervals at which the function should
                        be evaluated
    :param float t0,u0,tE,base: Coefficients of the point-source point-lens function
    
    Returns:
    :param np.array f(x): Series of function values at the intervals in x
    """


    return fx

def localize_event(lightcurve,t0,dt=10):
    """Function to estimate roughly the area around the peak of an event, 
    and identify which timestamps in the array belong to the event versus
    the baseline
    """
    
    idx1 = np.where(lightcurve[:,0] >= t0-dt)[0]
    idx2 = np.where(lightcurve[:,0] <= t0+dt)[0]
    event = list(set(idx1).intersection(set(idx2)))
    
    baseline = np.arange(0,len(lightcurve),1)
    baseline = np.delete(baseline,event)
    
    it0 = np.where(lightcurve[:,0] == t0)[0][0]
    
    return baseline, event, it0
    
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
    base = np.median(lightcurve[baseline,1])
    
    d_t = abs(lightcurve[idx[0],0] - lightcurve[idx[1],0])
    
    if model == 'gaussian':
        g_init_par = [ a, 0.0, d_t ]
    
    elif model == 'bellcurve':
        g_init_par = [ d_t, 1.0, 0.0, a]

    elif model == 'PSPL':
        g_init_par = [ 0.0, 0.1, d_t, base]
        
    else:
        raise RuntimeError('Unrecognised model type '+model)
        
    return g_init_par