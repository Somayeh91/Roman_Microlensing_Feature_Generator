# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:38:44 2019

@author: rstreet
"""

from os import getcwd, path
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import numpy as np
from scipy import stats
import finite_source_point_lens_feature as feature
from matplotlib import pyplot as plt

def test_finite_source_point_lens_feature():
    """Unittest to verify the feature to detect finite source effects in 
    an event lightcurve"""

    (lightcurve,peaks) = generate_microlensing_lightcurve(0.001, 400)

    f = feature.finite_source_point_lens_feature(lightcurve,peaks,
                                                       diagnostics=True)
    
    assert len(f) == len(peaks)
    
def generate_microlensing_lightcurve(u0, ndp, diagnostics=False):
    """Function to generate a microlensing lightcurve for testing purposes.
    
    WARNING: Full photometric uncertainties not implemented as not currently
    necessary for testing purposes.
    
    Inputs:
    :param float u0: Event impact parameter
    :param int ndp:  Number of datapoints in the lightcurve
    
    Returns:
    :param np.array lightcurve: Timeseries photmetry 3-column array
                                (timestamp, mag/flux, mag error/flux error)
                                for N datapoints
    """

    lightcurve = np.zeros((ndp,3))
    
    lightcurve[:,0] = np.linspace(2459000.0, 2459040.0, ndp)
    
    u = np.linspace(0.01, 10.0, int(float(ndp)/2))
    u = np.concatenate([u[::-1],u])

    lightcurve[:,1] = np.random.normal(16.0, 0.003, size=ndp)
    lightcurve[:,1] += -2.5 * np.log10(microlensing_magnification(u))
    
    peaks = np.zeros(1)
    peaks[0] = lightcurve[int(float(ndp)/2),0]
    
    lightcurve[:,2] = np.random.normal(lightcurve[:,1],0.001)

    if diagnostics:
        fig = plt.figure(1,(10,10))
        plt.plot(lightcurve[:,0],microlensing_magnification(u))
        plt.xlabel('HJD')
        plt.ylabel('Mag')
        plt.show()
    
    return lightcurve, peaks
    
def microlensing_magnification(u):
    """Function to calculate the microlensing magnification for a range of 
    angular projected separations between lens and source, u"""
    
    A = ((u*u) + 2) / (u * np.sqrt(u*u + 4))
    
    return A

def test_bell_curve(diagnostics=False):
    """Unittest to verify that the bell curve function produces the
    expected series of evaluated-function values.
    """
    
    x = np.linspace(-100.0, 100.0, 200)
    
    a = 2.0
    b = 4.0
    c = 6.0
    
    fx = feature.bell_curve(x,a,b,c)
    
    if diagnostics:
        fig = plt.figure(1,(10,10))
        plt.plot(x,fx,'k-')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.show()
        
    assert len(fx) == len(x)
    assert fx.max() < a
    
def test_gaussian(diagnostics=False):
    """Unittest to verify that the bell curve function produces the
    expected series of evaluated-function values.
    """
    
    x = np.linspace(-100.0, 100.0, 200)
    
    a = 2000.0
    b = 4.0
    c = 6.0
    
    fx = feature.gaussian(x,a,b,c)
    
    G = stats.norm(b, c)
    
    assert len(fx) == len(x)
    
    if diagnostics:
        fig = plt.figure(1,(10,10))
        plt.plot(x,fx,'k-')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.show()
        
if __name__ == '__main__':
    
    #test_bell_curve(diagnostics=True)
    #test_gaussian()
    #generate_microlensing_lightcurve(0.001, 400, diagnostics=True)
    test_finite_source_point_lens_feature()
    