import numpy as np
import plotlc as plc
import matplotlib.pyplot as plt
import glob
import pdb

def interseason_stds(lc_data, n_var=2):
    """
    Computes the intraseason stds and compares them, identifies if one season
    is larger than the other

    n_var is the factor 
    """
    #determine the indices that separate the seasons
    season_bounds = get_season_bounds(lc_data)
    #set up an empty list for the stds
    mag_stds = []
    #for each season, go through and find the standard deviation of that season
    for i in range(len(season_bounds)):

        #set the bound to the two item list found in get_season_bounds
        bound = season_bounds[i]
        #print(season_bounds[i])

        #get the dates from the light curve of the season
        #dates = lc_data[:,0][bound[0]:bound[1]+1]
        #get the magnitudes from the light curve of the season
        mags = lc_data[:,1][bound[0]:bound[1]+1]
        #get the error in the magnitudes from the light curve of the season
        #errs = lc_data[:,2][bound[0]:bound[1]+1]

        #append the std of the mags to a running list
        mag_stds.append(np.std(mags))
    
    #just converts to an array
    mag_stds = np.array(mag_stds)
    #if the season with a maximum variance is larger than some factor of the average variance
    if mag_stds.max() > n_var*mag_stds.mean():
        #just plot it up for inspection
        print plc.plot_lc(lc_data)
    
    print mag_stds


def get_season_bounds(lc_data):
    """
    Find the indices for the beginning and ends of a season
    Probably not the most efficient, but does the trick
    """
    #find the indices for the dates of season beginnings and ends
    ##compute differences between the dates
    date_jumps = np.diff(lc_data[:,0])
    ##find where the jumps are larger than the average difference between days
    season_jumps = np.where(date_jumps>np.average(date_jumps))[0]
    ##for each index found above, we want two bounds, one at the start and one at the end of the season
    ##so, for each, we'll go through and 
    bounds_list = []
    for i in range(len(season_jumps)+1):
        #for the first season
        if i == 0:
            bounds_list.append([0,season_jumps[i]])
        elif i == (len(season_jumps)):
            bounds_list.append([season_jumps[i-1]+1,len(lc_data[:,0])-1])
        else:
            bounds_list.append([season_jumps[i-1]+1,season_jumps[i]])
    #pdb.set_trace()
    #for bound in bounds_list:
    #    print(lc_data[:,0][bound[1]] - lc_data[:,0][bound[0]])
    return bounds_list

        

    

if __name__ == '__main__':
    #get list of W149 lightcurves
    #pdb.set_trace()
    #gather a list of the W149 lightcurves from the data challenge
    lcs = glob.glob('./lc/*W149*')
    

    #pdb.set_trace()
    #for each of the lightcurves in the lc directory
    for lc in lcs:
        pdb.set_trace()
        #read in the lc, columns are 0:date, 1:magnitude, 2:error
        data = np.genfromtxt(lc)
        #see if there is a season with a large variance 
        interseason_stds(data)
    
    

