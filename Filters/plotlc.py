import numpy as np
import matplotlib.pyplot as plt
import pdb
import glob

def get_data(filename):
    #quick function to get data from lc file
    data = np.genfromtxt(filename)
    return data


def plot_lc(data):
    #date is 0, magnitude is 1, error is 2
    
    #make figure
    fig,ax = plt.subplots()
    #generate plot
    ax.errorbar(data[:,0],data[:,1],data[:,2],linestyle='None')
    #setup axes
    ax.set_xlabel('Date')
    ax.set_ylabel('Mag')
    plt.gca().invert_yaxis()
    plt.show()

def plot_dlcdt(data):
    fig,ax = plt.subplots()
    dt = np.diff(data[:,0])
    dm = np.diff(data[:,1])
    ax.plot(data[:,0][:-1],dm/dt)
    #ax.plot(data[:,0],data[:,1],data[:,2],linestyle='None')
    ax.set_xlabel('Date')
    ax.set_ylabel(r'\frac{dMag}{dt}')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    filename = raw_input('filename:')
    pdb.set_trace()
    #data = get_data(filename)
    lcs = glob.glob('./lc/*W149*')
    for lc in lcs:
        data = get_data(lc)
        dt = np.diff(data[:,0])
        dm = np.diff(data[:,1])
        divs = dm/dt
        if np.max(divs) > 5:
        
            plot_lc(data)
            plot_dlcdt(data)
