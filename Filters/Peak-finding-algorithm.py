# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
# Description : If a new datapoint is a given x number of standard deviations (also called z-score) away from some moving mean, the algorithm signals. 
# Lag = The lag of the moving window
# Threshold = the z-score at which the algorithm signals (x number of standard deviations)
# Influence = the influence (between 0 and 1) of new signals on the mean and standard deviation --> And an influence of 0.5 gives signals half of the influence that normal datapoints have. 
# Likewise, an influence of 0 ignores signals completely for recalculating the new threshold. 
# An influence of 0 is therefore the most robust option (but assumes stationarity); putting the influence option at 1 is least robust. 
# For non-stationary data, the influence option should therefore be put somewhere between 0 and 1.


import numpy as np
import pylab

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

#define lag, threshold, influence. For example:
lag = 30
threshold = 6
influence = 0

# y is the magnitude or magnification

y = data                          
                                                
result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

# Plot result

pylab.subplot(211)
pylab.plot(np.arange(1, len(y)+1), y)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"], color="cyan", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

pylab.plot(np.arange(1, len(y)+1),
           result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

pylab.subplot(212)
pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
pylab.ylim(-1.5, 1.5)