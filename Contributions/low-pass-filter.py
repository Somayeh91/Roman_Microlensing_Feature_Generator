import numpy as np

def low_pass_filter(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_filtered = np.convolve(y, box, mode='same')
    return y_filtered