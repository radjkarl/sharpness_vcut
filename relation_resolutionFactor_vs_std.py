'''
Calculate the relationship between standard deviation of a Gaussian blur kernel [std]
and resolution factor [fres]
'''

import cv2
import pylab as plt
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import brent
from scipy.optimize.minpack import curve_fit

# local
from generate import patSiemensStar
from utils.transforms import _std2ResFactor_fitfn


def calc(pattern_fn, s0=1000, N=30):
    '''
    pattern_fn ... function(image_size) returning systhetic test image 
    s0         ... pattern size
    N          ... number of data points to be calculated  
    
    1. down-sample image (img0->img1)
        e.g.: fres = 2, image size=100x100
        -> down-sample to 50x50
    2. up-sample back to initial resolution (img1->img2)
    3. determine that standard deviation of Gaussian blur kernel 
        applied on img0 that causes smallest average absolute deviation (AAD) to img2
        
    returns [standard deviations], [resolution factors]
    '''
    # resolution factors
    fres0 = np.logspace(0.01, 1, N)  # 1-10
    fres1 = []  # same, corrected for integer image size
    stds = []
    img0 = pattern_fn(s0).astype(float)  # initial image

    for f in fres0:

        res = int(round(s0 / f))  # res resolution to resize to
        f = s0 / float(res)
        fres1.append(f)  # corrected scale factor

        # low qual:
            # smaller
        img1 = cv2.resize(img0, (res, res), interpolation=cv2.INTER_AREA)
            # bigger
        img2 = cv2.resize(img1, (s0, s0), interpolation=cv2.INTER_LINEAR)
 
        # determine std which gives most similar sharpness values
        # blured(img0) -> img2
        # use average-absolute-deviation(AAD) to calculate error:
        fn = lambda std: np.abs(img2 - (gaussian_filter(img0, abs(std))) 
                                ).mean() 

        std = abs(brent(fn))  # minimize algorithm
        stds.append(std)

    return np.array(stds), fres1


def plot(stds, fres):

    plt.xlabel('standard deviation [px]')
    plt.ylabel('resolution factor [-]')
    
    plt.plot(stds, fres, '--', linewidth=5, label='data')
    
    ind = np.argmax(stds >= 1)
    upper_stds = stds[ind:]
    lower_stds = stds[:ind ]

    ####fit 1 (linear, for std>1)
    xx = upper_stds[:, np.newaxis]
    m, _, _, _ = np.linalg.lstsq(xx, fres[ind:])  # y = a*x
    print('ascent for std>=1: ', m)
    plt.plot(upper_stds, m * upper_stds, label='''fit: y=m*x
        m=%.2f''' % m)
    
    ####fit 2 (log+lin, for std<1)
    # this fit can fail if not enough point are avail

    param, _perr = curve_fit(_std2ResFactor_fitfn, lower_stds, fres[:ind], (1, 1, 1, 1.5))
    print ('linlog fn params: ', param)
    plt.plot(lower_stds, _std2ResFactor_fitfn(lower_stds, *param), label='''fit: y=m*log(x*n)+x*o+p
        m=%.2f
        n=%.2f
        o=%.2f
        p=%.2f''' % (param[0], param[1], param[2], param[3]))
    
    plt.legend()     
    plt.show()


if __name__ == '__main__':
    stds, fres = calc(patSiemensStar)
    plot(stds, fres)
