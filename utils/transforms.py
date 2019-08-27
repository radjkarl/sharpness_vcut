import numpy as np
from scipy.ndimage.filters import gaussian_filter
from utils.findXAt import findXAt

# <<<<<<<<<<<<<<<<<<<<<<<<<
# FOR THE CALCULATION OF THE FOLLOWING CONSTANTS SEE
# relation_resolutionFactor_vs_std.py
# fres<->std
_std2fres_bigger1 = 1.92675391
_std2fres_smaller1 = (1.11308561, 0.73214517, 0.78312647, 1.42350311)
# >>>>>>>>>>>>>>>>>>>>>>>>>


def _kSize(std, k, kSize):
    if kSize is None:
        try:
            kSize = int(k * std)
        except TypeError:
            kSize = int(k * max(std))
        kSize += 1 - kSize % 2  # make odd
        if kSize < 5:
            kSize = 5
    return kSize


def std2PSF(std, k=9, kSize=None):
    '''
    std[float, tuple] ... standard deviation - optionally different in x,y
    k[int] ... determine kSize through having k times std within
    kSize[float, tuple] ... kernel size in x and y
    '''
    # create point spread function (PSF) as 2d gaussian:
    kSize = _kSize(std, k, kSize)
    if type(kSize) not in (list, tuple):
        kSize = (kSize, kSize)
    # create nxn zeros
    inp = np.zeros(kSize)
    # set element at the middle to one, a dirac delta
    inp[kSize[0] // 2, kSize[0] // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    psf = gaussian_filter(inp, std, mode='constant')
    # ensure sum(psf) = 1 # can be different for small k values
    psf /= psf.sum()
    return psf


def _std2ResFactor_fitfn(x, m, n, o, p):
    return m * np.log(x * n) + x * o + p


def std2ResFactor(std):
    if std > 1:
        return _std2fres_bigger1 * std
#     a, b, c, d = _std2fres_smaller1
    return max(1,  # factor cannot be smaller than 1
               _std2ResFactor_fitfn(std, *_std2fres_smaller1))
#                a * np.log(std * b) + std * c - d)


def resFactor2std(fres):
    # TODO: invert std2ResFactor properly
    # lazy inverse
    if fres < 1:
        return 0.5
    
    xArr = np.linspace(0.5, 10, 1000)
    yArr = np.array([std2ResFactor(x) for x in xArr])
    return findXAt(xArr, yArr, fres)


if __name__ == '__main__':
    import pylab as plt
    
    for std in np.linspace(0.5, 3.5, 4):
        psf = std2PSF(std)
        plt.figure("psf for std=%.1f" % std)
        plt.imshow(psf, interpolation='none')
    plt.show()
    
