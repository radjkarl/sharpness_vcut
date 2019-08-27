# local
from utils.findXAt import findXAt

_resolutionFactor_coeff = [1.98784598, 0.38044078, 0.91077515, 1.00170451]


def _resolutionFactor_corr(f, a, b, c, d):
    if f < 2.2:
        return (f ** a) * b
    return (f ** c) * d


def resolutionFactor(radii, contrasts, vcut_angle):
    '''
    current implementation of simplified image sharpness calculation expressed as 
    'resolution factor'  [see file #######.py]
    using 'v-cut' method [see file ####.py]
    '''
    r50 = findXAt(radii, contrasts, 0.5)  # distance from v-cut line intersection to half contrast
    fres0 = vcut_angle * r50
    return _resolutionFactor_corr(fres0, *_resolutionFactor_coeff)
