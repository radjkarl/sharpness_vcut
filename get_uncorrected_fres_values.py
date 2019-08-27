'''
in order to obtain empirical resolution factor coeff. calc. uncorrected resolution factor
for various input std
printed result can then be used in get_fres_coeffs.py

'''

import numpy as np
import pylab as plt

# local
from generate import patVcut
from measure import measureVcut

from utils.findXAt import findXAt


def main():
    X = []  # input image sharpness (standard deviation)
    Y = []  # uncorrected resolution factor
    Ys = []  # std for Y
    
    for std in np.linspace(0.5, 5, 40):  # for variable image sharpness expressed as standard deviation of gaussian blur kernel
        yi = []
        for _ in range(20):  # repeat every measurement 10 times

            # synth. measurement pattern:
            img_masked, img_unmasked, line = patVcut(std)
            # get image contrast along middle line:
            r, y, calc_angle = measureVcut(img_masked, line=line, img_unmasked=img_unmasked)[0]
            
            # uncorrected resolution factor:
            r50 = findXAt(r, y, 0.5) 
            yi.append(calc_angle * r50)
            
        if(len(yi)):
            X.append(std)
            ym = np.mean(yi)
            ys = np.std(yi)
    
            Ys.append(np.std(yi))
            Y.append(ym)

    print("stds=", X)  # use those values for resolution factor correction
    print('fres0=', Y)
    Y = np.array(Y)
    ys = np.array(ys)

    # 1 to 1 relation
    plt.plot([0, max(X)], [0, max(X)], c='k')
    
    # +-standard deviation
    plt.fill_between(X, Y - Ys, Y + Ys, alpha=0.1, color='k')
    # average data
    plt.plot(X, Y)
    
    plt.xlabel("Given image sharpness (std)")
    plt.ylabel("Uncorrected resolution factors")

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

