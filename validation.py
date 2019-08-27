'''
validate v-cut measurement on synthetic data
-> resolution factor measured via v-cut method
 ... transformed back to standard deviation <std> of Gaussian blur kernel
 ... relates to input <std> used to blur the sharp input image
'''

import numpy as np
import pylab as plt

# local
from generate import patVcut
from measure import measureVcut
from utils.transforms import resFactor2std
from resolutionFactor import resolutionFactor


def main():
    X = []  # input image sharpness (standard deviation)
    Y = []  # measured image sharpness
    Ys = []  # std for Y
    
    for std in np.linspace(0.5, 5, 30):  # for variable image sharpness expressed as standard deviation of gaussian blur kernel
        yi = []
        print(std)
        for _ in range(3):  # repeat every measurement 10 times

            # synth. measurement pattern:
            img_masked, img_unmasked, line = patVcut(std)
            # get image contrast along middle line:
            r, y, calc_angle = measureVcut(img_masked, line=line, img_unmasked=img_unmasked)[0]
            
            f_res = resolutionFactor(r, y, calc_angle)
            
            std2 = resFactor2std(f_res)
            yi.append(std2)
            
        if(len(yi)):
            X.append(std)
            Y.append(np.mean(yi))
            Ys.append(np.std(yi))

    print(X, Y)

    # plot
    Ys = np.array(Ys)
    Y = np.array(Y)

    # 1 to 1 relation
    plt.plot([0, max(X)], [0, max(X)], c='k')
    
    # +-standard deviation
    plt.fill_between(X, Y - Ys, Y + Ys, alpha=0.1, color='k')
    # average data
    plt.plot(X, Y)
    
    plt.xlabel("Given image sharpness (std)")
    plt.ylabel("Determined image sharpness (std)")

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

