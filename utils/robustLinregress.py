# coding=utf-8
from scipy.stats import linregress


def robustLinregress(x, y, n_iter=3, nstd=2):
    """
    do linear regression [n_iter] times
    successively removing outliers
    return result of normal linregress
    
    this method is faster and more stable than scipy.stats.theilslopes
    """
    for _ in range(n_iter):
        m, n = linregress(x, y)[:2]
        y_fit = x * m + n
        dy = y - y_fit
        std = (dy ** 2).mean() ** 0.5
        inliers = abs(dy) < nstd * std
        if inliers.sum() > 2:
            x = x[inliers]
            y = y[inliers]
        else:
            break
    return linregress(x, y)


if __name__ == '__main__':
    import numpy as np
    import pylab as plt
    import sys
    from time import time

    from scipy.stats import theilslopes

    n = 1000
    x = np.arange(n)
    yorig = np.linspace(0, 10, n)

    # add noise:
    y = yorig + ((np.random.rand(n) - 0.5) * 3)

    # add some outliers:
    pos = np.random.randint(0, 10, n) > 7
#     vals = np.random.rand(pos.sum()) * 3
    y[pos] *= np.random.rand((pos).sum()) * 3

    # ACTION
    t0 = time()
    m, n = linregress(x, y)[:2]
    y_fit1 = m * x + n
    print('time linregress: ', time() - t0)
    
    t0 = time()
    m, n = robustLinregress(x, y)[:2]
    y_fit2 = m * x + n
    print('time this: ', time() - t0)
    
    t0 = time()
    m, n = theilslopes(y, x, 0.90)[:2]
    y_fit3 = m * x + n
    print('time theilslopes: ', time() - t0)

    if 'no_window' not in sys.argv:
        # PLOT
        plt.plot(x, y, label='values')
        plt.plot(x, yorig, label='truth', linewidth=3)
        plt.plot(x, y_fit1, label='normal fit')
        plt.plot(x, y_fit2, label='THIS fit ignoring outliers', linewidth=3)
        plt.plot(x, y_fit3, label='theilslopes')

        plt.legend()
        plt.show()
