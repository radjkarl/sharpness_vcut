# coding=utf-8
import numpy as np


def findXAt(xArr, yArr, yVal, index=0, s=0.0):
    """
    index: position of root (return index=0 by default)

    return all x values where y would be equal to given yVal
    if arrays are spline interpolated
    """
    from scipy import interpolate

    if xArr[1] < xArr[0]:
        # numbers must be in ascending order, otherwise method crashes...
        xArr = xArr[::-1]
        yArr = yArr[::-1]

    yArr = yArr - yVal
    if len(yArr) < 5:
        xn = np.linspace(xArr[0], xArr[-1], 5)
        yArr = np.interp(xn, xArr, yArr)
        xArr = xn
    f = interpolate.UnivariateSpline(xArr, yArr, s=s)
    return f.roots()[index]

#     except ValueError:
#         valid = np.isfinite(f(xArr))
#         xArr = xArr[valid]
#         yArr = yArr[valid]
#         f = interpolate.UnivariateSpline(xArr, yArr, s=s)
#         return f.roots()[index]
#     except IndexError:
#         #TODO: shouldnt ne necarrary
#         #sometimes fails... // that one is unclean, but ... well TODO
#         i = np.argmax(yArr<0)
#         x0 = xArr[i-1]
#         x1 = xArr[i]
#         return 0.5 * (x0 + x1)


if __name__ == '__main__':
    import sys
    import pylab as plt
    
    x = np.linspace(-3, 3, 100)
    y = x ** 2 - 5
    
    x0 = findXAt(x, y, 0)
    
    if 'no_window' not in sys.argv:
        
        plt.plot(x, y)
        plt.scatter(x0, 0)

        plt.grid(True)
        plt.show()
