import cv2
import numpy as np
import pylab as plt
from scipy.signal.signaltools import convolve2d
# local
from utils.line import resize
from utils.transforms import std2PSF


def randAngle_rad(low_deg=3, high_deg=6):
    '''
    random angle [radians] within <low_deg>...<high_deg> [degrees]
    '''
    return np.random.uniform(low=np.radians(low_deg),
                             high=np.radians(high_deg), size=(1,))[0]


def patVcut(std=None, psf=None, phi=None, angle_rad=None, size=501, SNR=30):
    '''
    psf       ... blur kernel (2d numpy array)
    phi       ... rotation of v-cut (phi=0 -> v-cut opening at 3:00)
    angle_rad ... v-cut opening angle 
    size      ... output image size (x,y) [px] 
    SNR       ... signal to noise ratio
    '''
    assert std is not None or psf is not None, "either std or pdf have to be provided"
    if psf is None:
        psf = std2PSF(std)  # obtain guassian blur kernel (=point spread function [psf] from standard deviation
    
    if angle_rad is None:
        angle_rad = randAngle_rad()
    if phi is None:
        phi = np.random.rand() * 2 * np.pi  # random rotation of v-cut angle
    
    c = size / 2
    rel_noise = 1.0 / SNR
    s = (size, size)
    s2 = size * size

    img_masked = np.zeros(s)

    rad = 0.5 * angle_rad
    # points of vCut:
    p0 = c + np.sin(rad + phi) * 2 * size
    p1 = c + np.cos(rad + phi) * 2 * size
    p2 = c + np.sin(-rad + phi) * 2 * size
    p3 = c + np.cos(-rad + phi) * 2 * size
    # v-cut positions (triangle):
    pts = np.array(((c, c),
                    (p0, p1),
                    (p2, p3)), dtype=int)
    # draw vCut:
    cv2.fillConvexPoly(img_masked, pts,
                       color=1,
                       lineType=0)

    # blur:
    img_masked = convolve2d(img_masked, psf, mode='same', boundary='symm')
    # add noise:
    img_masked += np.random.rand(s2).reshape(s) * rel_noise
    img_unmasked = np.ones(s) + np.random.rand(s2).reshape(s) * rel_noise
    # line indicating gap position:
    line = [c, c, c + np.sin(phi) * 0.5 * size, c + np.cos(phi) * 0.5 * size]
    line = resize(line, 1.2)
    return img_masked, img_unmasked, line


def patSiemensStar(s0, n=72, vhigh=255, vlow=0, antiasing=False):
    '''
    Siemens star pattern
    s0 ... image sie x,y [px]
    n ... number of black/white line pairs
    vlow/vhigh lower/upper image intensity
    antialising ... introduces additional blur - use carefully
    '''
    arr = np.full((s0, s0), vlow, dtype=np.uint8)
    c = int(round(s0 / 2.))
    s = 2 * np.pi / (2 * n)
    step = 0
    for i in range(2 * n):
        p0 = round(c + np.sin(step) * 2 * s0)
        p1 = round(c + np.cos(step) * 2 * s0)

        step += s

        p2 = round(c + np.sin(step) * 2 * s0)
        p3 = round(c + np.cos(step) * 2 * s0)

        pts = np.array(((c, c),
                        (p0, p1),
                        (p2, p3)), dtype=int)

        cv2.fillConvexPoly(arr, pts,
                           color=vhigh if i % 2 else vlow,
                           lineType=cv2.LINE_AA if antiasing else 0)
    arr[c, c] = 0

    return arr.astype(float)


def plotVcut(title, img_masked, img_unmasked, line):
    f, (a0, a1) = plt.subplots(2)
    f.canvas.set_window_title(title)
    a0.set_title('masked')
    a0.imshow(img_masked, vmin=0, vmax=1)
    x0, y0, x1, y1 = line
    a0.plot((x0, x1), (y0, y1))

    a1.set_title('unmasked')
    a1.imshow(img_unmasked, vmin=0, vmax=1)


if __name__ == '__main__':
    # generate and plot 4 synthetic v-cut images with different image sharpness
    for std in np.linspace(1.5, 4.5, 3):
        print(std)
        img_masked, img_unmasked, line = patVcut(std)
        plotVcut("vcut std=%.1f" % std, img_masked, img_unmasked, line)
    
    cv2.imwrite("masked.png", img_masked * 255)
    cv2.imwrite("unmasked.png", img_unmasked * 255)
    with open("line.txt", 'w') as f:
        f.write(','.join([str(int(l)) for l in line]))
    
    plt.figure("Siemens star")
    plt.imshow(patSiemensStar(640))
    plt.show()

