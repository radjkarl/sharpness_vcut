import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

# local
from utils.line import angle2, fromFn, intersection, cutToFitIntoPolygon
from utils.alignImageAlongLine import alignImageAlongLine
from utils.robustLinregress import robustLinregress
from utils.findXAt import findXAt


def measureVcut(img_masked, line, img_unmasked=None, img_bg=None,
             max_width=101,
             mask_is_dark=False,
             v_isDark=None):
    '''
    img_masked   ... image with v-cut mask
    img_unmasked ... image without v-cut mask
    img_bg - either average background level or background (or dark current) image
    line = (x0,y0,x1,y1) - line from a place within the gap untill a place behind
              the v-cut intersection 
            line points do not have to be precise
  
    v_isDark - whether v-cut mask is dark - if None, value is determined from image intensities at line start/end
    mask_is_dark - whether mask used to build V is absolutely dark (0) 
    '''
    
    if img_unmasked is not None:
        img_unmasked = img_unmasked.astype(float)
        
    img_masked = img_masked.astype(float)
   
    if img_bg is not None:
        img_unmasked = img_unmasked - img_bg
        img_masked = img_masked - img_bg
      
    s0, s1 = img_masked.shape
    poly = ((0, 0), (s1, 0), (s1, s0), (0, s0), (0, 0))
    line = cutToFitIntoPolygon(line, poly)
    # rectify image to given line: (this is not necassarily precise)
    if img_unmasked is not None:
        sub_img_unmasked = alignImageAlongLine(img_unmasked, line, max_width)
    sub_img_masked = alignImageAlongLine(img_masked, line, max_width)

    if not mask_is_dark:
        # assume mtf being 0 at given end value
        offs = sub_img_masked[-1].mean()
        sub_img_masked -= offs
        if img_unmasked is not None:
            sub_img_unmasked -= offs
    
    # create 0...1 scaled contrast image:
    if img_unmasked is None:
        contrast_img = sub_img_masked
    else:
        # make relative
        contrast_img = (sub_img_unmasked - sub_img_masked) / (sub_img_unmasked + sub_img_masked)

    if v_isDark is None:
        # determine whether vmask is dark from eval sub_img_masked intensities at x=0
        # if mid brighter than corner:
        if contrast_img[contrast_img.shape[0] // 2].mean() > contrast_img[0].mean():
            v_isDark = True

    if not v_isDark:
        contrast_img = 1 - contrast_img

    s0, s1 = contrast_img.shape
    x = np.arange(s1, dtype=int)

    dsub = cv2.Sobel(contrast_img, cv2.CV_64F, 0, 1, ksize=5)
  
    # FIND LINES: (precise)
    line1 = np.argmin(dsub, axis=0)
    line3 = np.argmax(dsub, axis=0)
    line2 = 0.5 * (line1 + line3)

    xx = x
    # find out where V ends:
    i = np.argmax(contrast_img[line2.round().astype(int), x] < 0.22)
    if i:
        s1 = i
        xx = x[:i]
        line1 = line1[:i]
        line2 = line2[:i]
        line3 = line3[:i]

    # FIT LINEAR LINES:
    m1, n1 = robustLinregress(xx, line1)[:2]  # upper edge
    m3, n3 = robustLinregress(xx, line3)[:2]  # lower edge

    l1 = fromFn(m1, n1)
    l3 = fromFn(m3, n3)

    # lines y-pos:
    fitline1 = x * m1 + n1  # REMOVE NOT NEEDED
    fitline3 = x * m3 + n3  # R..
    fitline2 = 0.5 * (fitline1 + fitline3)  # middle line
    y = map_coordinates(contrast_img, [fitline2, x], order=2)

    try:
        # Intersection of detected v-cut lines:
        i0, i1 = intersection(l1, l3)
    except TypeError:
        raise Exception('no intersection found')
    # Angle of intersection:
    angle = abs(angle2(l1, l3))
    dx = np.asfarray(x) - i0
    dy = fitline2 - i1
    # radii from intersection:
    r = np.hypot(dx, dy)  

#     print(dx, i0, i1)
    # exclude area behind intersection:
    behind_intersection = np.argmax(dx > 0)
    
    if behind_intersection:
#         print(behind_intersection)
#         import pylab as plt
#         plt.imshow(contrast_img)
#         plt.show()
#         plt.plot(r, y)
#         plt.plot(r[behind_intersection:], y[behind_intersection:])
#      
#         plt.show()
        
        r = r[behind_intersection:]
        y = y[behind_intersection:]

    if img_unmasked is None:
        # only one masked image avail. normalize y 0...1:
        mx = y.max()
        mn = y.min()
        mean = y.mean()
        t0 = 0.7 * mx - 0.3 * mean
        t1 = 0.7 * mn + 0.3 * mean
        low = np.median(y[y < t1])
        high = np.median(y[y > t0])
        y -= low
        y /= (high - low)

    return (r, y, angle), (fitline1, fitline2, fitline3), (contrast_img, dsub)

 
def plot(VALS, LINES, SUB):
    '''
    plot the result of vCut2MTF
    '''
    r, y, angle = VALS
    (fitline1, fitline2, fitline3) = LINES
    (sub, _dsub) = SUB
    
    r50 = findXAt(r, y, 0.5)

    degangle = np.degrees(angle)

    plt.figure('Rectified relative image, calc. angle=%.2f DEG' % degangle)

    plt.imshow(sub, cmap='Greys_r', aspect='auto', clim=(0, 1))
    plt.colorbar()
    
    plt.plot(fitline1)
    plt.plot(fitline2)
    plt.plot(fitline3)

    plt.figure("Line plot")
    plt.plot(r, y)
    plt.xlabel('distance [px[')
    plt.ylabel('image contrast [-]')
    plt.vlines([r50], 0, 1)
    plt.annotate('r50=%.2f' % r50, (r.mean(), 0.6))
    plt.hlines([0.5], r.min(), r.max())
    

if __name__ == '__main__':
    from generate import patVcut
    import pylab as plt
    
    std = 2
    img_masked, img_unmasked, line = patVcut(std)
    out = measureVcut(img_masked, line, img_unmasked)
        
    plot(*out)    
    plt.show()
    
