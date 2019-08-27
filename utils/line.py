'''
various functions for lines[x0,y0,x1,y1] 
'''

import numpy as np


def dxdy(line):
    """
    return line slope
    """
    x0, y0, x1, y1 = line
    dx = float(x1) - x0
    dy = float(y1) - y0
    return dx, dy 


def dxdy_normal(line):
    """
    return normalised ascent vector
    """
    x0, y0, x1, y1 = line
    dx = float(x1) - x0
    dy = float(y1) - y0
    f = np.hypot(dx, dy)
    if f == 0:
        return dx, dy
    return dx / f, dy / f


def resize(line, factor, anchor=0.5):
    """
    compress/stretch line
    factor: relative length (1->no change, 2-> double, 0.5:half)
    anchor: 0.5 ... line middle    0...start    1...end
    """
    dx, dy = dxdy(line)
    px, py = line[0] + (anchor * dx), line[1] + (anchor * dy) 
    dx *= factor 
    dy *= factor 
    a = 1 - anchor
    return px - (dx * anchor), py - (dy * anchor), px + (dx * a), py + (dy * a)


# code taken from: http://www.ariel.com.au/a/python-point-int-poly.html
def pointInsidePolygon(x, y, poly):
    """
    Determine if a point is inside a given polygon or not
    Polygon is a list of (x,y) pairs.
    
    returns bool

    let's make an easy square:

    >>> poly = [ (0,0),\
                 (1,0),\
                 (1,1),\
                 (0,1) ]
    >>> pointInsidePolygon(0.5,0.5, poly)
    True
    >>> pointInsidePolygon(1.5,1.5, poly)
    False
    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def normal(line):
    """return the unit normal vector"""
    dx, dy = dxdy_normal(line)
    return -dy, dx  # other normal v would be dy,-dx


def length(line):
    x0, y0, x1, y1 = line
    dx = x1 - x0
    dy = y1 - y0
    return np.hypot(dx, dy)


def angle(line):
    x0, y0, x1, y1 = line
    return np.arctan2(float(y1) - y0, float(x1) - x0)


def fromFn(ascent, offs, length=1, px=0):
    dx = length
    py = px * ascent + offs
    dy = ascent * length
    if length != 1:  # --> normalize
        l = length / (dx ** 2 + dy ** 2) ** 0.5
        dx *= l
        dy *= l
    return px, py, px + dx, py + dy


def angle2(line1, line2):
    '''return smallest angle between two lines'''
    # from
    # http://stackoverflow.com/questions/13226038/calculating-angle-between-two-lines-in-python
    x0, y0, x1, y1 = line1
    x1 -= x0
    y1 -= y0
    x0, y0, x2, y2 = line2
    x2 -= x0
    y2 -= y0
    inner_product = x1 * x2 + y1 * y2
    len1 = np.hypot(x1, y1)
    len2 = np.hypot(x2, y2)
    a = inner_product / (len1 * len2)
    return np.copysign(np.arccos(min(1, max(a, -1))), y2)


def isHoriz(line):
    a = abs(angle(line))
    return 0.25 * np.pi > a or a > 0.75 * np.pi


def intersectionWithX(line, x):
    dx, dy = dxdy(line)
    if (dx == 0):
        return None;  # fail - lines are parallel
    m = dy / dx;
    n = line[1] - (m * line[0])
    return (m * x) + n


def intersectionWithY(line, y):
    dx, dy = dxdy(line)
    if (dy == 0):
        return None;  # fail - lines are parallel
    m = dy / dx
    n = line[1] - (m * line[0])
    return (y - n) / m


def _near(a, b, rtol=1e-5, atol=1e-8):
    return abs(a - b) < (atol + rtol * abs(b))


def intersection(line1, line2):
    """
    Return the coordinates of a point of intersection given two lines.
    Return None if the lines are parallel, but non-colli_near.
    Return an arbitrary point of intersection if the lines are colli_near.

    Parameters:
    line1 and line2: lines given by 4 points (x0,y0,x1,y1).
    """
    x1, y1, x2, y2 = line1
    u1, v1, u2, v2 = line2
    (a, b), (c, d) = (x2 - x1, u1 - u2), (y2 - y1, v1 - v2)
    e, f = u1 - x1, v1 - y1
    
    # Solve ((a,b), (c,d)) * (t,s) = (e,f)
    denom = float(a * d - b * c)
    if _near(denom, 0):
        # parallel
        # If colli_near, the equation is solvable with t = 0.
        # When t=0, s would have to equal e/b and f/d
        if b == 0:
            yy = intersectionWithX(line1, u1)
            if yy is None:
                return None
            return u1, yy
        if d == 0:
            xx = intersectionWithY(line1, v1)
            if xx is None:
                return None
            return xx, v1
  
        if _near(e / b, f / d):
            # colli_near
            px = x1
            py = y1
        else:
            # no intersection
            return (0, 0) 
    else:
        t = (e * d - b * f) / denom
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
    return px, py


def segmentIntersection(line1, line2):
    i = intersection(line1, line2)
    if i is None:
        return None
    # line1 and line2 are finite:
    # check whether intersection is on both lines:
    if (pointIsBetween(line1[0:2], line1[2:], i)
            and pointIsBetween(line2[0:2], line2[2:], i)):
        return i
    return None


def distancePoint(p1, p2):
    # return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    dx = p1[1] - p2[1]
    dy = p1[0] - p1[0]
    return np.hypot(dx, dy)


def pointIsBetween(startP, endP, p):
    """
    whether point is (+-1e-6) on a straight
    line in-between two other points
    """
    return (distancePoint(startP, p) + distancePoint(p, endP)
            - distancePoint(startP, endP) < 1e-6)


def cutToFitIntoPolygon(line, polygon):
    """
    cut line so it fits into polygon
    polygon = (( x0,y0), (x1,y1) ,...)
    """
    p0_inside = pointInsidePolygon(line[0], line[1], polygon)
    p1_inside = pointInsidePolygon(line[2], line[3], polygon)

    if not p0_inside or not p1_inside:
        for (i0, j0), (i1, j1) in zip(polygon[:-1], polygon[1:]):

            isec = segmentIntersection(line, (i0, j0, i1, j1))

            if isec is not None:
                if not p0_inside:
                    line = (isec[0], isec[1], line[2], line[3])
                    p0_inside = True
                elif not p1_inside:
                    line = (line[0], line[1], isec[0], isec[1])
                    p1_inside = True

            if p0_inside and p1_inside:
                break
    return line
