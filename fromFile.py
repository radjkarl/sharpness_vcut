import cv2
import argparse
import sys
# local
from measure import measureVcut
from utils.transforms import resFactor2std
from resolutionFactor import resolutionFactor

# use in a command line 
# >> python fromFile.py ""

if __name__ == '__main__':

    if(len(sys.argv) == 1):
        # add args created in generate.py' to test this module
        sys.argv.append("masked.png")
        sys.argv.append(open("line.txt", 'r').read())
        sys.argv.append('-u unmasked.png')
        
    parser = argparse.ArgumentParser(description='Image sharpness (resolution factor) from masked images via v-cut method')
    parser.add_argument('masked', type=str, help='Path to masked image')
    parser.add_argument('line', type=str, help='Line within v-cut e.g. 10,20,100,25 for x0=10,y0=20,x1=100,y1=25')

    parser.add_argument('-u', '--unmasked', type=str, default=None, help='Path to unmasked image')
    parser.add_argument('-b', '--background', type=str, default=None, help='Path to background image')
    parser.add_argument('-w', '--max_width', type=int, default=101, help='Maximum width [px] of v=cut')
    
    parser.add_argument('--mask_not_dark', dest='mask_is_dark', action='store_false', help='use flag, if mask if not completely opaque')
    parser.set_defaults(mask_is_dark=True)
    
    args = parser.parse_args()
    
    try:
        line = args.line.split(',')
        if(len(line) != 4):
            raise Exception()
        line = [float(l) for l in line]
    except:
        raise Exception("need to write line as x0,y0,x1,y1  e.g.: 10,20,100,25")
    
    img_masked = cv2.imread(args.masked, cv2.IMREAD_GRAYSCALE)
    
    img_unmasked = args.unmasked
    if(img_unmasked != None):
        img_unmasked = cv2.imread(img_unmasked, cv2.IMREAD_GRAYSCALE)

    img_bg = args.background
    if(img_bg != None):
        img_bg = cv2.imread(img_bg, cv2.IMREAD_GRAYSCALE)

    out = measureVcut(img_masked, line, img_unmasked, img_bg,
             args.max_width, mask_is_dark=args.mask_is_dark)[0]
    
    fres = resolutionFactor(*out)
    print("Resolution factor=", fres)
    print("Corresp. std of Gaussian blur kernel=", resFactor2std(fres))    
    
