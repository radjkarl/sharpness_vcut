# sharpness_vcut
A reference implementation for measuring image sharpness using 'v-cut' method.
Sharpness is measured as 'resolution factor' (fres) and standard deviation of Gaussian blur kernel (std).

Most modules in this project can be tested through execution.
For calculating image sharpness, an image with a V-shapes mask is required as well as the approximate position of the v-cut as line[x0,y0,x1,y1] {px}
Point (x0,y0) should be behind v-cut intersection in masked area
Point (x1,y1) should be within the unmasked area in the v-cut 

Run the following in a comand prompt to calculate image sharpness:

    python fromFile.py path_to_masked_img.py x0,y1,x1,y1

It is recommended to also create an unmasked image and a background image.
This allows to calculate a relative contrast image which is not influenced by local intensity deviations:

    python fromFile.py path_to_masked_img.py x0,y1,x1,y1 unmasked.png background.png

## Requirements
- Python 3 with numpy, scipy and opencv installed


## Contents
- validation.py
    - validate v-cut method. Here a synthetic pattern is blurred with a known [std].
    This value should match the value optained through measurement.
- relation_resolutionFactor_vs_std.py
    - [fres] and [std] are mainly direct proportional.
    the relation between both is calulated in this file and used in /utils/transforms
    to obtain [std] from [fres] and vice versa.
- measure.py
    - measure contrast reduction for the middle line in a v-cut
- resolutionFactor.py
    - obtain [fres] from result in measure.py
- get_uncorrected_fres_values.py
    - resolutionFactor.py uses a set of coefficients to map measurement results to [fres]
    execute this module to obtain raw values used in get_fres_coeffs.py
- get_fres_coeffs.py
    - obtain coefficients from result of get_uncorrected_fres_values.py
- generate.py
    - synthetic v-cut pattern generation
- /utils 
    - various methods needs to run modules of this project

## References
https://www.researchgate.net/publication/320921460_Quantitative_Electroluminescence_Measurements_of_PV_Devices
... V-cut method p73
... resolution factor p72 (eq 3.31), validation p92
... standard deviation p70
... image masking p72