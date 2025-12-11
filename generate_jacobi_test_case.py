import numpy as np
from scipy.special import elliprj, elliprf, elliprd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import json

def jacobi_residual(a, b):
    c = 1. / a / b
    a2, b2, c2 = a*a, b*b, c*c
    
    #rja = elliprj(b2, c2, a2, a2)
    #rjb = elliprj(a2, c2, b2, b2)
    #rjc = elliprj(a2, b2, c2, c2)

    rja = elliprd(b2, c2, a2)
    rjb = elliprd(a2, c2, b2)
    rjc = elliprd(a2, b2, c2)


    #print(rja, rjb, rjc)
    #print(rja2, rjb2, rjc2)


    if a==b:
        return np.inf
    
    return (a2*b2)/(b2-a2)*(rja-rjb) - c2*rjc

def generate_test_case(a):
    # We're given 'a', and we know abc=1, and that b>c, so we can
    # say that the bounds for b is sqrt(1/a) < b < a
    lo, hi = np.sqrt(1/a), a
    #print(lo, hi)
    b = brentq(lambda b: jacobi_residual(a, b), lo, hi)
    
    c = 1./a/b
    
    config = {
        'angular_momentum': 0.,
        'layers': [
            {
                'abc': (a, b, c),
                'density': 1.,
            },
        ]
    }
    
    return config


if __name__ == "__main__":
    #config = generate_test_case(1.19723)
    config = generate_test_case(1.3)
    print(json.dumps(config, indent=4))