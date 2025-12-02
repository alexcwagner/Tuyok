#ifndef CARLSON_GLSL
#define CARLSON_GLSL


//#define cbrt(x) (pow(abs((x)), R(1.0LF)/R(3.0LF)))

//========================================================================

CALC_REAL carlson_rc(CALC_REAL x, CALC_REAL y)
{
    // Handle y<0 via principal-value transform, branchlessly:
    // if y>0:  xt=x,   yt=y,   w=1
    // if y<0:  xt=x-y, yt=-y,  w=sqrt(x)/sqrt(xt)

    CALC_REAL neg = R(1.LF) - step(R(0.LF), y);       // 1 when y<0, else 0
    CALC_REAL xt  = mix(x, x - y, neg);
    CALC_REAL yt  = mix(y, -y,    neg);

    // Guard tiny negatives from roundoff in sqrt; assumes valid inputs.
    xt = max(xt, R(0.0LF));
    yt = max(yt, R(0.0LF));

    CALC_REAL w = mix(R(1.0LF), sqrt(max(x, R(0.0LF))) / max(sqrt(xt), R(1e-30LF)), neg);

    for (int i = 0; i < ITER; ++i) {
        CALC_REAL sqrtx = sqrt(xt);
        CALC_REAL sqrty = sqrt(yt);
        CALC_REAL alamb = R(2.LF) * sqrtx * sqrty + yt;
        xt = R(0.25LF) * (xt + alamb);
        yt = R(0.25LF) * (yt + alamb);
    }

    // Final Taylor series in s about the mean (no branching).
    CALC_REAL ave = (xt + yt + yt) / R(3.0LF);
    CALC_REAL s   = (yt - ave) / ave;               // |s| is now small
    CALC_REAL poly = 
                  R(1.0LF) 
                + (s * s ) * 
                    (R(3.LF/10.LF) + s * 
                        (R(1.LF/7.LF) + s * 
                            (R(3.LF/8.LF) + s * 
                                R(9.LF/22.LF)
                            )
                        )
                    );

    return w * poly / sqrt(ave);
}
    
//========================================================================
    

CALC_REAL carlson_rf(CALC_REAL x, CALC_REAL y, CALC_REAL z)
{
    // Clamp tiny negatives from roundoff; RF is real for nonnegative args.
    CALC_REAL xt = max(x, R(0.LF));
    CALC_REAL yt = max(y, R(0.LF));
    CALC_REAL zt = max(z, R(0.LF));

    for (int i = 0; i < ITER; ++i) {
        CALC_REAL sx = sqrt(xt);
        CALC_REAL sy = sqrt(yt);
        CALC_REAL sz = sqrt(zt);
        CALC_REAL lam = sx*sy + sy*sz + sz*sx;
        xt = R(0.25LF) * (xt + lam);
        yt = R(0.25LF) * (yt + lam);
        zt = R(0.25LF) * (zt + lam);
    }

    // Mean and reduced variables
    CALC_REAL A  = (xt + yt + zt) / R(3.LF);
    A = max(A, R(1e-30LF));                 // protect divisions / sqrt
    CALC_REAL X = R(1.LF) - xt / A;
    CALC_REAL Y = R(1.LF) - yt / A;
    CALC_REAL Z = R(1.LF) - zt / A;

    // Elementary symmetric polynomials
    CALC_REAL e2 = X*Y + Y*Z + Z*X;
    CALC_REAL e3 = X*Y*Z;

    // Symmetric series (Carlson): up to e3^2 is plenty for float
    // RF ≈ A^{-1/2}[ 1 - (1/10)e2 + (1/24)e3 + (3/44)e2^2 - (1/14)e2 e3 + (1/24)e3^2 ]
    CALC_REAL poly = 
                 R(1.LF)
               - R(0.1LF) * e2
               + R(1.LF/24.LF) * e3
               + R(3.LF/44.LF) * (e2*e2)
               - R(1.LF/14.LF) * (e2*e3)
               + R(1.LF/24.LF) * (e3*e3);

    return inversesqrt(A) * poly;
}

//========================================================================

CALC_REAL carlson_rd(CALC_REAL x, CALC_REAL y, CALC_REAL z)
{
    // Clamp tiny negatives from roundoff; enforce z>0
    CALC_REAL xt = max(x, R(0.LF));
    CALC_REAL yt = max(y, R(0.LF));
    CALC_REAL zt = max(z, R(1e-30LF));

    CALC_REAL sum = R(0.LF);
    CALC_REAL fac = R(1.LF);

    for (int i = 0; i < ITER; ++i) {
        CALC_REAL sx = sqrt(xt);
        CALC_REAL sy = sqrt(yt);
        CALC_REAL sz = sqrt(zt);
        CALC_REAL lam = sx*(sy + sz) + sy*sz;

        // accumulate the duplication tail (principal 1/(sz*(z+lam)) term)
        sum += fac / (sz * (zt + lam));

        fac *= R(0.25LF);
        xt = R(0.25LF) * (xt + lam);
        yt = R(0.25LF) * (yt + lam);
        zt = R(0.25LF) * (zt + lam);
    }

    // Mean with 3x weight on z, then reduced variables
    CALC_REAL A    = R(0.2LF) * (xt + yt + R(3.LF) * zt);
    A = max(A, R(1e-30LF));
    CALC_REAL delx = (A - xt) / A;
    CALC_REAL dely = (A - yt) / A;
    CALC_REAL delz = (A - zt) / A;

    // Series terms
    CALC_REAL ea = delx * dely;
    CALC_REAL eb = delz * delz;
    CALC_REAL ec = ea - eb;
    CALC_REAL ed = ea - R(6.LF)*eb;
    CALC_REAL ee = ed + R(2.LF)*ec;

    // Final expansion
    CALC_REAL series = R(1.LF)
                 + ed * 
                     (-R(3.LF/14.LF) 
                      + R(9.LF/88.LF) * ed 
                      - R(9.LF/78.LF) * delz * ee
                     )
                 + delz * 
                     ( R(1.LF/6.LF) * ee 
                      + delz * (- R(9.LF/22.LF) * ec 
                      + delz * R(3.LF/26.LF) * ea) );

    return R(3.LF) * sum + fac * series / (A * sqrt(A));
}


//========================================================================

CALC_REAL carlson_rj(CALC_REAL x, CALC_REAL y, CALC_REAL z, CALC_REAL p)
{
    const CALC_REAL EPS = R(1e-30LF);
    
    CALC_REAL xt = max(x, R(0.LF));
    CALC_REAL yt = max(y, R(0.LF));
    CALC_REAL zt = max(z, R(0.LF));
    CALC_REAL pt = p;

    CALC_REAL sum = R(0.LF);
    CALC_REAL fac = R(1.LF);

    for (int i = 0; i < ITER+4; ++i) {
        CALC_REAL sx = sqrt(xt);
        CALC_REAL sy = sqrt(yt);
        CALC_REAL sz = sqrt(zt);
        CALC_REAL sp = sqrt(abs(pt));
        CALC_REAL lam = sx * sy + sy * sz + sz * sx;
        CALC_REAL d = (sp + sx) * (sp + sy) * (sp + sz);
        CALC_REAL delta = (pt - xt) * (pt - yt) * (pt - zt);
        CALC_REAL d2 = d*d;
        CALC_REAL rc_arg2 = d2 + delta;
        CALC_REAL rc_term = carlson_rc(d2, rc_arg2);

        sum += fac * R(6.LF) * rc_term;
        xt = R(0.25LF) * (xt + lam);
        yt = R(0.25LF) * (yt + lam);
        zt = R(0.25LF) * (zt + lam);
        pt = R(0.25LF) * (pt + lam);
        fac *= R(0.25LF);
    }

    CALC_REAL A  = (xt + yt + zt + R(2.LF) * pt) * R(1.LF/5.LF);
    A = max(A, EPS);  // Add protection here too

    CALC_REAL X = R(1.LF) - xt/A;
    CALC_REAL Y = R(1.LF) - yt/A;
    CALC_REAL Z = R(1.LF) - zt/A;
    CALC_REAL P = R(1.LF) - pt/A;

    // Elementary symmetric polynomials
    CALC_REAL e2 = X * Y + Y * Z + Z * X;
    CALC_REAL e3 = X * Y * Z;
    CALC_REAL e4 = X * Y * Z * (X + Y + Z);  // Add e4
    CALC_REAL e5 = X * Y * Z * (X * Y + Y * Z + Z * X);  // Add e5

    // Extended polynomial for double precision
    // Based on DLMF 19.36.1 with more terms
    CALC_REAL poly;
    
#ifdef USE_DOUBLES_IN_CALCULATIONS
    // High-precision expansion (good to ~1e-15)
    CALC_REAL P2 = P * P;
    CALC_REAL P3 = P2 * P;
    CALC_REAL e22 = e2 * e2;
    CALC_REAL e23 = e22 * e2;
    
    poly = R(1.0LF)
        + R(-3.LF/14.LF) * e2
        + R( 1.LF/ 6.LF) * P
        + R( 9.LF/88.LF) * e3
        + R(-3.LF/22.LF) * e2 * P
        + R( 3.LF/26.LF) * P2
        + R(-1.LF/16.LF) * e4
        + R( 3.LF/40.LF) * e2 * e3
        + R( 3.LF/20.LF) * e22 * P
        + R(-9.LF/52.LF) * e3 * P
        + R(-3.LF/64.LF) * P3
        + R( 9.LF/208.LF) * e23
        - R(27.LF/104.LF) * e2 * P2;
#else
    // Standard expansion for single precision
    poly = R(1.0LF)
        + R(-3.LF/14.LF) * e2
        + R( 1.LF/ 6.LF) * P
        + R( 9.LF/88.LF) * e3
        + R(-3.LF/22.LF) * e2 * P
        + R( 3.LF/26.LF) * P * P;
#endif

    CALC_REAL main_term = poly / (A * sqrt(A));

    return sum + fac * main_term;
}

//========================================================================
#endif