// potential.glsl.c
// Gravitational potential of a homogeneous ellipsoid at on-axis points
//
// Uses Carlson elliptic integrals R_F and R_D
// Returns potential per unit (G * density), i.e. Φ / (G ρ)

#ifndef POTENTIAL_GLSL_C
#define POTENTIAL_GLSL_C

#include "shader/precision.glsl.c"
#include "shader/carlson.glsl.c"

// ============================================================================
// Interior potential - point inside the ellipsoid on an axis
// ============================================================================

// Interior potential at point (x, 0, 0) where |x| <= a
CALC_REAL potential_interior_x(CALC_REAL a, CALC_REAL b, CALC_REAL c, CALC_REAL x)
{
    CALC_REAL a2 = a * a;
    CALC_REAL b2 = b * b;
    CALC_REAL c2 = c * c;
    
    // I(0) = 2 * a * b * c * R_F(a², b², c²)
    CALC_REAL I0 = R(2.0LF) * a * b * c * carlson_rf(a2, b2, c2);
    
    // A_x(0) = (2/3) * a * b * c * R_D(b², c², a²)
    CALC_REAL Ax = R(2.0LF / 3.0LF) * a * b * c * carlson_rd(b2, c2, a2);
    
    // Φ = π G ρ [I(0) - A_x(0) * x²]
    return PI * (I0 - Ax * x * x);
}

// Interior potential at point (0, y, 0) where |y| <= b
CALC_REAL potential_interior_y(CALC_REAL a, CALC_REAL b, CALC_REAL c, CALC_REAL y)
{
    CALC_REAL a2 = a * a;
    CALC_REAL b2 = b * b;
    CALC_REAL c2 = c * c;
    
    CALC_REAL I0 = R(2.0LF) * a * b * c * carlson_rf(a2, b2, c2);
    CALC_REAL Ay = R(2.0LF / 3.0LF) * a * b * c * carlson_rd(a2, c2, b2);
    
    return PI * (I0 - Ay * y * y);
}

// Interior potential at point (0, 0, z) where |z| <= c
CALC_REAL potential_interior_z(CALC_REAL a, CALC_REAL b, CALC_REAL c, CALC_REAL z)
{
    CALC_REAL a2 = a * a;
    CALC_REAL b2 = b * b;
    CALC_REAL c2 = c * c;
    
    CALC_REAL I0 = R(2.0LF) * a * b * c * carlson_rf(a2, b2, c2);
    CALC_REAL Az = R(2.0LF / 3.0LF) * a * b * c * carlson_rd(a2, b2, c2);
    
    return PI * (I0 - Az * z * z);
}


// ============================================================================
// Exterior potential - point outside the ellipsoid on an axis
//
// For exterior points, we integrate from λ instead of 0, where λ is the
// positive root of: x²/(a²+λ) + y²/(b²+λ) + z²/(c²+λ) = 1
//
// On-axis, this simplifies:
//   Point (x,0,0) with |x| > a:  λ = x² - a²
//   Point (0,y,0) with |y| > b:  λ = y² - b²
//   Point (0,0,z) with |z| > c:  λ = z² - c²
// ============================================================================

// Exterior potential at point (x, 0, 0) where |x| > a
CALC_REAL potential_exterior_x(CALC_REAL a, CALC_REAL b, CALC_REAL c, CALC_REAL x)
{
    CALC_REAL a2 = a * a;
    CALC_REAL b2 = b * b;
    CALC_REAL c2 = c * c;
    CALC_REAL x2 = x * x;
    
    // λ = x² - a² for point (x, 0, 0) outside ellipsoid
    CALC_REAL lam = x2 - a2;
    
    // Shifted squares
    CALC_REAL a2_lam = a2 + lam;  // = x²
    CALC_REAL b2_lam = b2 + lam;
    CALC_REAL c2_lam = c2 + lam;
    
    // I(λ) = 2 * a * b * c * R_F(a²+λ, b²+λ, c²+λ)
    CALC_REAL I_lam = R(2.0LF) * a * b * c * carlson_rf(a2_lam, b2_lam, c2_lam);
    
    // A_x(λ) = (2/3) * a * b * c * R_D(b²+λ, c²+λ, a²+λ)
    CALC_REAL Ax_lam = R(2.0LF / 3.0LF) * a * b * c * carlson_rd(b2_lam, c2_lam, a2_lam);
    
    // Φ = π G ρ [I(λ) - A_x(λ) * x²]
    return PI * (I_lam - Ax_lam * x2);
}

// Exterior potential at point (0, y, 0) where |y| > b
CALC_REAL potential_exterior_y(CALC_REAL a, CALC_REAL b, CALC_REAL c, CALC_REAL y)
{
    CALC_REAL a2 = a * a;
    CALC_REAL b2 = b * b;
    CALC_REAL c2 = c * c;
    CALC_REAL y2 = y * y;
    
    CALC_REAL lam = y2 - b2;
    
    CALC_REAL a2_lam = a2 + lam;
    CALC_REAL b2_lam = b2 + lam;  // = y²
    CALC_REAL c2_lam = c2 + lam;
    
    CALC_REAL I_lam = R(2.0LF) * a * b * c * carlson_rf(a2_lam, b2_lam, c2_lam);
    CALC_REAL Ay_lam = R(2.0LF / 3.0LF) * a * b * c * carlson_rd(a2_lam, c2_lam, b2_lam);
    
    return PI * (I_lam - Ay_lam * y2);
}

// Exterior potential at point (0, 0, z) where |z| > c
CALC_REAL potential_exterior_z(CALC_REAL a, CALC_REAL b, CALC_REAL c, CALC_REAL z)
{
    CALC_REAL a2 = a * a;
    CALC_REAL b2 = b * b;
    CALC_REAL c2 = c * c;
    CALC_REAL z2 = z * z;
    
    CALC_REAL lam = z2 - c2;
    
    CALC_REAL a2_lam = a2 + lam;
    CALC_REAL b2_lam = b2 + lam;
    CALC_REAL c2_lam = c2 + lam;  // = z²
    
    CALC_REAL I_lam = R(2.0LF) * a * b * c * carlson_rf(a2_lam, b2_lam, c2_lam);
    CALC_REAL Az_lam = R(2.0LF / 3.0LF) * a * b * c * carlson_rd(a2_lam, b2_lam, c2_lam);
    
    return PI * (I_lam - Az_lam * z2);
}


// ============================================================================
// Convenience: potential at surface point (tip of axis)
// These are the interior functions evaluated at x=a, y=b, or z=c
// (Equivalent to exterior with λ=0)
// ============================================================================

CALC_REAL potential_surface_x(CALC_REAL a, CALC_REAL b, CALC_REAL c)
{
    return potential_interior_x(a, b, c, a);
}

CALC_REAL potential_surface_y(CALC_REAL a, CALC_REAL b, CALC_REAL c)
{
    return potential_interior_y(a, b, c, b);
}

CALC_REAL potential_surface_z(CALC_REAL a, CALC_REAL b, CALC_REAL c)
{
    return potential_interior_z(a, b, c, c);
}


CALC_REAL layer_potential_energy(CALC_REAL a, CALC_REAL b, CALC_REAL c, CALC_REAL density)
{
    CALC_REAL a2 = a * a;
    CALC_REAL b2 = b * b;
    CALC_REAL c2 = c * c;
    CALC_REAL abc = a * b * c;
    
    // I_0 = 2 * a * b * c * R_F(a², b², c²)
    CALC_REAL I0 = R(2.0LF) * abc * carlson_rf(a2, b2, c2);
    
    // PE = -(4π²/5) × G × ρ² × (abc)² × R_F(a², b², c²)
    // Since R_F appears in I0, and I0 = 2abc × R_F:
    // PE = -(2π²/5) × G × ρ² × abc × I0
    
    CALC_REAL PE = -R(2.0LF * PI * PI / 5.0LF) * density * density * abc * I0;
    
    return PE;
}


#endif
