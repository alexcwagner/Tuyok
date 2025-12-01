#version 460 core

#ifndef TEST_POTENTIAL_GLSL_C
#define TEST_POTENTIAL_GLSL_C

#include "shader/precision.glsl.c"
#include "shader/potential.glsl.c"

// ============================================================================
// Test cases for ellipsoid gravitational potential
// ============================================================================

struct PotentialTestResult {
    // Inputs
    BUFF_REAL a, b, c;       // semiaxes
    BUFF_REAL test_coord;    // coordinate of test point (on whichever axis)
    uint test_type;          // 0=sphere_surface, 1=sphere_exterior, 2=continuity, 3=oblate
    uint _pad0;
    
    // Outputs
    BUFF_REAL potential_x;   // potential at (a,0,0) or (test_coord,0,0)
    BUFF_REAL potential_y;   // potential at (0,b,0) or (0,test_coord,0)
    BUFF_REAL potential_z;   // potential at (0,0,c) or (0,0,test_coord)
    BUFF_REAL expected;      // analytic expectation (where known)
    BUFF_REAL error;         // relative error or difference metric
    BUFF_REAL _pad1;
};

layout(std430, binding = 0) buffer OutputBuffer {
    PotentialTestResult results[];
};

uniform uint num_tests;
uniform uint seed;

layout(local_size_x = 256) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    if (idx >= num_tests) {
        return;
    }
    
    PotentialTestResult result;
    
    // ========================================================================
    // Test 0: Sphere surface - all three potentials should be equal
    //         and match analytic value (2/3)πR² for unit density
    // ========================================================================
    if (idx == 0u) {
        CALC_REAL R = R(1.0LF);
        result.a = BR(R);
        result.b = BR(R);
        result.c = BR(R);
        result.test_coord = BR(R);
        result.test_type = 0u;
        
        CALC_REAL phi_x = potential_surface_x(R, R, R);
        CALC_REAL phi_y = potential_surface_y(R, R, R);
        CALC_REAL phi_z = potential_surface_z(R, R, R);
        
        result.potential_x = BR(phi_x);
        result.potential_y = BR(phi_y);
        result.potential_z = BR(phi_z);
        
        // Analytic: Φ = (4/3)πGρR² -> for G=1, ρ=1, R=1: (4/3)π ≈ 4.1888
        CALC_REAL expected = R(4.0LF / 3.0LF) * R(3.14159265358979323846LF);
        result.expected = BR(expected);
        
        // Error: max deviation from expected
        CALC_REAL err_x = abs(phi_x - expected) / expected;
        CALC_REAL err_y = abs(phi_y - expected) / expected;
        CALC_REAL err_z = abs(phi_z - expected) / expected;
        result.error = BR(max(max(err_x, err_y), err_z));
    }
    
    // ========================================================================
    // Test 1: Sphere surface with R = 2
    // ========================================================================
    else if (idx == 1u) {
        CALC_REAL R = R(2.0LF);
        result.a = BR(R);
        result.b = BR(R);
        result.c = BR(R);
        result.test_coord = BR(R);
        result.test_type = 0u;
        
        CALC_REAL phi_x = potential_surface_x(R, R, R);
        CALC_REAL phi_y = potential_surface_y(R, R, R);
        CALC_REAL phi_z = potential_surface_z(R, R, R);
        
        result.potential_x = BR(phi_x);
        result.potential_y = BR(phi_y);
        result.potential_z = BR(phi_z);
        
        // Analytic: (4/3)πR² = (4/3)π(4) = (16/3)π ≈ 16.755
        CALC_REAL expected = R(4.0LF / 3.0LF) * R(3.14159265358979323846LF) * R * R;
        result.expected = BR(expected);
        
        CALC_REAL err_x = abs(phi_x - expected) / expected;
        CALC_REAL err_y = abs(phi_y - expected) / expected;
        CALC_REAL err_z = abs(phi_z - expected) / expected;
        result.error = BR(max(max(err_x, err_y), err_z));
    }
    
    // ========================================================================
    // Test 2: Sphere exterior - potential at r = 2R should equal point mass
    //         Φ_exterior = (4/3)πR³ / r = (4/3)πR³ / (2R) = (2/3)πR²
    // ========================================================================
    else if (idx == 2u) {
        CALC_REAL R = R(1.0LF);
        CALC_REAL r = R(2.0LF);  // exterior point at 2R
        result.a = BR(R);
        result.b = BR(R);
        result.c = BR(R);
        result.test_coord = BR(r);
        result.test_type = 1u;
        
        CALC_REAL phi_x = potential_exterior_x(R, R, R, r);
        CALC_REAL phi_y = potential_exterior_y(R, R, R, r);
        CALC_REAL phi_z = potential_exterior_z(R, R, R, r);
        
        result.potential_x = BR(phi_x);
        result.potential_y = BR(phi_y);
        result.potential_z = BR(phi_z);
        
        // Analytic: Φ = GM/r = (4/3)πρR³ / r, for ρ=1, R=1, r=2: (4/3)π(1)/(2) = (2/3)π
        CALC_REAL expected = R(4.0LF / 3.0LF) * R(3.14159265358979323846LF) * R * R * R / r;
        result.expected = BR(expected);
        
        CALC_REAL err_x = abs(phi_x - expected) / expected;
        CALC_REAL err_y = abs(phi_y - expected) / expected;
        CALC_REAL err_z = abs(phi_z - expected) / expected;
        result.error = BR(max(max(err_x, err_y), err_z));
    }
    
    // ========================================================================
    // Test 3: Continuity at surface - interior(a-ε) ≈ exterior(a+ε)
    // ========================================================================
    else if (idx == 3u) {
        CALC_REAL R = R(1.0LF);
        CALC_REAL eps = R(1e-9LF);
        result.a = BR(R);
        result.b = BR(R);
        result.c = BR(R);
        result.test_coord = BR(eps);
        result.test_type = 2u;
        
        CALC_REAL phi_int = potential_interior_x(R, R, R, R - eps);
        CALC_REAL phi_ext = potential_exterior_x(R, R, R, R + eps);
        CALC_REAL phi_surf = potential_surface_x(R, R, R);
        
        result.potential_x = BR(phi_int);
        result.potential_y = BR(phi_ext);
        result.potential_z = BR(phi_surf);
        result.expected = BR(phi_surf);
        
        // Error: how well do interior and exterior match at the boundary?
        CALC_REAL err = abs(phi_int - phi_ext) / phi_surf;
        result.error = BR(err);
    }
    
    // ========================================================================
    // Test 4: Oblate spheroid (a = b > c) - x and y potentials should match
    // ========================================================================
    else if (idx == 4u) {
        CALC_REAL a = R(2.0LF);
        CALC_REAL b = R(2.0LF);
        CALC_REAL c = R(1.0LF);
        result.a = BR(a);
        result.b = BR(b);
        result.c = BR(c);
        result.test_coord = BR(0.0LF);
        result.test_type = 3u;
        
        CALC_REAL phi_x = potential_surface_x(a, b, c);
        CALC_REAL phi_y = potential_surface_y(a, b, c);
        CALC_REAL phi_z = potential_surface_z(a, b, c);
        
        result.potential_x = BR(phi_x);
        result.potential_y = BR(phi_y);
        result.potential_z = BR(phi_z);
        result.expected = BR(phi_x);  // x and y should match
        
        // Error: difference between x and y potentials (should be ~0)
        CALC_REAL err_xy = abs(phi_x - phi_y) / phi_x;
        result.error = BR(err_xy);
    }
    
    // ========================================================================
    // Test 5: Prolate spheroid (a > b = c) - y and z potentials should match
    // ========================================================================
    else if (idx == 5u) {
        CALC_REAL a = R(2.0LF);
        CALC_REAL b = R(1.0LF);
        CALC_REAL c = R(1.0LF);
        result.a = BR(a);
        result.b = BR(b);
        result.c = BR(c);
        result.test_coord = BR(0.0LF);
        result.test_type = 3u;
        
        CALC_REAL phi_x = potential_surface_x(a, b, c);
        CALC_REAL phi_y = potential_surface_y(a, b, c);
        CALC_REAL phi_z = potential_surface_z(a, b, c);
        
        result.potential_x = BR(phi_x);
        result.potential_y = BR(phi_y);
        result.potential_z = BR(phi_z);
        result.expected = BR(phi_y);  // y and z should match
        
        // Error: difference between y and z potentials (should be ~0)
        CALC_REAL err_yz = abs(phi_y - phi_z) / phi_y;
        result.error = BR(err_yz);
    }
    
    // ========================================================================
    // Test 6: Triaxial ellipsoid - all three potentials should differ
    //         Just a sanity check that we get different values
    // ========================================================================
    else if (idx == 6u) {
        CALC_REAL a = R(3.0LF);
        CALC_REAL b = R(2.0LF);
        CALC_REAL c = R(1.0LF);
        result.a = BR(a);
        result.b = BR(b);
        result.c = BR(c);
        result.test_coord = BR(0.0LF);
        result.test_type = 3u;
        
        CALC_REAL phi_x = potential_surface_x(a, b, c);
        CALC_REAL phi_y = potential_surface_y(a, b, c);
        CALC_REAL phi_z = potential_surface_z(a, b, c);
        
        result.potential_x = BR(phi_x);
        result.potential_y = BR(phi_y);
        result.potential_z = BR(phi_z);
        result.expected = BR(0.0LF);  // no single expected value
        
        // "Error" here is just the spread - should be nonzero
        CALC_REAL mean = (phi_x + phi_y + phi_z) / R(3.0LF);
        CALC_REAL var = ((phi_x - mean) * (phi_x - mean) +
                         (phi_y - mean) * (phi_y - mean) +
                         (phi_z - mean) * (phi_z - mean)) / R(3.0LF);
        result.error = BR(sqrt(var) / mean);  // coefficient of variation
    }
    
    // ========================================================================
    // Padding for unused test slots
    // ========================================================================
    else {
        result.a = BR(0.0LF);
        result.b = BR(0.0LF);
        result.c = BR(0.0LF);
        result.test_coord = BR(0.0LF);
        result.test_type = 99u;
        result._pad0 = 0u;
        result.potential_x = BR(0.0LF);
        result.potential_y = BR(0.0LF);
        result.potential_z = BR(0.0LF);
        result.expected = BR(0.0LF);
        result.error = BR(0.0LF);
        result._pad1 = BR(0.0LF);
    }
    
    results[idx] = result;
}

#endif