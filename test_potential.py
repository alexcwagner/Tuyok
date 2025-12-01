# -*- coding: utf-8 -*-
"""
Test harness for ellipsoid gravitational potential functions.

Validates against known analytic results:
- Sphere: uniform surface potential = (2/3)πR²
- Sphere exterior: point mass equivalent
- Continuity at surface boundary
- Symmetry for oblate/prolate spheroids
"""

from compute_harness import GLSLComputeHarness, ShaderConfig, BufferSpec, UniformSpec
import numpy as np

def test_potential():
    harness = GLSLComputeHarness()
    
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/test_potential.glsl.c", config)
    
    N = 10  # Number of test slots (we use 7 currently)
    
    # Define output dtype matching the GLSL struct
    result_dtype = np.dtype([
        ('a', np.float64),
        ('b', np.float64),
        ('c', np.float64),
        ('test_coord', np.float64),
        ('test_type', np.uint32),
        ('_pad0', np.uint32),
        ('potential_x', np.float64),
        ('potential_y', np.float64),
        ('potential_z', np.float64),
        ('expected', np.float64),
        ('error', np.float64),
        ('_pad1', np.float64),
    ])
    
    buffers = [
        BufferSpec(
            binding=0,
            dtype=result_dtype,
            count=N,
            mode="out"
        )
    ]
    
    uniforms = [
        UniformSpec("num_tests", N, "1ui"),
        UniformSpec("seed", 42, "1ui"),
    ]
    
    print("Running potential function tests...")
    print("=" * 70)
    
    results = program.run(buffers, uniforms, num_invocations=N)
    data = results[0]
    
    test_names = [
        "Sphere surface (R=1): uniform potential = (4/3)π",
        "Sphere surface (R=2): uniform potential = (16/3)π",
        "Sphere exterior (r=2R): point mass equivalence",
        "Continuity: interior(R-ε) ≈ exterior(R+ε)",
        "Oblate spheroid (a=b>c): φ_x = φ_y",
        "Prolate spheroid (a>b=c): φ_y = φ_z",
        "Triaxial (a≠b≠c): all potentials differ",
    ]
    
    all_passed = True
    
    for i, row in enumerate(data):
        if row['test_type'] == 99:
            continue
            
        name = test_names[i] if i < len(test_names) else f"Test {i}"
        
        print(f"\nTest {i}: {name}")
        print(f"  Ellipsoid: a={row['a']:.4f}, b={row['b']:.4f}, c={row['c']:.4f}")
        print(f"  Potentials: φ_x={row['potential_x']:.10f}")
        print(f"              φ_y={row['potential_y']:.10f}")
        print(f"              φ_z={row['potential_z']:.10f}")
        
        if row['expected'] != 0:
            print(f"  Expected:   {row['expected']:.10f}")
        
        print(f"  Rel error:  {row['error']:.2e}")
        
        # Determine pass/fail
        if i == 3:  # Continuity test - use looser tolerance
            tolerance = 1e-6
        else:
            tolerance = 1e-10
        if i == 6:  # Triaxial test - error should be NONZERO
            passed = row['error'] > 0.01  # Should have significant spread
            status = "PASS (potentials differ)" if passed else "FAIL (potentials too similar)"
        else:
            passed = row['error'] < tolerance
            status = "PASS" if passed else f"FAIL (tolerance: {tolerance:.0e})"
        
        print(f"  Status:     {status}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
    
    program.cleanup()
    return all_passed


if __name__ == '__main__':
    test_potential()