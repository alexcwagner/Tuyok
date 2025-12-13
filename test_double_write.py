#!/usr/bin/env python3
"""
Test if BUFF_REAL macro actually writes doubles
"""
import sys
sys.path.insert(0, '.')

from compute_harness import GLSLComputeHarness, ShaderConfig, BufferSpec
import numpy as np

harness = GLSLComputeHarness()

config = ShaderConfig.precision_config("double", "double")
program = harness.create_program("shader/test_double_write.glsl.c", config)

# Check shader source
print("Shader BUFF_REAL definition:")
for line in program.source_code.split('\n'):
    if 'BUFF_REAL' in line and '#define' in line:
        print(f"  {line}")

buffers = [
    BufferSpec(
        binding=0,
        dtype=np.float64,
        count=3,
        mode="out"
    )
]

results = program.run(buffers, num_invocations=1)

print("\nResults:")
print(f"  test_values[0] = {results[0][0]}  (BUFF_REAL(1.3))")
print(f"  test_values[1] = {results[0][1]}  (BR(1.3))")
print(f"  test_values[2] = {results[0][2]}  (double(1.3))")

print("\nAs hex:")
for i in range(3):
    val_bytes = results[0][i:i+1].tobytes()
    print(f"  test_values[{i}] = {val_bytes.hex()}")

if abs(results[0][0] - 1.3) < 1e-15:
    print("\n✓ BUFF_REAL writes full double precision")
else:
    print(f"\n✗ BUFF_REAL writes float32 precision (value: {results[0][0]})")
    print(f"   Expected: 1.3")
    print(f"   Got: {results[0][0]}")
    print(f"   float32(1.3) = {np.float32(1.3)}")

program.cleanup()