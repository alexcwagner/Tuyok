#!/usr/bin/env python3
"""
Test if BUFF_REAL macro actually writes doubles in struct fields
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

# Define matching struct
test_dtype = np.dtype([
    ('a', np.float64),
    ('b', np.float64),
    ('c', np.float64),
    ('d', np.float64),
])

buffers = [
    BufferSpec(
        binding=0,
        dtype=test_dtype,
        count=1,
        mode="out"
    )
]

results = program.run(buffers, num_invocations=1)

print("\nResults from struct fields:")
print(f"  test_structs[0].a = {results[0][0]['a']}  (BUFF_REAL(1.3LF))")
print(f"  test_structs[0].b = {results[0][0]['b']}  (BR(1.3LF))")
print(f"  test_structs[0].c = {results[0][0]['c']}  (double(1.3LF))")
print(f"  test_structs[0].d = {results[0][0]['d']}  (1.3LF direct)")

print("\nAs hex:")
raw_bytes = results[0].tobytes()
for i, field in enumerate(['a', 'b', 'c', 'd']):
    offset = i * 8
    field_bytes = raw_bytes[offset:offset+8]
    print(f"  {field} = {field_bytes.hex()}")

all_correct = True
for field in ['a', 'b', 'c', 'd']:
    val = results[0][0][field]
    if abs(val - 1.3) < 1e-15:
        print(f"  {field}: ✓ full double precision")
    else:
        print(f"  {field}: ✗ precision loss (got {val})")
        print(f"         float32(1.3) = {np.float32(1.3)}")
        all_correct = False

if all_correct:
    print("\n✓ ALL struct fields write full double precision")
else:
    print("\n✗ SOME struct fields have precision loss")

program.cleanup()