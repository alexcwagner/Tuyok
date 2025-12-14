#!/usr/bin/env python3
"""
Test if input buffer is being read correctly by GPU
"""
import sys
sys.path.insert(0, '.')

from Model import Model
from compute_harness import GLSLComputeHarness, ShaderConfig, BufferSpec
import numpy as np

# Create a test model
model = Model({
    "angular_momentum": 1.2874789457385492,
    "layers": [
        {
            "abc": [1.3, 1.105575570700132, 0.6957740290368712],
            "density": 1.0
        }
    ]
})

# Pack it exactly as Model.to_struct() does
input_bytes = model.to_struct()

print("=== INPUT DATA ===")
print(f"angular_momentum: {model['angular_momentum']}")
print(f"layers[0].a: {model['layers'][0]['abc'][0]}")
print(f"layers[0].b: {model['layers'][0]['abc'][1]}")
print(f"layers[0].c: {model['layers'][0]['abc'][2]}")
print(f"layers[0].r: {model['layers'][0]['r']}")
print(f"layers[0].density: {model['layers'][0]['density']}")

print(f"\nInput buffer size: {len(input_bytes)} bytes")

# Create shader and run
harness = GLSLComputeHarness()
config = ShaderConfig.precision_config("double", "double")
program = harness.create_program("shader/test_input_reading.glsl.c", config)

input_array = np.frombuffer(input_bytes, dtype=np.uint8)

buffers = [
    BufferSpec(
        binding=0,
        dtype=np.uint8,
        count=len(input_array),
        mode="in",
        initial_data=input_array
    ),
    BufferSpec(
        binding=1,
        dtype=np.float64,
        count=6,
        mode="out"
    )
]

results = program.run(buffers, num_invocations=1)

print("\n=== GPU READ VALUES ===")
print(f"angular_momentum: {results[1][0]}")
print(f"layers[0].a: {results[1][1]}")
print(f"layers[0].b: {results[1][2]}")
print(f"layers[0].c: {results[1][3]}")
print(f"layers[0].r: {results[1][4]}")
print(f"layers[0].density: {results[1][5]}")

print("\n=== COMPARISON ===")
fields = ['angular_momentum', 'a', 'b', 'c', 'r', 'density']
expected = [
    model['angular_momentum'],
    model['layers'][0]['abc'][0],
    model['layers'][0]['abc'][1],
    model['layers'][0]['abc'][2],
    model['layers'][0]['r'],
    model['layers'][0]['density']
]

all_match = True
for i, (field, exp) in enumerate(zip(fields, expected)):
    got = results[1][i]
    match = abs(got - exp) < 1e-10
    status = "✓" if match else "✗"
    print(f"{status} {field:20s}: expected {exp:.15f}, got {got:.15f}")
    if not match:
        all_match = False
        print(f"   Difference: {got - exp:.15e}")
        print(f"   As float32: {np.float32(exp)}")

if all_match:
    print("\n✓ ALL values match - input buffer reading works correctly")
else:
    print("\n✗ MISMATCH - input buffer has precision loss")

program.cleanup()