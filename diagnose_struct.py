# -*- coding: utf-8 -*-
"""
Diagnostic test for struct packing issues
"""

from Model import Model
import numpy as np
import struct

print("=" * 70)
print("DIAGNOSTIC TEST: STRUCT PACKING")
print("=" * 70)

# Create a simple model
model = Model({
    'angular_momentum': 0.5,
    'layers': [
        {
            'abc': (1.0, 1.0, 1.0),
            'density': 1.0,
        },
        {
            'abc': (2.0, 2.0, 2.0),
            'density': 1.0,
        }
    ]
})

print("\nOriginal model:")
print(f"  angular_momentum: {model['angular_momentum']}")
print(f"  num_layers: {len(model['layers'])}")
print(f"  layer[0]: {model['layers'][0]}")
print(f"  layer[1]: {model['layers'][1]}")

# Convert to struct
input_bytes = model.to_struct()
print(f"\nConverted to struct: {len(input_bytes)} bytes")

# Dump the struct
model.dump_struct_hex(input_bytes)

# Now read it back as numpy array to verify
input_array = np.frombuffer(input_bytes, dtype=Model._model_dtype, count=1)
print("\nRead back as numpy array:")
model.dump_numpy_struct(input_array[0])

# Now let's test with the GPU
print("\n" + "=" * 70)
print("TESTING WITH GPU")
print("=" * 70)

num_variants = 10  # Small number for testing
temperature = 0.01  # Very small temperature so variations are minimal

best, top_models = model.explore_variations(num_variants, temperature, top_k=3, seed=12345)

print("\n" + "=" * 70)
print("RESULTS FROM GPU")
print("=" * 70)

print(f"\nBest model rel_equipotential_err: {best['rel_equipotential_err']}")
print(f"Best model total_energy: {best['total_energy']}")

print("\nTop 3 models:")
for i, m in enumerate(top_models):
    print(f"  {i}: rel_equipotential_err={m['rel_equipotential_err']:.6e}, "
          f"total_energy={m['total_energy']}")

# Let's also look at the raw bytes if we can get them
print("\n" + "=" * 70)
print("EXAMINING SPECIFIC RESULT")
print("=" * 70)

# Create a test model with the result and convert it back to see what happened
test_struct = np.zeros(1, dtype=Model._model_dtype)
test_struct['angular_momentum'] = best['angular_momentum']
test_struct['num_layers'] = len(best['layers'])
for i, layer in enumerate(best['layers']):
    test_struct['layers'][0][i]['a'] = layer['abc'][0]
    test_struct['layers'][0][i]['b'] = layer['abc'][1]
    test_struct['layers'][0][i]['c'] = layer['abc'][2]
    test_struct['layers'][0][i]['r'] = layer['r']
    test_struct['layers'][0][i]['density'] = layer['density']
test_struct['rel_equipotential_err'] = best['rel_equipotential_err']
test_struct['total_energy'] = best['total_energy']

# Convert to bytes and dump
result_bytes = test_struct.tobytes()
print(f"\nResult struct as bytes: {len(result_bytes)} bytes")
model.dump_raw_bytes(result_bytes, "Result from GPU")

# Check if the value looks like garbage
if abs(best['rel_equipotential_err']) > 1e30:
    print("\n⚠️  WARNING: rel_equipotential_err appears to be garbage!")
    print(f"   Value: {best['rel_equipotential_err']}")
    print("   This suggests struct packing mismatch between Python and GLSL")
elif best['rel_equipotential_err'] < 0:
    print("\n⚠️  WARNING: rel_equipotential_err is negative!")
    print(f"   Value: {best['rel_equipotential_err']}")
    print("   This is physically impossible - struct packing issue likely")
else:
    print("\n✓ rel_equipotential_err looks reasonable")
    print(f"   Value: {best['rel_equipotential_err']}")