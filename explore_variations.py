# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:19:59 2025

@author: alexc
"""

from compute_harness import GLSLComputeHarness, ShaderConfig, BufferSpec, UniformSpec
import numpy as np
import struct

harness = GLSLComputeHarness()


def test_variations():
    
    # Test with double/double precision
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/explore_variations.glsl.c", config)
    
    N = 5
    
    angular_momentum = 4.01
    layers = [
        {
            'semiaxes': (0.99, 1., 1.01),
            'volumetric_radius': 1.,
            'density': 1.05,
        },
        {
            'semiaxes': (1.98, 2., 2.02),
            'volumetric_radius': 2.,
            'density': 2.10,
        },
        {
            'semiaxes': (2.97, 3., 3.03),
            'volumetric_radius': 3.,
            'density': 3.15,
        }
    ]
   

    # Header: 32 bytes total
    input_bytes = struct.pack('dI',  # double + uint = 12 bytes
                              angular_momentum,
                              len(layers))
    input_bytes += b'\x00' * 20  # 20 bytes padding to reach offset 32

    for layer in layers:
        # Layer: 64 bytes each
        input_bytes += struct.pack('ddddd',  # 8 doubles = 64 bytes
                               layer['semiaxes'][0],    # offset 0
                               layer['semiaxes'][1],    # offset 8
                               layer['semiaxes'][2],    # offset 16
                               #0.0,                     # offset 24 (dvec3 padding)
                               layer['volumetric_radius'], # offset 32
                               layer['density'],        # offset 40
                               #0.0,                     # offset 48 (struct padding)
                               #0.0)                     # offset 56 (struct padding)
                               )

    input_array = np.frombuffer(input_bytes, dtype=np.uint8)
    
    print(f"Input buffer size: {len(input_array)} bytes")
    print(f"  Expected: 32 + {len(layers)} * 64 = {32 + len(layers) * 64} bytes")
    
    # ========================================================================
    # Define output buffer dtype to match GLSL std430 layout
    # ========================================================================
    
    layer_dtype = np.dtype([
        ('a', np.float64),
        ('b', np.float64),
        ('c', np.float64),
        ('volumetric_radius', np.float64),
        ('density', np.float64),         
        #('_pad_struct', np.float64, (2,))
    ])
    
    # Model struct layout:
    #   offset 0:    double angular_momentum (8 bytes)
    #   offset 8:    uint num_layers (4 bytes)
    #   offset 12:   padding (20 bytes to align to 32)
    #   offset 32:   Layer layers[20] (20 * 64 = 1280 bytes)
    #   offset 1312: double rel_equipotential_err (8 bytes)
    #   offset 1320: double total_energy (8 bytes)
    #   Total: 1328 bytes, rounded to 1344 for struct alignment
    
    model_dtype = np.dtype([
        ('angular_momentum', np.float64),      # 8 bytes at offset 0
        ('num_layers', np.uint32),             # 4 bytes at offset 8
        ('_pad_to_32', np.uint8, (4,)),       # 20 bytes at offset 12
        ('layers', layer_dtype, (20,)),        # 1280 bytes at offset 32
        ('rel_equipotential_err', np.float64), # 8 bytes at offset 1312
        ('total_energy', np.float64),          # 8 bytes at offset 1320
        #('_pad_struct', np.uint8, (16,))       # 16 bytes to reach 1344
    ])
    
    print(f"Model dtype size: {model_dtype.itemsize} bytes (expected 1344)")
    print(f"Layer dtype size: {layer_dtype.itemsize} bytes (expected 64)")
    
    # Verify offsets
    print("\nModel dtype field offsets:")
    for name in model_dtype.names:
        offset = model_dtype.fields[name][1]
        size = model_dtype.fields[name][0].itemsize
        print(f"  {name}: offset {offset}, size {size}")
    
    # ========================================================================
    # Define buffers
    # ========================================================================
    
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
            dtype=model_dtype,
            count=N,
            mode="out"
        )
    ]
    
    uniforms = [
        UniformSpec("num_variations", N, "1ui"),
        UniformSpec("seed", 12345, "1ui"),
        UniformSpec("annealing_temperature", 0.001, "1d")
    ]    

    print(f"\nGenerating {N} variations...")
    results = program.run(buffers, uniforms, num_invocations=N)
    
    variations = results[1]
    #print(bytes(variations))
    # ========================================================================
    # Examine results
    # ========================================================================
    
    #print(variations[0])
    #print(variations[1])
    
    
    print(f"\nFirst 5 variations:")
    for i in range(min(5, N)):
        var = variations[i]
        
        ang_mom = var['angular_momentum']
        num_layers = var['num_layers']
        rel_err = var['rel_equipotential_err']
        tot_energy = var['total_energy']
        
        print(f"\nVariation {i}:")
        print(f"  Angular momentum: {ang_mom:.6f}")
        print(f"  Num layers: {num_layers}")
        print(f"  Rel Eqp Err: {rel_err:.6f}")
        print(f"  Total Energy: {tot_energy:.6f}")
        for j in range(min(3, num_layers)):
            layer = var['layers'][j]
            a, b, c = layer['a'], layer['b'], layer['c']
            vol_radius = layer['volumetric_radius']
            density = layer['density']
            
            print(f"  Layer {j}:")
            print(f"    Semiaxes: a={a:.6f}, b={b:.6f}, c={c:.6f}")
            print(f"    Vol. radius: {vol_radius:.6f}")
            print(f"    Density: {density:.6f}")
            
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    first_layers = variations['layers'][:, 0]
    all_a_values = first_layers['a'][:]
    
    print(f"\nStatistics for first layer 'a' semiaxis:")
    print(f"  Mean: {np.mean(all_a_values):.6f}")
    print(f"  Std:  {np.std(all_a_values):.6f}")
    print(f"  Min:  {np.min(all_a_values):.6f}")
    print(f"  Max:  {np.max(all_a_values):.6f}")


if __name__ == '__main__':
    test_variations()