# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:19:59 2025

@author: alexc
"""

from claude_harness import GLSLComputeHarness, ShaderConfig, BufferSpec, UniformSpec
import numpy as np
import time
import struct

harness = GLSLComputeHarness()



def test_variations():
    
    # Test with double/double precision
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/explore_variations.glsl.c", config)
    
    N = 1_000
    
    angular_momentum = 1.0;
    layers = [
        {
            'semiaxes': (1., 1., 1.),
            'average_radius': 1.,
            'density': 1.,
        },
        {
            'semiaxes': (2., 2., 2.),
            'average_radius': 2.,
            'density': 1.,
        },
        {
            'semiaxes': (3., 3., 3.),
            'average_radius': 3.,
            'density': 1.,
        }
    ]
   
    # Pack: angular_momentum + num_layers + padding + layers
    input_bytes = struct.pack('dII',  # double, uint, uint (padding)
                              angular_momentum,
                              len(layers),
                              0)  # padding for alignment

    for layer in layers:
        # Pack: vec3 (3 doubles) + density (1 double) = 32 bytes per layer
        input_bytes += struct.pack('ddddd', 
                               layer['semiaxes'][0],  # a
                               layer['semiaxes'][1],  # b
                               layer['semiaxes'][2],  # c
                               layer['average_radius'],
                               layer['density'])

    input_array = np.frombuffer(input_bytes, dtype=np.uint8)
    
    
    # ========================================================================
    # Define output buffer (N variations)
    # ========================================================================
    
    layer_dtype = np.dtype([
        ('semiaxes', np.float64, (3,)),  # vec3
        ('average_radius', np.float64),
        ('density', np.float64)
    ])
    
    model_dtype = np.dtype([
        ('angular_momentum', np.float64),
        ('num_layers', np.uint32),
        ('_padding', np.uint32),
        ('layers', layer_dtype, (20,)),
        ('rel_equipotential_err', np.float64),
        ('total_energy', np.float64)
    ])
    
    # ========================================================================
    # Define buffers
    # ========================================================================
    
    buffers = [
        # Binding 0: Input template model
        BufferSpec(
            binding=0,
            dtype=np.uint8,
            count=len(input_array),
            mode="in",
            initial_data=input_array
        ),
        
        # Binding 1: Output variations
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
        UniformSpec("annealing_temperature", 1.0, "1f")
    ]    

    
    print(f"Generating {N} variations...")
    results = program.run(buffers, uniforms, num_invocations=N)
    
    # Get output
    variations = results[1]  # Binding 1
    
    # ========================================================================
    # Examine results
    # ========================================================================
    
    print(f"\nFirst 5 variations:")
    for i in range(min(5, N)):
        var = variations[i]
        print(f"\nVariation {i}:")
        print(f"  Angular momentum: {var['angular_momentum']:.6f}")
        print(f"  Num layers: {var['num_layers']}")
        #print(f"  Rel Eqp Err: {var['rel_equipotential_error']:.6f}")
        #print(f"  Total Energy: {var['total_energy']:.6f}")
        print(f"  First layer:")
        print(f"    Semiaxes: a={var['layers'][0]['semiaxes'][0]:.6f}, "
              f"b={var['layers'][0]['semiaxes'][1]:.6f}, "
              f"c={var['layers'][0]['semiaxes'][2]:.6f}")
        print(f"    Density: {var['layers'][0]['density']:.6f}")
    
    # ========================================================================
    # Example: Access specific layer data
    # ========================================================================
    
    # Get all first layers from all variations
    first_layers = variations['layers'][:, 0]
    
    # Get all semiaxes 'a' values from first layers
    all_a_values = first_layers['semiaxes'][:, 0]
    
    print(f"\nStatistics for first layer 'a' semiaxis:")
    print(f"  Mean: {np.mean(all_a_values):.6f}")
    print(f"  Std:  {np.std(all_a_values):.6f}")
    print(f"  Min:  {np.min(all_a_values):.6f}")
    print(f"  Max:  {np.max(all_a_values):.6f}")
    
    

if __name__ == '__main__':
    test_variations()
    