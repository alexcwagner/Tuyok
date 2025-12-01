# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:19:59 2025

@author: alexc
"""

from compute_harness import GLSLComputeHarness, ShaderConfig, BufferSpec, UniformSpec
import numpy as np
import time
import struct

harness = GLSLComputeHarness()


def test_variations():
    
    # Test with double/double precision
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/explore_variations.glsl.c", config)
    
    N = 1_000
    
    angular_momentum = 4.01
    layers = [
        {
            'semiaxes': (0.99, 1., 1.01),
            'average_radius': 1.,
            'density': 1.05,
        },
        {
            'semiaxes': (1.98, 2., 2.02),
            'average_radius': 2.,
            'density': 2.10,
        },
        {
            'semiaxes': (2.97, 3., 3.03),
            'average_radius': 3.,
            'density': 3.15,
        }
    ]
   
    # ========================================================================
    # Pack input buffer to match GLSL std430 layout
    # ========================================================================
    
    # Header: double + uint + uint = 16 bytes
    # But layers need 32-byte alignment for dvec3, so we need padding to 32
    input_bytes = struct.pack('dII',  # 16 bytes
                              angular_momentum,
                              len(layers),
                              0)  # padding
    # Add 16 more bytes of padding to reach 32-byte alignment for first layer
    input_bytes += struct.pack('dd', 0.0, 0.0)  # 16 bytes padding -> total 32

    for layer in layers:
        # Layer in std430 with dvec3:
        #   dvec3 semiaxes:      offset 0,  24 bytes data + 8 bytes padding = 32
        #   double avg_radius:   offset 32, 8 bytes
        #   double density:      offset 40, 8 bytes
        #   Total: 48 bytes per layer
        input_bytes += struct.pack('dddddd',  # 48 bytes
                               layer['semiaxes'][0],
                               layer['semiaxes'][1],
                               layer['semiaxes'][2],
                               0.0,  # padding after dvec3
                               layer['average_radius'],
                               layer['density'])

    input_array = np.frombuffer(input_bytes, dtype=np.uint8)
    
    print(f"Input buffer size: {len(input_array)} bytes")
    
    # ========================================================================
    # Define output buffer dtype to match GLSL std430 layout
    # ========================================================================
    
    # Layer: 48 bytes
    layer_dtype = np.dtype([
        ('semiaxes', np.float64, (3,)),   # 24 bytes
        ('_pad0', np.float64),            # 8 bytes padding (dvec3 -> 32)
        ('average_radius', np.float64),   # 8 bytes
        ('density', np.float64)           # 8 bytes
    ])  # Total: 48 bytes
    
    # Model struct in GLSL:
    #   double angular_momentum   offset 0   (8 bytes)
    #   uint num_layers           offset 8   (4 bytes)
    #   <padding>                 offset 12  (4 bytes to align to 8)
    #   <padding>                 offset 16  (16 bytes to align layers to 32)
    #   Layer layers[20]          offset 32  (20 * 48 = 960 bytes)
    #   double rel_equi_err       offset 992 (8 bytes)
    #   double total_energy       offset 1000 (8 bytes)
    # Total: 1008 bytes
    
    model_dtype = np.dtype([
        ('angular_momentum', np.float64),      # 8 bytes, offset 0
        ('num_layers', np.uint32),             # 4 bytes, offset 8
        ('_pad0', np.uint32),                  # 4 bytes, offset 12
        ('_pad1', np.float64, (2,)),           # 16 bytes padding, offset 16
        ('layers', layer_dtype, (20,)),        # 960 bytes, offset 32
        ('rel_equipotential_err', np.float64), # 8 bytes, offset 992
        ('total_energy', np.float64)           # 8 bytes, offset 1000
    ])  # Total: 1008 bytes
    
    print(f"Model dtype size: {model_dtype.itemsize} bytes")
    print(f"Layer dtype size: {layer_dtype.itemsize} bytes")
    
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
        UniformSpec("annealing_temperature", 1.0, "1d")
    ]    

    print(f"Generating {N} variations...")
    results = program.run(buffers, uniforms, num_invocations=N)
    
    variations = results[1]
    
    # ========================================================================
    # Examine results
    # ========================================================================
    
    print(f"\nFirst 5 variations:")
    for i in range(min(5, N)):
        var = variations[i]
        #print(var)
        
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
            semiaxes = layer['semiaxes'][0], layer['semiaxes'][1], layer['semiaxes'][2], 
            avg_radius = layer['average_radius']
            density = layer['density']
            print(f"  Layer {j}:")
            print(f"    Semiaxes: "
                  f"a={semiaxes[0]:.6f}, "
                  f"b={semiaxes[1]:.6f}, "
                  f"c={semiaxes[2]:.6f}")
            print(f"    Avg radius: {avg_radius:.6f}")
            print(f"    Density: {density:.6f}")
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    first_layers = variations['layers'][:, 0]
    all_a_values = first_layers['semiaxes'][:, 0]
    
    print(f"\nStatistics for first layer 'a' semiaxis:")
    print(f"  Mean: {np.mean(all_a_values):.6f}")
    print(f"  Std:  {np.std(all_a_values):.6f}")
    print(f"  Min:  {np.min(all_a_values):.6f}")
    print(f"  Max:  {np.max(all_a_values):.6f}")


if __name__ == '__main__':
    test_variations()