# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:19:59 2025

@author: alexc
"""

from compute_harness import GLSLComputeHarness, ShaderConfig, BufferSpec, UniformSpec
import numpy as np
import time

harness = GLSLComputeHarness()


def test_carlson_rj():
    from scipy.special import elliprj
    
    # Test with double/double precision
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/test_carlson_rj.glsl.c", config)
    
    N = 1_000_000
    
    # Define output buffer structure
    dtype = np.dtype([
        ('a', np.float64),
        ('b', np.float64),
        ('c', np.float64),
        ('p', np.float64),
        ('result', np.float64)
    ])
    
    # Define buffers and uniforms
    buffers = [
        BufferSpec(
            binding=0,
            dtype=dtype,
            count=N,
            mode="out"
        )
    ]
    
    uniforms = [
        UniformSpec("num_samples", N, "1ui"),
        UniformSpec("seed", 42, "1ui")
    ]
    
    # Run GPU version
    print(f"Computing {N} samples on GPU...")
    start_time = time.time()
    results = program.run(buffers, uniforms, num_invocations=N)
    gpu_elapsed = time.time() - start_time
    print(f"\033[1;32mGPU completed in {gpu_elapsed:.3f} seconds\033[m")
    
    # Get output from binding 0
    data = results[0]
    
    # Run SciPy version on same inputs (vectorized)
    print(f"Computing {N} samples with SciPy...")
    start_time = time.time()
    sci_ans = elliprj(data['a'], data['b'], data['c'], data['p'])
    scipy_elapsed = time.time() - start_time
    print(f"\033[1;32mSciPy completed in {scipy_elapsed:.3f} seconds\033[m")
    print(f"\033[1;36mSpeedup: {scipy_elapsed/gpu_elapsed:.2f}x\033[m")
    
    # Validate
    gpu_ans = data['result']
    
    # Calculate errors (avoid division by zero)
    valid_mask = sci_ans != 0
    rel_errors = np.abs((gpu_ans[valid_mask] - sci_ans[valid_mask]) / sci_ans[valid_mask])
    
    if len(rel_errors) > 0:
        worst_idx = np.argmax(rel_errors)
        worst_err = rel_errors[worst_idx]
        worst_precision = int(np.abs(np.round(np.log10(worst_err)))) if worst_err > 0 else 50
        
        # Find original index
        valid_indices = np.where(valid_mask)[0]
        original_idx = valid_indices[worst_idx]
        worst_case = data[original_idx]
        
        print(f"\nWorst precision: {worst_precision} decimal places")
        print(f"Worst case: RJ({worst_case['a']:.6g}, {worst_case['b']:.6g}, "
              f"{worst_case['c']:.6g}, {worst_case['p']:.6g})")
        print(f"  GPU:   {worst_case['result']:.15e}")
        print(f"  SciPy: {sci_ans[original_idx]:.15e}")
        print(f"  Error: {worst_err:.3e}")
    
    program.cleanup()


def test_carlson_rf():
    from scipy.special import elliprf
    
    # Test with double/double precision
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/test_carlson_rf.glsl.c", config)
    
    N = 1_000_000
    
    # Define output buffer structure
    dtype = np.dtype([
        ('a', np.float64),
        ('b', np.float64),
        ('c', np.float64),
        ('result', np.float64)
    ])
    
    # Define buffers and uniforms
    buffers = [
        BufferSpec(
            binding=0,
            dtype=dtype,
            count=N,
            mode="out"
        )
    ]
    
    uniforms = [
        UniformSpec("num_samples", N, "1ui"),
        UniformSpec("seed", 42, "1ui")
    ]
    
    # Run GPU version
    print(f"Computing {N} samples on GPU...")
    start_time = time.time()
    results = program.run(buffers, uniforms, num_invocations=N)
    gpu_elapsed = time.time() - start_time
    print(f"\033[1;32mGPU completed in {gpu_elapsed:.3f} seconds\033[m")
    
    # Get output from binding 0
    data = results[0]
    
    # Run SciPy version on same inputs (vectorized)
    print(f"Computing {N} samples with SciPy...")
    start_time = time.time()
    sci_ans = elliprf(data['a'], data['b'], data['c'])
    scipy_elapsed = time.time() - start_time
    print(f"\033[1;32mSciPy completed in {scipy_elapsed:.3f} seconds\033[m")
    print(f"\033[1;36mSpeedup: {scipy_elapsed/gpu_elapsed:.2f}x\033[m")
    
    # Validate
    gpu_ans = data['result']
    
    # Calculate errors
    valid_mask = sci_ans != 0
    rel_errors = np.abs((gpu_ans[valid_mask] - sci_ans[valid_mask]) / sci_ans[valid_mask])
    
    if len(rel_errors) > 0:
        worst_idx = np.argmax(rel_errors)
        worst_err = rel_errors[worst_idx]
        worst_precision = int(np.abs(np.round(np.log10(worst_err)))) if worst_err > 0 else 50
        
        valid_indices = np.where(valid_mask)[0]
        original_idx = valid_indices[worst_idx]
        worst_case = data[original_idx]
        
        print(f"\nWorst precision: {worst_precision} decimal places")
        print(f"Worst case: RF({worst_case['a']:.6g}, {worst_case['b']:.6g}, "
              f"{worst_case['c']:.6g})")
        print(f"  GPU:   {worst_case['result']:.15e}")
        print(f"  SciPy: {sci_ans[original_idx]:.15e}")
        print(f"  Error: {worst_err:.3e}")
    
    program.cleanup()


def test_carlson_rc():
    from scipy.special import elliprc
    
    # Test with double/double precision
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/test_carlson_rc.glsl.c", config)
    
    N = 1_000_000
    
    # Define output buffer structure
    dtype = np.dtype([
        ('a', np.float64),
        ('b', np.float64),
        ('result', np.float64)
    ])
    
    # Define buffers and uniforms
    buffers = [
        BufferSpec(
            binding=0,
            dtype=dtype,
            count=N,
            mode="out"
        )
    ]
    
    uniforms = [
        UniformSpec("num_samples", N, "1ui"),
        UniformSpec("seed", 42, "1ui")
    ]
    
    # Run GPU version
    print(f"Computing {N} samples on GPU...")
    start_time = time.time()
    results = program.run(buffers, uniforms, num_invocations=N)
    gpu_elapsed = time.time() - start_time
    print(f"\033[1;32mGPU completed in {gpu_elapsed:.3f} seconds\033[m")
    
    # Get output from binding 0
    data = results[0]
    
    # Run SciPy version on same inputs (vectorized)
    print(f"Computing {N} samples with SciPy...")
    start_time = time.time()
    sci_ans = elliprc(data['a'], data['b'])
    scipy_elapsed = time.time() - start_time
    print(f"\033[1;32mSciPy completed in {scipy_elapsed:.3f} seconds\033[m")
    print(f"\033[1;36mSpeedup: {scipy_elapsed/gpu_elapsed:.2f}x\033[m")
    
    # Validate
    gpu_ans = data['result']
    
    # Calculate errors
    valid_mask = sci_ans != 0
    rel_errors = np.abs((gpu_ans[valid_mask] - sci_ans[valid_mask]) / sci_ans[valid_mask])
    
    if len(rel_errors) > 0:
        worst_idx = np.argmax(rel_errors)
        worst_err = rel_errors[worst_idx]
        worst_precision = int(np.abs(np.round(np.log10(worst_err)))) if worst_err > 0 else 50
        
        valid_indices = np.where(valid_mask)[0]
        original_idx = valid_indices[worst_idx]
        worst_case = data[original_idx]
        
        print(f"\nWorst precision: {worst_precision} decimal places")
        print(f"Worst case: RC({worst_case['a']:.6g}, {worst_case['b']:.6g})")
        print(f"  GPU:   {worst_case['result']:.15e}")
        print(f"  SciPy: {sci_ans[original_idx]:.15e}")
        print(f"  Error: {worst_err:.3e}")
    
    program.cleanup()


def test_carlson_rd():
    from scipy.special import elliprd
    
    # Test with double/double precision
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/test_carlson_rd.glsl.c", config)
    
    N = 1_000_000
    
    # Define output buffer structure
    dtype = np.dtype([
        ('a', np.float64),
        ('b', np.float64),
        ('c', np.float64),
        ('result', np.float64)
    ])
    
    # Define buffers and uniforms
    buffers = [
        BufferSpec(
            binding=0,
            dtype=dtype,
            count=N,
            mode="out"
        )
    ]
    
    uniforms = [
        UniformSpec("num_samples", N, "1ui"),
        UniformSpec("seed", 42, "1ui")
    ]
    
    # Run GPU version
    print(f"Computing {N} samples on GPU...")
    start_time = time.time()
    results = program.run(buffers, uniforms, num_invocations=N)
    gpu_elapsed = time.time() - start_time
    print(f"\033[1;32mGPU completed in {gpu_elapsed:.3f} seconds\033[m")
    
    # Get output from binding 0
    data = results[0]
    
    # Run SciPy version on same inputs (vectorized)
    print(f"Computing {N} samples with SciPy...")
    start_time = time.time()
    sci_ans = elliprd(data['a'], data['b'], data['c'])
    scipy_elapsed = time.time() - start_time
    print(f"\033[1;32mSciPy completed in {scipy_elapsed:.3f} seconds\033[m")
    print(f"\033[1;36mSpeedup: {scipy_elapsed/gpu_elapsed:.2f}x\033[m")
    
    # Validate
    gpu_ans = data['result']
    
    # Calculate errors
    valid_mask = sci_ans != 0
    rel_errors = np.abs((gpu_ans[valid_mask] - sci_ans[valid_mask]) / sci_ans[valid_mask])
    
    if len(rel_errors) > 0:
        worst_idx = np.argmax(rel_errors)
        worst_err = rel_errors[worst_idx]
        worst_precision = int(np.abs(np.round(np.log10(worst_err)))) if worst_err > 0 else 50
        
        valid_indices = np.where(valid_mask)[0]
        original_idx = valid_indices[worst_idx]
        worst_case = data[original_idx]
        
        print(f"\nWorst precision: {worst_precision} decimal places")
        print(f"Worst case: RD({worst_case['a']:.6g}, {worst_case['b']:.6g}, "
              f"{worst_case['c']:.6g})")
        print(f"  GPU:   {worst_case['result']:.15e}")
        print(f"  SciPy: {sci_ans[original_idx]:.15e}")
        print(f"  Error: {worst_err:.3e}")
    
    program.cleanup()


if __name__ == '__main__':
    print("="*70)
    print("Testing Carlson RJ")
    print("="*70)
    test_carlson_rj()
    
    print("\n" + "="*70)
    print("Testing Carlson RF")
    print("="*70)
    test_carlson_rf()
    
    print("\n" + "="*70)
    print("Testing Carlson RC")
    print("="*70)
    test_carlson_rc()
    
    print("\n" + "="*70)
    print("Testing Carlson RD")
    print("="*70)
    test_carlson_rd()