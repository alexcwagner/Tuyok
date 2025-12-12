# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 18:19:59 2025

@author: alexc
"""

from compute_harness import GLSLComputeHarness, ShaderConfig, BufferSpec, UniformSpec
import numpy as np
import struct
import random
import json
import time

harness = GLSLComputeHarness()

class Model(dict):
    
    _layer_dtype = np.dtype([
        ('a', np.float64),
        ('b', np.float64),
        ('c', np.float64),
        ('r', np.float64),
        ('density', np.float64),         
    ])
    
    _model_dtype = np.dtype([
        ('angular_momentum', np.float64),  # offset 0
        ('num_layers', np.uint32),         # offset 8
        ('_pad_to_16', np.uint32),         # offset 12 (4 bytes padding for std430 alignment)
        ('layers', _layer_dtype, (20,)),   # offset 16
        ('rel_equipotential_err', np.float64),  # offset 816
        ('total_energy', np.float64),      # offset 824
    ])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recalculate()
        
        self._shader_initialized = False
        
    def _recalculate(self):
        for layer in self['layers']:
            if 'abc' not in layer:
                if 'r' not in layer:
                    raise ValueError('''Model layer must contain 'abc' or 'r''')
                layer['abc'] = (layer['r'], layer['r'], layer['r'])
            else:
                r = np.cbrt(layer['abc'][0] * layer['abc'][1] * layer['abc'][2])
                if 'r' not in layer:
                    layer['r'] = r
                else:
                    if not np.isclose(r, layer['r'], rtol=1e-8):
                        raise ValueError('''If model layer contains both 'abc' and 'r', they must be consistent''')
                        
    @classmethod
    def from_struct(cls, s):
        num_layers = s['num_layers']
        model = {
            'angular_momentum': s['angular_momentum'],
            'layers': [
                {
                    'abc': [layer['a'], layer['b'], layer['c']],
                    'r': layer['r'],
                    'density': layer['density']
                } for layer in s['layers'][:num_layers]
            ],
            'rel_equipotential_err': s['rel_equipotential_err'],
            'total_energy': s['total_energy']
        }
        return model
    
    def to_struct(self):
        
        # Header: 16 bytes total (std430 layout)
        input_bytes = struct.pack('dI',  # double + uint = 12 bytes
                                  self["angular_momentum"],
                                  len(self["layers"])
                                  )
        input_bytes += b'\x00' * 4  # 4 bytes padding to reach offset 16 (std430 alignment)
        
        # Layers: 40 bytes each × 20 = 800 bytes (offsets 16-815)
        for i in range(20):
            if i < len(self['layers']):
                layer = self['layers'][i]
                input_bytes += struct.pack(
                                    'ddddd',  # 5 doubles = 40 bytes
                                    layer['abc'][0],
                                    layer['abc'][1],
                                    layer['abc'][2],
                                    layer['r'], 
                                    layer['density']
                                )
            else:
                # Pad unused layer slots with zeros
                input_bytes += b'\x00' * 40
        
        # Output fields: 16 bytes total (offsets 816-831)
        rel_equipotential_err = self.get('rel_equipotential_err', 0.0)
        total_energy = self.get('total_energy', 0.0)
        input_bytes += struct.pack('dd',  # 2 doubles = 16 bytes
                                   rel_equipotential_err,
                                   total_energy)

        return input_bytes
    
    def explore_variations(self, num_variants, temperature, top_k=None, seed=None):
        """
        Generate variations of the model and return the best ones.
        
        Args:
            num_variants: Number of variations to generate
            temperature: Annealing temperature for variation size
            top_k: Number of best results to return (default: 1)
            seed: Random seed (default: random)
        
        Returns:
            best_model: The single best Model found
            top_models: List of top_k Model instances (if top_k > 1)
        """
        if top_k is None:
            top_k = 1
            
        if not self._shader_initialized:
            config = ShaderConfig.precision_config("double", "double")
            self.program = harness.create_program("shader/explore_variations.glsl.c", config)
            self._shader_initialized = True
    
        if seed is None:
            seed = random.randint(0, 0xFFFFFFFF)
    
        input_bytes = self.to_struct()
        input_array = np.frombuffer(input_bytes, dtype=np.uint8)
        
        # Calculate number of workgroups
        local_size = 256
        num_workgroups = (num_variants + local_size - 1) // local_size
    
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
                dtype=Model._model_dtype,
                count=num_variants,
                mode="out"
            ),
            BufferSpec(
                binding=2,
                dtype=Model._model_dtype,
                count=num_workgroups,
                mode="out"
            ),
            BufferSpec(
                binding=3,
                dtype=np.float64,
                count=num_workgroups,
                mode="out"
            )
        ]
        
        print(f"USING SEED: {seed}")
        uniforms = [
            UniformSpec("num_variations", num_variants, "1ui"),
            UniformSpec("seed", seed, "1ui"),
            UniformSpec("annealing_temperature", temperature, "1d")
        ]   
        
        time_start = time.time()
        results = self.program.run(buffers, uniforms, num_invocations=num_variants)
        time_compute = time.time()
        print(f"GPU compute: {(time_compute - time_start):.3f} seconds")
        
        # Get workgroup bests
        workgroup_models = results[2]
        workgroup_scores = results[3]
        
        # Find the best among workgroup bests (tiny array, fast on CPU)
        best_workgroup_idx = np.argmin(workgroup_scores)
        best_model = Model.from_struct(workgroup_models[best_workgroup_idx])
        best_score = workgroup_scores[best_workgroup_idx]
        
        time_best = time.time()
        print(f"Find best: {(time_best - time_compute):.3f} seconds")
        print(f"Best score: {best_score:.6e} (from workgroup {best_workgroup_idx})")
        
        # If user wants top_k > 1, sort full results
        if top_k > 1:
            raw_results = results[1]
            raw_results.sort(order='rel_equipotential_err')
            top_models = [Model.from_struct(v) for v in raw_results[:top_k]]
            time_sort = time.time()
            print(f"Sort and convert top {top_k}: {(time_sort - time_best):.3f} seconds")
            return best_model, top_models
        else:
            return best_model, [best_model]

if __name__ == '__main__':
    
    model = Model({
        'angular_momentum': 0.,
        'layers': [
            {
                'abc': (1., 1., 1.),
                'density': 1.,
            },
            {
                'abc': (2., 2., 2.),
                'density': 1.,
            }
        ]
    })
    
    num_variants = 1000000
    temperature = 0.1
    top_k = 1000
    
    best, top_models = model.explore_variations(num_variants, temperature, top_k=top_k, seed=12345)
    
    print("\n" + "="*60)
    print("BEST MODEL:")
    print(json.dumps(best, indent=4))
    
    # Verify best is actually in the top results
    ree = best['rel_equipotential_err']
    for idx, result in enumerate(top_models):
        if result['rel_equipotential_err'] == ree:
            print(f"\nBest model found at position {idx} in top {top_k}")
            break
    else:
        print(f"\nWARNING: Best model not found in top {top_k} results!")
