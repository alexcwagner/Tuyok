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
        #('_pad_struct', np.float64, (2,))
    ])
    
    _model_dtype = np.dtype([
        ('angular_momentum', np.float64),
        ('num_layers', np.uint32),       
        ('_pad_to_32', np.uint8, (4,)),  
        ('layers', _layer_dtype, (20,)), 
        ('rel_equipotential_err', np.float64), 
        ('total_energy', np.float64),          
    ])
    
    _best_dtype = np.dtype([
        ('model', _model_dtype),
        ('best_idx', np.uint32),
        ('_pad', np.uint8, (4,))  # Alignment padding if needed
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
        
        # Header: 32 bytes total
        input_bytes = struct.pack('dI',  # double + uint = 12 bytes
                                  self["angular_momentum"],
                                  len(self["layers"])
                                  )
        input_bytes += b'\x00' * 20  # 20 bytes padding to reach offset 32      
        
        for layer in self['layers']:
            # Layer: 64 bytes each
            input_bytes += struct.pack(
                                'ddddd',
                                layer['abc'][0],
                                layer['abc'][1],
                                layer['abc'][2],
                                layer['r'], 
                                layer['density']
                            )

        return input_bytes
    
    def explore_variations(self, num_variants, temperature, top_k=None, seed=None):
        """
        Generate variations of the model and return the best ones.
        
        Args:
            num_variants: Number of variations to generate
            temperature: Annealing temperature for variation size
            top_k: Number of best results to return (default: return all)
            seed: Random seed (default: random)
        
        Returns:
            List of Model instances, sorted by rel_equipotential_err (best first)
        """
        if not self._shader_initialized:
            config = ShaderConfig.precision_config("double", "double")
            self.program = harness.create_program("shader/explore_variations.glsl.c", config)
            self._shader_initialized = True
    
        if seed is None:
            seed = random.randint(0, 0xFFFFFFFF)
    
        input_bytes = self.to_struct()
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
                dtype=Model._model_dtype,
                count=num_variants,
                mode="out"
            ),
            BufferSpec(
                binding=2,
                dtype=Model._best_dtype,
                count=1,
                mode="out"
            )
        ]
        print(f"USING SEED: {seed}")
        uniforms = [
            UniformSpec("num_variations", num_variants, "1ui"),
            UniformSpec("seed", seed, "1ui"),
            UniformSpec("annealing_temperature", temperature, "1d")
        ]   
        
        results = self.program.run(buffers, uniforms, num_invocations=num_variants)
        
        # Get raw numpy structured array
        time1 = time.time()
        raw_results = results[1]
        time2 = time.time()
        print(time1, time2)
        print(f"Generated {num_variants} variants in {(time2-time1)} seconds")
        
        # Sort by rel_equipotential_err (in-place, very fast on numpy arrays)
        time3 = time.time()
        raw_results.sort(order='rel_equipotential_err')
        time4 = time.time()
        print(time3, time4)
        print(f"Sorting took {(time4-time3)} seconds")
        
        # Determine how many to convert
        n_to_convert = top_k if top_k is not None else len(raw_results)
        
        # Only convert the top K
        time5 = time.time()
        top_models = [Model.from_struct(v) for v in raw_results[:n_to_convert]]
        time6 = time.time()
        print(time5, time6)
        print(f"Conversion took {(time6-time5)} seconds")
        
        best_model = Model.from_struct(results[2][0]['model'])
        #best_idx = results[2][0]['best_idx']
        
        return top_models, best_model

if __name__ == '__main__':
    
    model = Model({
        'angular_momentum': 0.,
        'layers': [
            {
                'abc': (1., 1., 1.),
                'density': 1.,
            }
        ]
    })
    
    num_variants = 1000000
    temperature = 0.1
    top_k = 1000
    results, best = model.explore_variations(num_variants, temperature, top_k=top_k, seed=12345)
    print(json.dumps(best, indent=4))
    
    ree = best['rel_equipotential_err']
    
    for idx, result in enumerate(results):
        if result['rel_equipotential_err'] == ree:
            print(f"so-called 'best' found at position {idx}")
            break
    else:
        print("so-called 'best' not found in top {top_k} results")
        
    #print(json.dumps(results[:top_k], indent=4))
    