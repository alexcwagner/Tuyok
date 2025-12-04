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
        #('_pad_struct', np.uint8, (16,))      
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
    
    def explore_variations(self, num_variants, temperature, seed=None):
        if not self._shader_initialized:
            config = ShaderConfig.precision_config("double", "double")
            self.program = harness.create_program("shader/explore_variations.glsl.c", config)
            self._shader_initialized = True

        if seed is None:
            seed = random.randint(0, 0xFFFFFFFF)

        input_bytes = model.to_struct()
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
            )
        ]
        
        uniforms = [
            UniformSpec("num_variations", num_variants, "1ui"),
            UniformSpec("seed", seed, "1ui"),
            UniformSpec("annealing_temperature", temperature, "1d")
        ]   
        
        results = self.program.run(buffers, uniforms, num_invocations=num_variants)
        
        #return results[1]
        return [ Model.from_struct(v) for v in results[1]]

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
    
    results = model.explore_variations(10, 0.1, 12345)
    print(json.dumps(results, indent=4))