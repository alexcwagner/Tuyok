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
    
    def dump_struct_hex(self, data_bytes=None):
        """
        Dump the struct as hex for debugging.
        If data_bytes is None, uses self.to_struct()
        """
        if data_bytes is None:
            data_bytes = self.to_struct()
        
        print(f"\nStruct dump ({len(data_bytes)} bytes):")
        print("=" * 70)
        
        # Header
        print("Header (16 bytes):")
        print(f"  0-7   angular_momentum: {data_bytes[0:8].hex()}")
        ang_mom = struct.unpack('d', data_bytes[0:8])[0]
        print(f"        = {ang_mom}")
        print(f"  8-11  num_layers: {data_bytes[8:12].hex()}")
        num_layers = struct.unpack('I', data_bytes[8:12])[0]
        print(f"        = {num_layers}")
        print(f"  12-15 padding: {data_bytes[12:16].hex()}")
        
        # Layers (show first 2 and last 1)
        print("\nLayers (800 bytes, showing first 2):")
        for i in range(min(2, num_layers)):
            offset = 16 + i * 40
            layer_bytes = data_bytes[offset:offset+40]
            a, b, c, r, density = struct.unpack('ddddd', layer_bytes)
            print(f"  Layer {i} (offset {offset}):")
            print(f"    a={a}, b={b}, c={c}, r={r}, density={density}")
        
        # Output fields
        print("\nOutput fields (16 bytes):")
        print(f"  816-823 rel_equipotential_err: {data_bytes[816:824].hex()}")
        ree = struct.unpack('d', data_bytes[816:824])[0]
        print(f"          = {ree}")
        print(f"  824-831 total_energy: {data_bytes[824:832].hex()}")
        te = struct.unpack('d', data_bytes[824:832])[0]
        print(f"          = {te}")
        print("=" * 70)
    
    def dump_numpy_struct(self, numpy_array):
        """
        Dump a numpy structured array (as returned from GPU) for debugging.
        """
        print(f"\nNumPy struct dump:")
        print("=" * 70)
        print(f"  angular_momentum: {numpy_array['angular_momentum']}")
        print(f"  num_layers: {numpy_array['num_layers']}")
        print(f"  layers[0]: a={numpy_array['layers'][0][0]['a']}, "
              f"b={numpy_array['layers'][0][0]['b']}, "
              f"c={numpy_array['layers'][0][0]['c']}")
        if numpy_array['num_layers'] > 1:
            print(f"  layers[1]: a={numpy_array['layers'][0][1]['a']}, "
                  f"b={numpy_array['layers'][0][1]['b']}, "
                  f"c={numpy_array['layers'][0][1]['c']}")
        print(f"  rel_equipotential_err: {numpy_array['rel_equipotential_err']}")
        print(f"  total_energy: {numpy_array['total_energy']}")
        print("=" * 70)
        
    def dump_raw_bytes(self, raw_bytes, label="Raw bytes"):
        """
        Dump raw bytes in hex with offsets.
        """
        print(f"\n{label} ({len(raw_bytes)} bytes):")
        print("=" * 70)
        for i in range(0, min(len(raw_bytes), 832), 16):
            hex_str = ' '.join(f'{b:02x}' for b in raw_bytes[i:i+16])
            print(f"  {i:4d}: {hex_str}")
            if i == 0:
                print("        ^ header")
            elif i == 16:
                print("        ^ layers start")
            elif i == 816:
                print("        ^ output fields")
        print("=" * 70)
    
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
    
        #self.program._dump_source()
    
    
    
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
        
        # DIAGNOSTIC: Check raw bytes of first result
        if num_variants > 0:
            raw_result = results[1][0]
            print(f"\n=== DIAGNOSTIC: First result raw data ===")
            print(f"  Temperature: {temperature}")
            print(f"  GPU read template_layers[0].a as: {raw_result['total_energy']}")
            print(f"  Input 'a' value: {self['layers'][0]['abc'][0]}")
            print(f"  Output 'a' value: {raw_result['layers'][0]['a']}")
            print(f"  Match? {abs(raw_result['layers'][0]['a'] - self['layers'][0]['abc'][0]) < 1e-10}")
            
            if abs(raw_result['total_energy'] - self['layers'][0]['abc'][0]) > 1e-10:
                print(f"  ⚠️  GPU READ from template_layers has precision loss!")
                print(f"     Expected: {self['layers'][0]['abc'][0]}")
                print(f"     GPU read: {raw_result['total_energy']}")
            
            # Check the raw bytes
            raw_bytes = raw_result.tobytes()
            layer0_offset = 16  # offset to first layer
            a_bytes = raw_bytes[layer0_offset:layer0_offset+8]
            print(f"  Raw bytes for output 'a': {a_bytes.hex()}")
            print(f"==========================================\n")
        
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
        "angular_momentum": 1.2874789457385492,
        "layers": [
            {
                "abc": [
                    1.3,
                    1.105575570700132,
                    0.6957740290368712
                ],
                "r": 1.0,
                "density": 1.1
            }
        ]
    })
    
    num_variants = 1
    temperature = 0.0
    top_k = 1
    
    best, top_models = model.explore_variations(num_variants, temperature, top_k=top_k, seed=12345)
    
    print("\n" + "="*60)
    print("BEST MODEL:")
    print(json.dumps(best, indent=4))
    
    # Verify best is actually in the top results
    ree = best['rel_equipotential_err']
    for idx, result in enumerate(top_models):
        
        if result['rel_equipotential_err'] == ree:
            print(f"\nBest model found at position {idx} in top {top_k}")
            print(json.dumps(result, indent=4))
            break
    else:
        print(f"\nWARNING: Best model not found in top {top_k} results!")