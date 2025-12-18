# -*- coding: utf-8 -*-
import sys
import Model
import json


def main():
    if len(sys.argv) < 2:    
        print("No filename provided.")
        return
    
    filename = sys.argv[1]
    with open(filename, 'r') as fp:
        model = Model.Model(json.load(fp))
        
    print(json.dumps(model, indent=4))


    num_variants = 1000000
    temperature = 1.0
    top_k = 50
    seed = None
    
    best, top_models = model.explore_variations(num_variants, temperature, top_k=top_k, seed=seed)

    print(json.dumps(best, indent=4))
    print(" #      a        b        c     err (x1e6)   total energy")
    print("=== ======== ======== ======== ============ ==============")
    for idx, candidate in enumerate(top_models):
        a, b, c = candidate['layers'][0]['abc']
        if b>a:
            a, b = b, a
        err = candidate['rel_equipotential_err']
        energy = candidate['total_energy']
        print(f"{idx+1:3d} {a:8.5f} {b:8.5f} {c:8.5f}  {err*1e6:8.3f}     {energy:8.5f}")
    
    
if __name__ == '__main__':
    main()
    