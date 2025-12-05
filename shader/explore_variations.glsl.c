#version 460 core

#include "shader/precision.glsl.c"
#include "shader/potential.glsl.c"
#include "shader/random.glsl.c"

// ============================================================================
// Data Structures
// ============================================================================

struct Layer {
    BUFF_REAL a;
    BUFF_REAL b;
    BUFF_REAL c;
    BUFF_REAL volumetric_radius;
    BUFF_REAL density;
};

struct Model {
    BUFF_REAL angular_momentum;  // offset 0
    uint num_layers;             // offset 8
    // implicit 20 bytes padding here
    Layer layers[20];            // offset 32
    
    BUFF_REAL rel_equipotential_err;  // offset 1312
    BUFF_REAL total_energy;           // offset 1320
    // implicit 16 bytes padding here
};

// ============================================================================
// Buffers
// ============================================================================

// Input: The template model (flexible array, so slightly different layout)
layout(std430, binding = 0) buffer InputModel 
{
    double template_angular_momentum;  // offset 0, 8 bytes
    uint template_num_layers;          // offset 8, 4 bytes
    uint _pad0;                        // offset 12, 4 bytes (explicit)
    double _pad1[2];                   // offset 16, 16 bytes (to align to 32)
    Layer template_layers[];           // offset 32, flexible array
};

// Output: N variations of the model
layout(std430, binding = 1) buffer OutputModels {
    Model variations[];
};

// Add a new buffer for the global best
layout(std430, binding = 2) buffer BestModel {
    Model global_best;
    uint best_idx;
};

// ============================================================================
// Uniforms
// ============================================================================

uniform double annealing_temperature;
uniform uint num_variations;  // N
uniform uint seed;


void compute_statistics(inout Model model)
{
    bool valid = true;
    
    // compute Moment of Inertia
    CALC_REAL moi = R(0.LF);    
    for (uint idx = 0; idx < model.num_layers; idx++)
    {
        Layer layer = model.layers[idx];
        moi += layer.density * layer.a * layer.b * layer.c * (layer.a * layer.a + layer.b * layer.b);
    }
    moi *= R(4.LF/15.LF) * PI;
    
    // compute Angular Velocity
    CALC_REAL ang_vel = model.angular_momentum / moi;
    
    // Iterate through the layers to get the points we want to calculate the potential at
    for (uint s_idx = 0; s_idx < model.num_layers; s_idx++)
    {
        Layer surf_layer = model.layers[s_idx];
    
        // accumulate the effective potential at (a,0,0), (0,b,0), and (0,0,c)
        // start with the centrifugal contribution before iterating through layers
        CALC_REAL pot_a = R(-0.5LF) * ang_vel * ang_vel * surf_layer.a * surf_layer.a;
        CALC_REAL pot_b = R(-0.5LF) * ang_vel * ang_vel * surf_layer.b * surf_layer.b;
        CALC_REAL pot_c = 0.LF;
         
        // Iterate through the layers to get the ellipsoid creating a potential at the points
        for (uint m_idx = 0; m_idx < model.num_layers; m_idx++)
        {
            Layer mat_layer = model.layers[m_idx];
            
            if (s_idx <= m_idx)
            {
                // The surface points will be inside or on the ellipsoid
                
                pot_a += mat_layer.density * 
                                potential_interior_x(mat_layer.a, mat_layer.b, mat_layer.c, surf_layer.a);
                pot_b += mat_layer.density *
                                potential_interior_y(mat_layer.a, mat_layer.b, mat_layer.c, surf_layer.b);
                pot_c += mat_layer.density * 
                                potential_interior_z(mat_layer.a, mat_layer.b, mat_layer.c, surf_layer.c);
                
            }
            else
            {
                // The surface points will be outside the ellipsoid
                
                // check for bad overlap
                valid = valid 
                        && (surf_layer.a > mat_layer.a) 
                        && (surf_layer.b > mat_layer.b) 
                        && (surf_layer.c > mat_layer.c);
                
                pot_a += mat_layer.density * 
                                potential_exterior_x(mat_layer.a, mat_layer.b, mat_layer.c, surf_layer.a);
                pot_b += mat_layer.density *
                                potential_exterior_y(mat_layer.a, mat_layer.b, mat_layer.c, surf_layer.b);
                pot_c += mat_layer.density * 
                                potential_exterior_z(mat_layer.a, mat_layer.b, mat_layer.c, surf_layer.c);
                
            }
        }
        
        CALC_REAL max_pot = max(pot_a, max(pot_b, pot_c));
        CALC_REAL min_pot = min(pot_a, min(pot_b, pot_c));
      
        model.rel_equipotential_err += (max_pot - min_pot) / min_pot;  
    }
    
    
    
    
    //model.rel_equipotential_err = ang_vel;
    
    model.rel_equipotential_err = valid ? model.rel_equipotential_err / model.num_layers : BR(1e30LF);
    
    return;
}



// ============================================================================
// Main Compute Shader
// ============================================================================

layout(local_size_x = 256) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    // Guard against excess threads
    if (idx >= num_variations) {
        return;
    }
    
    // Initialize RNG for this thread
    PCGState rng;
    initPCG(rng, seed + idx, idx);
    
    // Create a variation based on the template
    Model variation;
    variation.num_layers = template_num_layers;
    variation.angular_momentum = template_angular_momentum;
    
    // Copy template layers
    //for (uint i = 0; i < template_num_layers; i++) {
    //    variation.layers[i] = template_layers[i];
    //}
    
    // ========================================================================
    // APPLY VARIATIONS
    // ========================================================================
    for (uint i = 0; i < template_num_layers; i++)
    {
        variation.layers[i].volumetric_radius = template_layers[i].volumetric_radius;
        variation.layers[i].density = template_layers[i].density;
        
        float rand1 = pcg_float(rng);
        float rand2 = pcg_float(rng);
        float rand3 = pcg_float(rng);
        
        BUFF_REAL mul1 = BR(exp2( (rand1 - 0.5) * float(annealing_temperature) ));
        BUFF_REAL mul2 = BR(exp2( (rand2 - 0.5) * float(annealing_temperature) ));
        BUFF_REAL mul3 = BR(1.LF) / (mul1 * mul2);  // Preserve volume
        
        variation.layers[i].a = template_layers[i].a * mul1;
        variation.layers[i].b = template_layers[i].b * mul2;
        variation.layers[i].c = template_layers[i].c * mul3;
    }
    
    // ========================================================================
    // SCORE THE VARIATION (placeholder)
    // ========================================================================
    compute_statistics(variation);
    
    
    variation.total_energy = BUFF_REAL(0.0);
    
    // Write output
    variations[idx] = variation;
    
   
    if (idx == 0) 
    {
        // First thread initializes
        global_best = variations[0];
        best_idx = 0;
    }
    barrier();

    // Simple (but racy) reduction - works okay for "good enough" best
    if (idx < num_variations) 
    {
        if (variations[idx].rel_equipotential_err < global_best.rel_equipotential_err) 
        {
            global_best = variations[idx];
            best_idx = idx;
        }
    }    
}