#version 460 core
#extension GL_ARB_gpu_shader_fp64 : require  // <-- ADD THIS

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
    BUFF_REAL r;
    BUFF_REAL density;
};

struct Model {
    BUFF_REAL angular_momentum;  // offset 0, 8 bytes
    uint num_layers;             // offset 8, 4 bytes
    // implicit 4 bytes padding to align array to 16-byte boundary
    Layer layers[20];            // offset 16, 800 bytes (20 × 40)
    
    BUFF_REAL rel_equipotential_err;  // offset 816, 8 bytes
    BUFF_REAL total_energy;           // offset 824, 8 bytes
    // Total size: 832 bytes
};

// struct Layer {
//     double a;
//     double b;
//     double c;
//     double r;
//     double density;
// };
// 
// struct Model {
//     double angular_momentum;  // offset 0, 8 bytes
//     uint num_layers;             // offset 8, 4 bytes
//     // implicit 4 bytes padding to align array to 16-byte boundary
//     Layer layers[20];            // offset 16, 800 bytes (20 × 40)
//     
//     double rel_equipotential_err;  // offset 816, 8 bytes
//     double total_energy;           // offset 824, 8 bytes
//     // Total size: 832 bytes
// };


// ============================================================================
// Buffers
// ============================================================================

// Input: The template model (flexible array, so slightly different layout)
layout(std430, binding = 0) buffer InputModel 
{
    double template_angular_momentum;  // offset 0, 8 bytes
    uint template_num_layers;          // offset 8, 4 bytes
    uint _pad0;                        // offset 12, 4 bytes (explicit padding to 16)
    //Layer template_layers[];           // offset 16, flexible array
    Layer template_layers[20];           // offset 16, fixed array
};

// Output: N variations of the model
layout(std430, binding = 1) buffer OutputModels {
    Model variations[];
};

// Workgroup best models
layout(std430, binding = 2) buffer WorkgroupBests {
    Model workgroup_best_models[];
};

// Workgroup best scores
layout(std430, binding = 3) buffer WorkgroupBestScores {
    double workgroup_best_scores[];
};

// ============================================================================
// Uniforms
// ============================================================================

uniform double annealing_temperature;
uniform uint num_variations;  // N
uniform uint seed;

// ============================================================================
// Shared memory for workgroup reduction
// ============================================================================

shared uint local_best_idx;
shared BUFF_REAL local_best_score;

// ============================================================================
// Statistics computation
// ============================================================================

void compute_statistics(inout Model model)
{
    bool valid = true;
    
    // Initialize error accumulator to zero
    model.rel_equipotential_err = BR(0.0LF);
    
    // compute Moment of Inertia
    CALC_REAL moi = R(0.LF);    
    for (uint idx = 0; idx < model.num_layers; idx++)
    {
        //Layer layer = model.layers[idx];
        moi += model.layers[idx].density 
                * model.layers[idx].a 
                * model.layers[idx].b 
                * model.layers[idx].c 
                * (model.layers[idx].a * model.layers[idx].a 
                    + model.layers[idx].b * model.layers[idx].b);
    }
    moi *= R(4.LF/15.LF) * PI;
    
    // compute Angular Velocity
    CALC_REAL ang_vel = model.angular_momentum / moi;
    
    // Iterate through the layers to get the points we want to calculate the potential at
    for (uint s_idx = 0; s_idx < model.num_layers; s_idx++)
    {
        //Layer surf_layer = model.layers[s_idx];
    
        // accumulate the effective potential at (a,0,0), (0,b,0), and (0,0,c)
        // start with the centrifugal contribution before iterating through layers
        CALC_REAL pot_a = R(-0.5LF) 
                        * ang_vel * ang_vel 
                        * model.layers[s_idx].a * model.layers[s_idx].a;
        CALC_REAL pot_b = R(-0.5LF) 
                        * ang_vel * ang_vel 
                        * model.layers[s_idx].b * model.layers[s_idx].b;
        CALC_REAL pot_c = 0.LF;
         
        // Iterate through the layers to get the ellipsoid creating a potential at the points
        for (uint m_idx = 0; m_idx < model.num_layers; m_idx++)
        {
            //Layer mat_layer = model.layers[m_idx];
            
            if (s_idx <= m_idx)
            {
                // The surface points will be inside or on the ellipsoid
                
                pot_a += model.layers[m_idx].density * 
                                potential_interior_x(
                                        model.layers[m_idx].a, 
                                        model.layers[m_idx].b, 
                                        model.layers[m_idx].c, 
                                        model.layers[s_idx].a);
                pot_b += model.layers[m_idx].density *
                                potential_interior_y(
                                        model.layers[m_idx].a, 
                                        model.layers[m_idx].b, 
                                        model.layers[m_idx].c, 
                                        model.layers[s_idx].b);
                pot_c += model.layers[m_idx].density * 
                                potential_interior_z(
                                        model.layers[m_idx].a, 
                                        model.layers[m_idx].b, 
                                        model.layers[m_idx].c, 
                                        model.layers[s_idx].c);
                
            }
            else
            {
                // The surface points will be outside the ellipsoid
                
                // check for bad overlap
                valid = valid 
                        && (model.layers[s_idx].a > model.layers[m_idx].a) 
                        && (model.layers[s_idx].b > model.layers[m_idx].b) 
                        && (model.layers[s_idx].c > model.layers[m_idx].c);
                
                pot_a += model.layers[m_idx].density * 
                                potential_exterior_x(
                                        model.layers[m_idx].a, 
                                        model.layers[m_idx].b, 
                                        model.layers[m_idx].c, 
                                        model.layers[s_idx].a);
                pot_b += model.layers[m_idx].density *
                                potential_exterior_y(
                                        model.layers[m_idx].a, 
                                        model.layers[m_idx].b, 
                                        model.layers[m_idx].c, 
                                        model.layers[s_idx].b);
                pot_c += model.layers[m_idx].density * 
                                potential_exterior_z(
                                        model.layers[m_idx].a, 
                                        model.layers[m_idx].b, 
                                        model.layers[m_idx].c, 
                                        model.layers[s_idx].c);
            }
        }
        
        CALC_REAL max_pot = max(pot_a, max(pot_b, pot_c));
        CALC_REAL min_pot = min(pot_a, min(pot_b, pot_c));
      
        model.rel_equipotential_err += (max_pot - min_pot) / min_pot;  
    }
    
    model.rel_equipotential_err = valid ? model.rel_equipotential_err / model.num_layers : BR(1e30LF);
    
    return;
}

// ============================================================================
// Main Compute Shader
// ============================================================================

layout(local_size_x = 256) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint local_idx = gl_LocalInvocationID.x;
    uint workgroup_id = gl_WorkGroupID.x;
    
    // Initialize shared memory once per workgroup
    if (local_idx == 0) {
        local_best_idx = 0;
        local_best_score = BR(1e30LF);
    }
    barrier();
    
    // Guard against excess threads
    if (idx < num_variations) {
        
        // Initialize RNG for this thread
        PCGState rng;
        initPCG(rng, seed + idx, idx);
        
        // Create a variation based on the template
        //Model variation;
        variations[idx].num_layers = template_num_layers;
        variations[idx].angular_momentum = template_angular_momentum;
        
        // DIAGNOSTIC: Read BEFORE any assignment - store in a temporary field
        BUFF_REAL debug_read_a = template_layers[0].a;
        
        // ====================================================================
        // APPLY VARIATIONS
        // ====================================================================
        for (uint i = 0; i < template_num_layers; i++)
        {
            variations[idx].layers[i].r = template_layers[i].r;
            variations[idx].layers[i].density = template_layers[i].density;
            
            if (annealing_temperature == 0.0) 
            {                
                variations[idx].layers[i].a = template_layers[i].a;
                variations[idx].layers[i].b = template_layers[i].b;
                variations[idx].layers[i].c = template_layers[i].c;
            } 
            else 
            {
                BUFF_REAL mul1, mul2, mul3;
            
                float rand1 = pcg_float(rng);
                float rand2 = pcg_float(rng);
                float rand3 = pcg_float(rng);
                
                // exp2 only accepts float in GLSL
                mul1 = BR(exp2( (rand1 - 0.5) * float(annealing_temperature) ));
                mul2 = BR(exp2( (rand2 - 0.5) * float(annealing_temperature) ));
                mul3 = BR(1.LF) / (mul1 * mul2);  // Preserve volume
            
                variations[idx].layers[i].a = template_layers[i].a * mul1;
                variations[idx].layers[i].b = template_layers[i].b * mul2;
                variations[idx].layers[i].c = template_layers[i].c * mul3;
            }
        }
        
        // ====================================================================
        // SCORE THE VARIATION
        // ====================================================================
        compute_statistics(variations[idx]);
        
        // DIAGNOSTIC: Store what we read AND what we wrote
        if (idx == 0) {
            variations[idx].total_energy = double(debug_read_a);
            variations[idx].angular_momentum = double(variations[idx].layers[0].a);  // What we wrote back
        } else {
            variations[idx].total_energy = BUFF_REAL(0.0);
            variations[idx].angular_momentum = BUFF_REAL(0.0);  // Clear for other threads
        }
        
        // Write output
        //variations[idx] = variation;
    }
    
    // ========================================================================
    // WORKGROUP REDUCTION (find best within workgroup)
    // ========================================================================
    
    barrier();
    
    // Each thread competes sequentially (slow but correct)
    if (idx < num_variations) {
        BUFF_REAL my_score = variations[idx].rel_equipotential_err;
        
        for (uint i = 0; i < 256; i++) {
            if (local_idx == i) {
                if (my_score < local_best_score) {
                    local_best_score = my_score;
                    local_best_idx = idx;
                }
            }
            barrier();
        }
    }
    
    // First thread in workgroup writes the workgroup's best
    if (local_idx == 0) {
        workgroup_best_models[workgroup_id] = variations[local_best_idx];
        workgroup_best_scores[workgroup_id] = local_best_score;
    }
}