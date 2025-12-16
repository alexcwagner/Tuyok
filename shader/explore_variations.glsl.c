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

void compute_statistics(uint idx)
{
    bool valid = true;
    
    // Initialize error accumulator to zero
    variations[idx].rel_equipotential_err = BR(0.0LF);
    
    // compute Moment of Inertia
    CALC_REAL moi = R(0.LF);    
    for (uint layer_idx = 0; layer_idx < variations[idx].num_layers; layer_idx++)
    {
        //Layer layer = variations[idx].layers[layer_idx];
        moi += variations[idx].layers[layer_idx].density 
                * variations[idx].layers[layer_idx].a 
                * variations[idx].layers[layer_idx].b 
                * variations[idx].layers[layer_idx].c 
                * (variations[idx].layers[layer_idx].a * variations[idx].layers[layer_idx].a 
                    + variations[idx].layers[layer_idx].b * variations[idx].layers[layer_idx].b);
    }
    moi *= R(4.LF/15.LF) * PI;
    
    // compute Angular Velocity
    CALC_REAL ang_vel = variations[idx].angular_momentum / moi;
    
    // Iterate through the layers to get the points we want to calculate the potential at
    for (uint surf_layer_idx = 0; surf_layer_idx < variations[idx].num_layers; surf_layer_idx++)
    {
        //Layer surf_layer = variations[idx].layers[surf_layer_idx];
    
        // accumulate the effective potential at (a,0,0), (0,b,0), and (0,0,c)
        // start with the centrifugal contribution before iterating through layers
        // (Note: Chandrasekhar convention is that these are positive. I'd argue with
        // him, but sadly, he has passed on.)
        CALC_REAL pot_a = R(0.5LF) 
                        * ang_vel * ang_vel 
                        * variations[idx].layers[surf_layer_idx].a * variations[idx].layers[surf_layer_idx].a;
        CALC_REAL pot_b = R(0.5LF) 
                        * ang_vel * ang_vel 
                        * variations[idx].layers[surf_layer_idx].b * variations[idx].layers[surf_layer_idx].b;
        CALC_REAL pot_c = 0.LF;
         
        // Iterate through the layers to get the ellipsoid creating a potential at the points
        for (uint mass_layer_idx = 0; mass_layer_idx < variations[idx].num_layers; mass_layer_idx++)
        {
            //Layer mat_layer = variations[idx].layers[mass_layer_idx];
            
            if (surf_layer_idx <= mass_layer_idx)
            {
                // The surface points will be inside or on the ellipsoid
                
                pot_a += variations[idx].layers[mass_layer_idx].density * 
                                potential_interior_x(
                                        variations[idx].layers[mass_layer_idx].a, 
                                        variations[idx].layers[mass_layer_idx].b, 
                                        variations[idx].layers[mass_layer_idx].c, 
                                        variations[idx].layers[surf_layer_idx].a);
                pot_b += variations[idx].layers[mass_layer_idx].density *
                                potential_interior_y(
                                        variations[idx].layers[mass_layer_idx].a, 
                                        variations[idx].layers[mass_layer_idx].b, 
                                        variations[idx].layers[mass_layer_idx].c, 
                                        variations[idx].layers[surf_layer_idx].b);
                pot_c += variations[idx].layers[mass_layer_idx].density * 
                                potential_interior_z(
                                        variations[idx].layers[mass_layer_idx].a, 
                                        variations[idx].layers[mass_layer_idx].b, 
                                        variations[idx].layers[mass_layer_idx].c, 
                                        variations[idx].layers[surf_layer_idx].c);
                
            }
            else
            {
                // The surface points will be outside the ellipsoid
                
                // check for bad overlap
                valid = valid 
                        && (variations[idx].layers[surf_layer_idx].a > variations[idx].layers[mass_layer_idx].a) 
                        && (variations[idx].layers[surf_layer_idx].b > variations[idx].layers[mass_layer_idx].b) 
                        && (variations[idx].layers[surf_layer_idx].c > variations[idx].layers[mass_layer_idx].c);
                
                pot_a += variations[idx].layers[mass_layer_idx].density * 
                                potential_exterior_x(
                                        variations[idx].layers[mass_layer_idx].a, 
                                        variations[idx].layers[mass_layer_idx].b, 
                                        variations[idx].layers[mass_layer_idx].c, 
                                        variations[idx].layers[surf_layer_idx].a);
                pot_b += variations[idx].layers[mass_layer_idx].density *
                                potential_exterior_y(
                                        variations[idx].layers[mass_layer_idx].a, 
                                        variations[idx].layers[mass_layer_idx].b, 
                                        variations[idx].layers[mass_layer_idx].c, 
                                        variations[idx].layers[surf_layer_idx].b);
                pot_c += variations[idx].layers[mass_layer_idx].density * 
                                potential_exterior_z(
                                        variations[idx].layers[mass_layer_idx].a, 
                                        variations[idx].layers[mass_layer_idx].b, 
                                        variations[idx].layers[mass_layer_idx].c, 
                                        variations[idx].layers[surf_layer_idx].c);
            }
        }
        
        CALC_REAL max_pot = max(pot_a, max(pot_b, pot_c));
        CALC_REAL min_pot = min(pot_a, min(pot_b, pot_c));
      
        variations[idx].rel_equipotential_err += (max_pot - min_pot) / min_pot;  
    }
    
    variations[idx].rel_equipotential_err = valid ? variations[idx].rel_equipotential_err / variations[idx].num_layers : BR(1e30LF);
    variations[idx].total_energy = BUFF_REAL(0.0);
    
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
        compute_statistics(idx);
        
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