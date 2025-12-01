#version 460 core

#include "shader/precision.glsl.c"
#include "shader/carlson.glsl.c"
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

// ============================================================================
// Uniforms
// ============================================================================

uniform double annealing_temperature;
uniform uint num_variations;  // N
uniform uint seed;

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
        
        variation.layers[i].a = template_layers[i].a; // * mul1;
        variation.layers[i].b = template_layers[i].b; // * mul2;
        variation.layers[i].c = template_layers[i].c; // * mul3;
        
    }
    
    // ========================================================================
    // SCORE THE VARIATION (placeholder)
    // ========================================================================
    variation.rel_equipotential_err = BUFF_REAL(0.0);
    variation.total_energy = BUFF_REAL(0.0);
    
    // Write output
    variations[idx] = variation;
}