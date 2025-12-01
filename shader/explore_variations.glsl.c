#version 460 core

#include "shader/precision.glsl.c"
#include "shader/carlson.glsl.c"
#include "shader/random.glsl.c"

// ============================================================================
// Data Structures
// ============================================================================

struct Layer {
    BUFF_VEC3 semiaxes;        // offset 0
    BUFF_REAL average_radius;  // offset 32 (b/c, vec3 has 8 extra bytes for alignment)
    BUFF_REAL density;         // offset 40
};

struct Model {
    BUFF_REAL angular_momentum;
    uint num_layers;
    Layer layers[20];
    
    BUFF_REAL rel_equipotential_err;
    BUFF_REAL total_energy;
};

// ============================================================================
// Buffers
// ============================================================================

// Input: The template model
layout(std430, binding = 0) buffer InputModel 
{
    double template_angular_momentum;  // offset 0, 8 bytes
    uint template_num_layers;          // offset 8, 4 bytes
    uint _pad0;                         // offset 12, 4 bytes
    double _pad1[2];                    // offset 16, 16 bytes (align to 32)
    Layer template_layers[];           // offset 32
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
    for (uint i = 0; i < template_num_layers; i++) {
        variation.layers[i] = template_layers[i];
    }
    
    // ========================================================================
    // TODO: APPLY VARIATIONS HERE
    // ========================================================================
    for (uint i = 0; i < template_num_layers; i++)
    {
        float rand1 = pcg_float(rng);
        float rand2 = pcg_float(rng);
        float rand3 = pcg_float(rng);
        float avg = (rand1 + rand2 + rand3) / 3.;
        
        BUFF_REAL mul1 = BR(exp2( (1.-rand1) * float(annealing_temperature) ));
        BUFF_REAL mul2 = BR(exp2( (1.-rand2) * float(annealing_temperature) ));
        BUFF_REAL mul3 = BR(1.LF) / (mul1 * mul2);
        
//         variation.layers[i].semiaxes[0] *= mul1;
//         variation.layers[i].semiaxes[1] *= mul2;
//         variation.layers[i].semiaxes[2] *= mul3;
        //variation.layers[i].semiaxes[0] = template_layers[i].semiaxes[0] + 100.LF;
        //variation.layers[i].semiaxes[1] = template_layers[i].semiaxes[1] + 200.LF;
        //variation.layers[i].semiaxes[2] = template_layers[i].semiaxes[2] + 300.LF;
    }
    
    // ========================================================================
    // TODO: SCORE THE VARIATION HERE
    // ========================================================================
    // Example (you'll implement the actual scoring logic):
    // variation.score = score_model(variation);
    variation.rel_equipotential_err = BUFF_REAL(0.0);  // Placeholder
    variation.total_energy = BUFF_REAL(0.0);
    
    // Write output
    variations[idx] = variation;
    
}