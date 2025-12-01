#version 460 core

#include "shader/precision.glsl.c"
#include "shader/carlson.glsl.c"
#include "shader/random.glsl.c"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

uniform uint num_samples;
uniform uint seed;

struct rj_sample
{
    BUFF_REAL a;
    BUFF_REAL b;
    BUFF_REAL c;
    BUFF_REAL result;
};

layout(std430, binding = 0) buffer 
OutBuffer
{ 
    rj_sample evaluation[]; 
};

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_samples) {
        return; // guard threads beyond N
    }
    
    PCGState rng;
    initPCG(rng, seed + idx, idx);
    
    evaluation[idx].a = R(pcg_float(rng));
    evaluation[idx].b = R(pcg_float(rng));
    evaluation[idx].c = R(pcg_float(rng));
        
    evaluation[idx].result = BUFF_REAL(carlson_rd(
            R(evaluation[idx].a),
            R(evaluation[idx].b),
            R(evaluation[idx].c)
            ));
    
}
