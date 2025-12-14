#version 460 core

#include "shader/precision.glsl.c"
#include "shader/potential.glsl.c"
#include "shader/random.glsl.c"

struct Layer {
    BUFF_REAL a;
    BUFF_REAL b;
    BUFF_REAL c;
    BUFF_REAL volumetric_radius;
    BUFF_REAL density;
};

// Input buffer - exactly like in explore_variations
layout(std430, binding = 0) buffer InputModel 
{
    double template_angular_momentum;
    uint template_num_layers;
    uint _pad0;
    Layer template_layers[];
};

// Output - just echo what we read
layout(std430, binding = 1) buffer Output {
    double out_values[];
};

layout(local_size_x = 1) in;

void main() {
    out_values[0] = template_angular_momentum;
    out_values[1] = double(template_layers[0].a);
    out_values[2] = double(template_layers[0].b);
    out_values[3] = double(template_layers[0].c);
    out_values[4] = double(template_layers[0].volumetric_radius);
    out_values[5] = double(template_layers[0].density);
}