#version 460 core

#include "shader/precision.glsl.c"

// Simple test: write a double value
layout(std430, binding = 0) buffer Output {
    BUFF_REAL test_values[];
};

layout(local_size_x = 1) in;

void main() {
    test_values[0] = BUFF_REAL(1.3);
    test_values[1] = BR(1.3);
    test_values[2] = double(1.3);
}
