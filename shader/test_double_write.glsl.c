#version 460 core

#include "shader/precision.glsl.c"

struct TestStruct {
    BUFF_REAL a;
    BUFF_REAL b;
    BUFF_REAL c;
    BUFF_REAL d;
};

// Simple test: write a double value into struct fields
layout(std430, binding = 0) buffer Output {
    TestStruct test_structs[];
};

layout(local_size_x = 1) in;

void main() {
    test_structs[0].a = BUFF_REAL(1.3LF);
    test_structs[0].b = BR(1.3LF);
    test_structs[0].c = double(1.3LF);
    test_structs[0].d = 1.3LF;  // Direct assignment
}