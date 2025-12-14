#ifndef PRECISION_GLSL
#define PRECISION_GLSL

// These will be injected by ShaderConfig:
// #define BUFFER_PRECISION double
// #define CALC_PRECISION double

// Map the precision settings to actual types and constants
#if BUFFER_PRECISION == double
    #define USE_DOUBLES_IN_BUFFER
    #define BUFF_REAL double
    #define BUFF_VEC4 dvec4
    #define BUFF_VEC3 dvec3
    #define BR(x) double(x)
#elif BUFFER_PRECISION == float
    #define BUFF_REAL float
    #define BUFF_VEC4 vec4
    #define BUFF_VEC3 vec3
    #define BR(x) float(x)
#else
    #error "BUFFER_PRECISION must be defined as either 'double' or 'float'"
#endif
    
#if CALC_PRECISION == double
    #define USE_DOUBLES_IN_CALCULATIONS
    #define CALC_REAL double
    #define CALC_VEC4 dvec4
    #define CALC_VEC3 dvec3
    #define ITER 11
    #define R(x) double(x)
#elif CALC_PRECISION == float
    #define CALC_REAL float
    #define CALC_VEC4 vec4
    #define CALC_VEC3 vec3
    #define ITER 8
    #define R(x) float(x)
#else
    #error "CALC_PRECISION must be defined as either 'double' or 'float'"
#endif

const CALC_REAL PI = R(3.14159265358979323846LF);

#endif