#ifndef PRECISION_GLSL
#define PRECISION_GLSL

#define USE_DOUBLES_IN_BUFFER
#define USE_DOUBLES_IN_CALCULATIONS

// Diagnostic: This will cause a compile error if USE_DOUBLES_IN_BUFFER is somehow not defined
#ifndef USE_DOUBLES_IN_BUFFER
#error "USE_DOUBLES_IN_BUFFER is not defined!"
#endif

#ifdef USE_DOUBLES_IN_BUFFER
    #define BUFF_REAL double
    #define BUFF_VEC4 dvec4
    #define BUFF_VEC3 dvec3
#else
    #define BUFF_REAL float
    #define BUFF_VEC4 vec4
    #define BUFF_VEC3 vec3
#endif
    
#ifdef USE_DOUBLES_IN_CALCULATIONS
    #define CALC_REAL double
    #define CALC_VEC4 dvec4
    #define CALC_VEC3 dvec3
    #define ITER 11
#else
    #define CALC_REAL float
    #define CALC_VEC4 vec4
    #define CALC_VEC3 vec3
    #define ITER 8
#endif

#define BR(x) BUFF_REAL(x)
#define R(x) CALC_REAL(x)

const CALC_REAL PI = R(3.14159265358979323846LF);

#endif