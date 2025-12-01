"""
 GL 4.6 Compute Shader Framework (PyQt5 + PyOpenGL)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
np.seterr(all='raise')

from PyQt5.QtGui import QSurfaceFormat, QOpenGLContext, QOffscreenSurface, QGuiApplication
from PyQt5.QtCore import QCoreApplication, Qt

from OpenGL import GL
from OpenGL.GL import shaders as gl_shaders
import ctypes
#import re
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj

import struct

buffer_type = "double"
calc_type = "double"

if buffer_type == "float":
    np_float = np.float32
    c_float = ctypes.c_float
    buffer_vec4 = "vec4"
elif buffer_type == "double":
    np_float = np.float64
    c_float = ctypes.c_double 
    buffer_vec4 = "dvec4"
else:
    raise RuntimeError("Invalid buffer type")
    
if calc_type == "float":
    calc_vec4 = "vec4"
    calc_iter = 7
elif calc_type == "double":
    calc_vec4 = "dvec4"
    calc_iter = 11
else:
    raise RuntimeError("Invalid calc type")
    


class GLSLComputeProgram:
    def __init__(self, harness, path):
        self.harness = harness
        self.source_code = GLSLComputeProgram.load(path)
        self.compile_program()
        self.program = 0
        self.ssbo_in = 0
        self.ssbo_out = 0

    @staticmethod
    def load(path):
        
        def load_lines(path):
            lines_out = []
            with open(path, 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#include"):
                    subpath = eval(stripped[8:].strip())
                    print(subpath)
                    lines_out.extend(load_lines(subpath))
                else:
                    lines_out.append(line)
            return lines_out
        
        lines_out = load_lines(path)
        
        return ''.join(lines_out)
 
    def _dump_src(self):
        for line_no, line in enumerate(self.source_code.split('\n')):
            print(f"\033[1;33m {line_no+1:4d}  \033[1;34m{line}\033[m")
           

    def compile_program(self):
        # Compile compute shader program
        try:
            cs = gl_shaders.compileShader(self.source_code, GL.GL_COMPUTE_SHADER) 
            self.program = gl_shaders.compileProgram(cs)
        except Exception as e:
            print('\033[1;31mError\033[m')
            log = GL.glGetShaderInfoLog(cs).decode(errors="ignore") if 'cs' in locals() else ""
            e_readable = e.args[0].encode('utf-8').decode('unicode_escape')
            msg = f"Compute shader compilation/link failed:\n{e_readable}\nShader log:\n{log}"
            print(f'\033[1;33m{msg}\033[m')
            self._dump_src()
            raise RuntimeError(msg) from None

        # Create SSBOs (names only; sizes set per-dispatch)
        # self.ssbo_in = GL.glGenBuffers(1)
        self.ssbo_out = GL.glGenBuffers(1)

    #def run(self, inputs_vec4: np.ndarray, local_size_x: int = 256) -> np.ndarray:
    def run(self, num_samples):
        print("\033[1;35mCheckpoint 1\033[m")
        local_size_x = 256

        def _alloc_or_update_ssbo(
                    buf_name: int, 
                    byte_size: int, 
                    initial: Optional[np.ndarray], 
                    usage=GL.GL_DYNAMIC_DRAW):
            print("\033[1;35mCheckpoint 1b1\033[m")
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buf_name)
            print("\033[1;35mCheckpoint 1b2\033[m")
            if initial is not None:
                assert initial.nbytes == byte_size, (initial.nbytes, byte_size)
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, byte_size, initial, usage)
            else:
                GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, byte_size, None, usage)
            print("\033[1;35mCheckpoint 1b3\033[m")
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)        
            print("\033[1;35mCheckpoint 1b4\033[m")

        #if inputs_vec4.dtype != np_float:
        #    inputs_vec4 = inputs_vec4.astype(np_float, copy=False)
        #if inputs_vec4.ndim != 2 or inputs_vec4.shape[1] != 4:
        #    raise ValueError("inputs_vec4 must have shape (N, 4)")

        evaluation_struct = struct.pack("ddddd", 0., 0., 0., 0., 0.)



        #N = inputs_vec4.shape[0]
        #in_bytes = inputs_vec4.nbytes  # N * 4 * 4
        #out_bytes = N * 4  # N * sizeof(float)

        out_bytes = num_samples * len(evaluation_struct)
        print(f"OUT_BYTES: {out_bytes}")

        # Allocate & upload SSBOs
        #_alloc_or_update_ssbo(self.ssbo_in, in_bytes, inputs_vec4, usage=GL.GL_STATIC_DRAW)
        print("\033[1;35mCheckpoint 1b\033[m")
        _alloc_or_update_ssbo(self.ssbo_out, out_bytes, None, usage=GL.GL_DYNAMIC_READ)

        # Bind SSBOs to binding points 0 and 1 as declared in GLSL
        #GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.ssbo_in)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, self.ssbo_out)


        print("\033[1;35mCheckpoint 2\033[m")

        # Use program and set uniforms
        GL.glUseProgram(self.program)

        print("\033[1;35mCheckpoint 3\033[m")

        loc = GL.glGetUniformLocation(self.program, "num_samples")
        GL.glUniform1ui(loc, np.uint32(num_samples))

        loc = GL.glGetUniformLocation(self.program, "seed")
        GL.glUniform1ui(loc, np.uint32(42))


        print("\033[1;35mCheckpoint 4\033[m")

        # Compute dispatch sizing
        groups_x = (num_samples + local_size_x - 1) // local_size_x
        print(f'groups_x={groups_x}')
        GL.glDispatchCompute(groups_x, 1, 1)
        GL.glFinish()

        print("\033[1;35mCheckpoint 5\033[m")

        # Ensure writes to SSBOs are visible to the CPU readback
        GL.glMemoryBarrier(GL.GL_ALL_BARRIER_BITS)

        # Read back output
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.ssbo_out)
        #ptr = GL.glMapBufferRange(GL.GL_SHADER_STORAGE_BUFFER, 0, out_bytes, GL.GL_MAP_READ_BIT)
        #assert ptr, "Map failed"


        dtype = np.dtype([
            ('a', np_float),
            ('b', np_float),
            ('c', np_float),
            ('p', np_float),
            ('result', np_float)
        ])

        
        raw_bytes = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, out_bytes)

        # Convert to numpy structured array
        results = np.frombuffer(raw_bytes, dtype=dtype, count=num_samples)
        
        
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        
        # Unbind program & SSBOs
        GL.glUseProgram(0)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, 0)
        #GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, 0)

        return results
    
class GLSLComputeHarness:
    def __init__(self):
        # Minimal Qt app loop to own the GL context
        # Must use QGuiApplication for GL surfaces/contexts
        if QCoreApplication.instance() is None:
            QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
            self.app = QGuiApplication(sys.argv)
        else:
            self.app = QCoreApplication.instance()

        # Request a 4.6 Core profile context
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setVersion(4, 6)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
        fmt.setDepthBufferSize(0)
        fmt.setStencilBufferSize(0)

        # Set default format BEFORE creating surfaces/contexts
        QSurfaceFormat.setDefaultFormat(fmt)

        # Create context
        self.context = context = QOpenGLContext()
        context.setFormat(fmt)
        created = context.create()
        if not created or not context.isValid():
            raise RuntimeError("Failed to create a valid QOpenGLContext (OpenGL 4.6 core)")

        # Create offscreen surface to make context current
        self.surface = surface = QOffscreenSurface()
        surface.setFormat(fmt)
        surface.create()
        if not surface.isValid():
            raise RuntimeError("Failed to create QOffscreenSurface (invalid surface)")

        if not context.makeCurrent(surface):
            raise RuntimeError("Failed to make context current")

    def create_program(self, path):
        prg = GLSLComputeProgram(self, path)
        prg.compile_program()
        return prg

    

    def cleanup(self):
        pass
        '''
        # Destroy GL objects (safe to call multiple times)
        print("\033[1;33mCleanup\033[m")
        try:
            if self.res.program:
                GL.glDeleteProgram(self.res.program)
                self.res.program = 0
            if self.res.ssbo_in:
                GL.glDeleteBuffers(1, [self.res.ssbo_in])
                self.res.ssbo_in = 0
            if self.res.ssbo_out:
                GL.glDeleteBuffers(1, [self.res.ssbo_out])
                self.res.ssbo_out = 0
        finally:
            # Release context
            self.res.context.doneCurrent()
        '''



def _demo():
    
    
    # Create runner
    runner = GLSLComputeHarness()
    
    


    # Verify version
    ver = GL.glGetString(GL.GL_VERSION)
    rend = GL.glGetString(GL.GL_RENDERER)
    vend = GL.glGetString(GL.GL_VENDOR)
    print("GL_VERSION:", ver.decode() if ver else "?")
    print("GL_RENDERER:", rend.decode() if rend else "?")
    print("GL_VENDOR:", vend.decode() if vend else "?")

        
    #shader_src = GLSLComputeProgram.load("shader/test_carlson_rj.glsl.c")
    #shader_src = GLSLComputeProgram.load("shader/explore_variations.glsl.c")
    #print(f"\033[1;34m{shader_src}\033[m")
   
    prg = runner.create_program("shader/test_carlson_rj.glsl.c")


    # Prepare input: N arbitrary vec4s
    N = 1_000_000
    
    # Run compute
    start_time = time.time()
    results = prg.run(N)
    end_time = time.time()
    elapsed = end_time-start_time
    print(f"\033[1;32mProgram took {elapsed} seconds to compute.\033[m")


    worst = 50
    case = None
        
    for idx, result in enumerate(results):
        #print(idx, result)
        gpu_ans = result['result']
        sci_ans = elliprj(result['a'], result['b'], result['c'], result['p'])
        
        err = (gpu_ans - sci_ans)/sci_ans
        
        if err:
            prec = int(np.abs(np.round(np.log(np.abs(err))/np.log(10))))
            worst = min(worst, prec)
            
            if prec == worst:
                case = result
        else:
            prec = "exact"

    print(f"worst precision: {worst}")
    print(f"case: RJ({result['a']}, {result['b']}, {result['c']}, {result['p']}) = {result['result']}")
    print(f"scipy: {sci_ans}")



if __name__ == "__main__":
    _demo()
