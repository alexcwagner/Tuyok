"""
Generic GL 4.6 Compute Shader Framework (PyQt5 + PyOpenGL)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time
import numpy as np

from PyQt5.QtGui import QSurfaceFormat, QOpenGLContext, QOffscreenSurface, QGuiApplication
from PyQt5.QtCore import QCoreApplication, Qt

from OpenGL import GL
from OpenGL.GL import shaders as gl_shaders


@dataclass
class ShaderConfig:
    """Configuration for shader compilation."""
    defines: Dict[str, str] = field(default_factory=dict)
    
    @staticmethod
    def precision_config(buffer_precision: str = "double", 
                        calc_precision: str = "double") -> 'ShaderConfig':
        """Create a precision configuration."""
        return ShaderConfig(defines={
            "BUFFER_PRECISION": buffer_precision,
            "CALC_PRECISION": calc_precision
        })


@dataclass
class BufferSpec:
    """Specification for an SSBO buffer."""
    binding: int
    dtype: np.dtype
    count: int  # Number of elements
    mode: str = "out"  # "in", "out", or "inout"
    initial_data: Optional[np.ndarray] = None
    
    @property
    def byte_size(self):
        # Handle different dtype scenarios
        if isinstance(self.dtype, type) and self.dtype == np.uint8:
            # For raw bytes, count IS the byte size
            return self.count
        
        # Always convert to np.dtype to get itemsize
        dt = np.dtype(self.dtype) if not isinstance(self.dtype, np.dtype) else self.dtype
        return self.count * dt.itemsize
    
    @property
    def usage(self):
        if self.mode == "in":
            return GL.GL_STATIC_DRAW
        elif self.mode == "out":
            return GL.GL_DYNAMIC_READ
        else:  # inout
            return GL.GL_DYNAMIC_COPY


@dataclass
class UniformSpec:
    """Specification for a uniform variable."""
    name: str
    value: Any
    uniform_type: str  # "1ui", "1i", "1f", "3f", "4fv", etc.


class GLSLComputeProgram:
    """A compiled compute shader program with buffer and uniform management."""
    
    def __init__(self, harness: 'GLSLComputeHarness', 
                 shader_path: str,
                 config: Optional[ShaderConfig] = None):
        self.harness = harness
        self.shader_path = shader_path
        self.config = config or ShaderConfig()
        self.source_code = self._load_and_configure_shader(shader_path)
        self.program = 0
        self.ssbos: Dict[int, int] = {}  # binding -> buffer_id
        self.local_size_x = 256  # Default
        
        self._compile()
    
    @staticmethod
    def _load_shader(path: str) -> str:
        """Load shader with #include support."""
        def load_lines(filepath):
            lines_out = []
            with open(filepath, 'r', encoding='utf-8') as fp:
                for line in fp:
                    stripped = line.strip()
                    if stripped.startswith("#include"):
                        subpath = eval(stripped[8:].strip())
                        lines_out.extend(load_lines(subpath))
                    else:
                        lines_out.append(line)
            return lines_out
        
        return ''.join(load_lines(path))
    
    def _load_and_configure_shader(self, path: str) -> str:
        """Load shader and inject configuration defines."""
        source = self._load_shader(path)
        
        if not self.config.defines:
            return source
        
        # Inject defines at the top (after #version)
        lines = source.split('\n')
        version_line_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#version'):
                version_line_idx = i + 1
                break
        
        # Build define block
        define_block = []
        for key, value in self.config.defines.items():
            define_block.append(f"#define {key} {value}")
        
        # Insert defines after version
        lines[version_line_idx:version_line_idx] = define_block
        
        return '\n'.join(lines)
    
    def _dump_source(self):
        """Pretty-print shader source with line numbers."""
        for line_no, line in enumerate(self.source_code.split('\n'), 1):
            print(f"\033[1;33m {line_no:4d}  \033[1;34m{line}\033[m")
    
    def _compile(self):
        """Compile the compute shader."""
        try:
            cs = gl_shaders.compileShader(self.source_code, GL.GL_COMPUTE_SHADER)
            self.program = gl_shaders.compileProgram(cs)
        except Exception as e:
            log = GL.glGetShaderInfoLog(cs).decode(errors="ignore") if 'cs' in locals() else ""
            e_readable = e.args[0].encode('utf-8').decode('unicode_escape')
            msg = f"Compute shader compilation failed:\n{e_readable}\nLog:\n{log}"
            print(f'\033[1;31m{msg}\033[m')
            self._dump_source()
            raise RuntimeError(msg) from None
    
    def _setup_buffer(self, spec: BufferSpec) -> int:
        """Create or update an SSBO based on spec."""
        if spec.binding not in self.ssbos:
            self.ssbos[spec.binding] = GL.glGenBuffers(1)
        
        ssbo = self.ssbos[spec.binding]
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo)
        
        # Ensure byte_size is a plain Python int
        byte_size = int(spec.byte_size)
        
        if spec.initial_data is not None:
            # Upload data
            assert spec.initial_data.nbytes == byte_size
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, 
                          byte_size, 
                          spec.initial_data, 
                          spec.usage)
        else:
            # Allocate empty
            GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, 
                          byte_size, 
                          None, 
                          spec.usage)
        
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        return ssbo
    
    def _set_uniform(self, spec: UniformSpec):
        """Set a uniform value."""
        loc = GL.glGetUniformLocation(self.program, spec.name)
        if loc == -1:
            print(f"\033[1;33mWarning: uniform '{spec.name}' not found\033[m")
            return
        
        utype = spec.uniform_type
        value = spec.value
        
        # Scalars
        if utype == "1i":
            GL.glUniform1i(loc, int(value))
        elif utype == "1ui":
            GL.glUniform1ui(loc, np.uint32(value))
        elif utype == "1f":
            GL.glUniform1f(loc, float(value))
        elif utype == "1d":
            GL.glUniform1d(loc, float(value))
        
        # Vectors
        elif utype == "2f":
            GL.glUniform2f(loc, *value)
        elif utype == "3f":
            GL.glUniform3f(loc, *value)
        elif utype == "4f":
            GL.glUniform4f(loc, *value)
        elif utype == "2i":
            GL.glUniform2i(loc, *value)
        elif utype == "3i":
            GL.glUniform3i(loc, *value)
        elif utype == "4i":
            GL.glUniform4i(loc, *value)
        elif utype == "2ui":
            GL.glUniform2ui(loc, *value)
        elif utype == "3ui":
            GL.glUniform3ui(loc, *value)
        elif utype == "4ui":
            GL.glUniform4ui(loc, *value)
        elif utype == "2d":
            GL.glUniform2d(loc, *value)
        elif utype == "3d":
            GL.glUniform3d(loc, *value)
        elif utype == "4d":
            GL.glUniform4d(loc, *value)
        
        # Arrays
        elif utype == "1fv":
            GL.glUniform1fv(loc, len(value), value)
        elif utype == "2fv":
            GL.glUniform2fv(loc, len(value)//2, value)
        elif utype == "3fv":
            GL.glUniform3fv(loc, len(value)//3, value)
        elif utype == "4fv":
            GL.glUniform4fv(loc, len(value)//4, value)
        elif utype == "1iv":
            GL.glUniform1iv(loc, len(value), value)
        elif utype == "2iv":
            GL.glUniform2iv(loc, len(value)//2, value)
        elif utype == "3iv":
            GL.glUniform3iv(loc, len(value)//3, value)
        elif utype == "4iv":
            GL.glUniform4iv(loc, len(value)//4, value)
        elif utype == "1uiv":
            GL.glUniform1uiv(loc, len(value), value)
        elif utype == "2uiv":
            GL.glUniform2uiv(loc, len(value)//2, value)
        elif utype == "3uiv":
            GL.glUniform3uiv(loc, len(value)//3, value)
        elif utype == "4uiv":
            GL.glUniform4uiv(loc, len(value)//4, value)
        elif utype == "1dv":
            GL.glUniform1dv(loc, len(value), value)
        elif utype == "2dv":
            GL.glUniform2dv(loc, len(value)//2, value)
        elif utype == "3dv":
            GL.glUniform3dv(loc, len(value)//3, value)
        elif utype == "4dv":
            GL.glUniform4dv(loc, len(value)//4, value)
        
        # Matrices
        elif utype == "matrix2fv":
            GL.glUniformMatrix2fv(loc, 1, GL.GL_FALSE, value)
        elif utype == "matrix3fv":
            GL.glUniformMatrix3fv(loc, 1, GL.GL_FALSE, value)
        elif utype == "matrix4fv":
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, value)
        elif utype == "matrix2dv":
            GL.glUniformMatrix2dv(loc, 1, GL.GL_FALSE, value)
        elif utype == "matrix3dv":
            GL.glUniformMatrix3dv(loc, 1, GL.GL_FALSE, value)
        elif utype == "matrix4dv":
            GL.glUniformMatrix4dv(loc, 1, GL.GL_FALSE, value)
        
        else:
            raise ValueError(f"Unsupported uniform type: {utype}")
    
    def _read_buffer(self, spec: BufferSpec) -> np.ndarray:
        """Read data back from an SSBO."""
        ssbo = self.ssbos[spec.binding]
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo)
        
        byte_size = int(spec.byte_size)
        raw_bytes = GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 
                                         0, 
                                         byte_size)
        
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        
        return np.frombuffer(raw_bytes, dtype=spec.dtype, count=spec.count)
    
    def run(self, 
            buffers: List[BufferSpec],
            uniforms: Optional[List[UniformSpec]] = None,
            num_invocations: Optional[int] = None,
            local_size_x: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        Run the compute shader.
        
        Args:
            buffers: List of buffer specifications
            uniforms: List of uniform specifications
            num_invocations: Total number of shader invocations (if None, inferred from first buffer)
            local_size_x: Workgroup size (default: 256)
        
        Returns:
            Dictionary mapping binding -> output data for "out" and "inout" buffers
        """
        if uniforms is None:
            uniforms = []
        
        if local_size_x is None:
            local_size_x = self.local_size_x
        
        if num_invocations is None:
            num_invocations = buffers[0].count if buffers else 0
        
        # Setup all buffers
        for spec in buffers:
            ssbo = self._setup_buffer(spec)
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, spec.binding, ssbo)
        
        # Use program
        GL.glUseProgram(self.program)
        
        # Set uniforms
        for uniform in uniforms:
            self._set_uniform(uniform)
        
        # Dispatch
        groups_x = (num_invocations + local_size_x - 1) // local_size_x
        GL.glDispatchCompute(groups_x, 1, 1)
        
        # Synchronize
        GL.glMemoryBarrier(GL.GL_ALL_BARRIER_BITS)
        GL.glFinish()
        
        # Read back output buffers
        results = {}
        for spec in buffers:
            if spec.mode in ("out", "inout"):
                results[spec.binding] = self._read_buffer(spec)
        
        # Cleanup
        GL.glUseProgram(0)
        for spec in buffers:
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, spec.binding, 0)
        
        return results
    
    def cleanup(self):
        """Free GPU resources."""
        if self.program:
            GL.glDeleteProgram(self.program)
            self.program = 0
        
        for ssbo in self.ssbos.values():
            GL.glDeleteBuffers(1, [ssbo])
        self.ssbos.clear()


class GLSLComputeHarness:
    """OpenGL context manager for compute shaders."""
    
    def __init__(self):
        if QCoreApplication.instance() is None:
            QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
            self.app = QGuiApplication(sys.argv)
        else:
            self.app = QCoreApplication.instance()
        
        # Request GL 4.6 Core
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setVersion(4, 6)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)
        
        # Create context
        self.context = QOpenGLContext()
        self.context.setFormat(fmt)
        if not self.context.create() or not self.context.isValid():
            raise RuntimeError("Failed to create OpenGL 4.6 context")
        
        # Create offscreen surface
        self.surface = QOffscreenSurface()
        self.surface.setFormat(fmt)
        self.surface.create()
        if not self.surface.isValid():
            raise RuntimeError("Failed to create offscreen surface")
        
        if not self.context.makeCurrent(self.surface):
            raise RuntimeError("Failed to make context current")
        
        self._print_gl_info()
    
    def _print_gl_info(self):
        """Print OpenGL version info."""
        ver = GL.glGetString(GL.GL_VERSION)
        rend = GL.glGetString(GL.GL_RENDERER)
        vend = GL.glGetString(GL.GL_VENDOR)
        print("GL_VERSION:", ver.decode() if ver else "?")
        print("GL_RENDERER:", rend.decode() if rend else "?")
        print("GL_VENDOR:", vend.decode() if vend else "?")
    
    def create_program(self, shader_path: str, 
                      config: Optional[ShaderConfig] = None) -> GLSLComputeProgram:
        """Create a compute program from a shader file."""
        return GLSLComputeProgram(self, shader_path, config)


# ============================================================================
# DEMO / EXAMPLE USAGE
# ============================================================================

def demo_carlson_rj():
    """Example: Test Carlson RJ implementation."""
    from scipy.special import elliprj
    
    harness = GLSLComputeHarness()
    
    # Test with double/double precision
    config = ShaderConfig.precision_config("double", "double")
    program = harness.create_program("shader/test_carlson_rj.glsl.c", config)
    
    N = 1_000_000
    
    # Define output buffer structure
    dtype = np.dtype([
        ('a', np.float64),
        ('b', np.float64),
        ('c', np.float64),
        ('p', np.float64),
        ('result', np.float64)
    ])
    
    # Define buffers and uniforms
    buffers = [
        BufferSpec(
            binding=0,
            dtype=dtype,
            count=N,
            mode="out"
        )
    ]
    
    uniforms = [
        UniformSpec("num_samples", N, "1ui"),
        UniformSpec("seed", 42, "1ui")
    ]
    
    # Run
    print(f"Computing {N} samples...")
    start_time = time.time()
    results = program.run(buffers, uniforms, num_invocations=N)
    elapsed = time.time() - start_time
    print(f"\033[1;32mCompleted in {elapsed:.3f} seconds\033[m")
    
    # Get output from binding 0
    data = results[0]
    
    # Validate against scipy
    worst_precision = 50
    worst_case = None
    
    for i, row in enumerate(data):
        gpu_ans = row['result']
        sci_ans = elliprj(row['a'], row['b'], row['c'], row['p'])
        
        if sci_ans != 0:
            err = abs((gpu_ans - sci_ans) / sci_ans)
            if err > 0:
                prec = int(np.abs(np.round(np.log10(err))))
                if prec < worst_precision:
                    worst_precision = prec
                    worst_case = row
    
    print(f"\nWorst precision: {worst_precision} decimal places")
    if worst_case is not None:
        print(f"Worst case: RJ({worst_case['a']:.6g}, {worst_case['b']:.6g}, "
              f"{worst_case['c']:.6g}, {worst_case['p']:.6g})")
        print(f"  GPU:   {worst_case['result']:.15e}")
        sci = elliprj(worst_case['a'], worst_case['b'], worst_case['c'], worst_case['p'])
        print(f"  SciPy: {sci:.15e}")
    
    program.cleanup()


if __name__ == "__main__":
    demo_carlson_rj()