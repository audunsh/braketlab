"""
Shader code generation from SymPy expressions.

Generates GLSL (WebGL) and WGSL (WebGPU) code snippets from the symbolic
wavefunction expressions in a ket, for use in fragment/compute shaders.

Pipeline: SymPy expr -> symbol substitution (tex[i]) -> r^2 -> q -> ccode/wgsl
"""

import numpy as np
import sympy as sp

from .algebra import basisfunction, ket, get_ordered_symbols


def _get_r2_sp(expr):
    """Extract the sum-of-squares term from an expression for q substitution."""
    symbols = get_ordered_symbols(expr)
    r2 = sum(s ** 2 for s in symbols)
    return r2


def _prepare_expression(coeff, basis_fn):
    """Prepare a single basis term for code generation."""
    ret = coeff * basis_fn.ket_sympy_expression
    symbol_list = get_ordered_symbols(ret)
    for i, sym in enumerate(symbol_list):
        ret = ret.replace(sym, sp.symbols(f"tex[{i}]"))

    r2 = _get_r2_sp(ret)
    if r2 != 0:
        ret = ret.subs(r2, sp.symbols("q"))

    ret = ret.simplify().subs(sp.pi, float(np.pi))
    return ret


def _glsl_fixup(code_str):
    """Post-process C code to be GLSL-compatible (float literals)."""
    for j in range(20):
        code_str = code_str.replace(f" {j})", f" {j}.0)")
        code_str = code_str.replace(f" {j},", f" {j}.0,")
    return code_str


def generate_glsl_snippets(ket_instance):
    """
    Generate GLSL-compatible code snippets from a ket.

    Returns a list of strings, one per basis function term.
    Each string is valid in a GLSL fragment shader context where
    tex[0], tex[1], tex[2] are the fragment coordinates and
    q = tex[0]^2 + tex[1]^2 + tex[2]^2.
    """
    snippets = []
    for i in range(len(ket_instance.coefficients)):
        bi = ket_instance.basis[i]
        if not isinstance(bi, (basisfunction, ket)):
            continue

        expr = _prepare_expression(ket_instance.coefficients[i], bi)
        code = sp.ccode(expr)
        code = _glsl_fixup(code)
        snippets.append(code)
    return snippets


def generate_wgsl_snippets(ket_instance):
    """
    Generate WGSL code snippets from a ket.

    Returns a list of strings, one per basis function term.
    Each string is valid WGSL where tex is a vec3<f32> and
    q: f32 = dot(tex, tex).

    WGSL differences from GLSL/C:
    - pow(x, y) -> pow(x, y)  (same)
    - exp(x) -> exp(x)        (same)
    - float literals required  (same requirement)
    - No implicit int->float conversion
    """
    snippets = []
    for i in range(len(ket_instance.coefficients)):
        bi = ket_instance.basis[i]
        if not isinstance(bi, (basisfunction, ket)):
            continue

        expr = _prepare_expression(ket_instance.coefficients[i], bi)
        code = sp.ccode(expr)
        code = _glsl_fixup(code)

        # WGSL-specific transforms
        code = code.replace("tex[0]", "tex.x")
        code = code.replace("tex[1]", "tex.y")
        code = code.replace("tex[2]", "tex.z")
        snippets.append(code)
    return snippets


def generate_fragment_shader(ket_instance, time_dependent=False, squared=False,
                              blender=None):
    """
    Generate a complete GLSL fragment shader for a ket.

    This is the modernized version of evince.braketview.generate_webgl_shader().
    """
    shader_code = generate_glsl_snippets(ket_instance)
    nd = len(ket_instance.ket_sympy_expression.free_symbols)

    header = """uniform vec3 user_color;
uniform float time;
varying vec2 vUv;
varying vec3 pos;
varying vec3 tex;

void main() {
    float q = """

    q_expr = " + ".join(f"tex[{i}]*tex[{i}]" for i in range(nd))
    header += q_expr + ";\n"

    body = ""
    for i, snippet in enumerate(shader_code):
        body += f"    float cs{i} = {snippet};\n"

    if time_dependent:
        csR = "    float csR = " + " + ".join(
            f"cs{i}*cos({ket_instance.energy[i]:.6f}*time)"
            for i in range(len(shader_code))
        ) + ";\n"
        csI = "    float csI = " + " + ".join(
            f"-1.0*cs{i}*sin({ket_instance.energy[i]:.6f}*time)"
            for i in range(len(shader_code))
        ) + ";\n"
    else:
        csR = "    float csR = " + " + ".join(
            f"cs{i}" for i in range(len(shader_code))
        ) + ";\n"
        csI = "    float csI = 0.0;\n"

    body += csI + csR

    if squared:
        body += "    csR = pow(csR, 2.0) + pow(csI, 2.0);\n"
        body += "    csI = 0.0;\n"

    blend_factor = "0.1" if nd >= 3 else "1.0"
    if blender is None:
        blender = f"    gl_FragColor = vec4(csR, csI, -1.0*csR, {blend_factor})"
    elif "%s" in blender:
        blender = blender % blend_factor

    footer = f"{blender};\n}}"

    return header + body + footer, nd


def generate_wgsl_fragment_shader(ket_instance, time_dependent=False, squared=False):
    """
    Generate a complete WGSL fragment shader for a ket.

    Returns (shader_code: str, dimensionality: int)
    """
    snippets = generate_wgsl_snippets(ket_instance)
    nd = len(ket_instance.ket_sympy_expression.free_symbols)

    lines = [
        "struct Uniforms { time: f32, }",
        "@group(0) @binding(0) var<uniform> u: Uniforms;",
        "",
        "@fragment",
        "fn main(@location(0) tex: vec3<f32>) -> @location(0) vec4<f32> {",
        "    let q: f32 = dot(tex, tex);",
    ]

    for i, snippet in enumerate(snippets):
        lines.append(f"    let cs{i}: f32 = {snippet};")

    if time_dependent:
        terms_r = [f"cs{i} * cos({ket_instance.energy[i]:.6f} * u.time)" for i in range(len(snippets))]
        terms_i = [f"-cs{i} * sin({ket_instance.energy[i]:.6f} * u.time)" for i in range(len(snippets))]
        lines.append(f"    var csR: f32 = {' + '.join(terms_r)};")
        lines.append(f"    var csI: f32 = {' + '.join(terms_i)};")
    else:
        terms = [f"cs{i}" for i in range(len(snippets))]
        lines.append(f"    var csR: f32 = {' + '.join(terms)};")
        lines.append("    var csI: f32 = 0.0;")

    if squared:
        lines.append("    csR = csR * csR + csI * csI;")
        lines.append("    csI = 0.0;")

    alpha = "0.1" if nd >= 3 else "1.0"
    lines.append(f"    return vec4<f32>(csR, csI, -csR, {alpha});")
    lines.append("}")

    return "\n".join(lines), nd
