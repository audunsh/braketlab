"""Browser orbital view used when braketlab runs under Pyodide.

Desktop sessions keep Evince. The shader text matches Evince's BraketView so
the same ket looks the same in both places.
"""

import pathlib
import sys

import numpy as np

_ESM_PATH = pathlib.Path(__file__).resolve().parent / "static" / "orbital.js"


def _in_browser():
    """True inside Pyodide, where Evince's labextension is not available."""
    return sys.platform == "emscripten" or "pyodide" in sys.modules


def generate_webgl_shader(
    ket_instance,
    time_dependent=False,
    blender="    gl_FragColor = gl_FragColor + vec4(csR, csI, -1.0*csR, .1)",
    squared=False,
):
    """WebGL fragment shader for a ket. Same construction as evince.braketview."""
    shader_code = ket_instance.get_ccode()

    fragment_shader = """uniform vec3 user_color;
uniform float time;

varying vec2 vUv;
varying vec3 pos;
varying vec3 tex;
varying float q;
varying float csI;
varying float csR;
"""
    for i in range(len(shader_code)):
        fragment_shader += "varying float cs%i;\n" % i

    fragment_shader += """varying float cs;

void main() {

    vec2 p = vUv;
    float q = """

    nd = len(ket_instance.ket_sympy_expression.free_symbols)

    for i in range(nd):
        fragment_shader += "tex[%i]*tex[%i] +" % (i, i)

    fragment_shader = fragment_shader[:-1] + ";\n"

    for i in range(len(shader_code)):
        fragment_shader += "    float cs%i = %s;\n" % (i, shader_code[i])

    if time_dependent:
        csI = "float csI ="
        csR = "float csR ="
        for i in range(len(shader_code)):
            csI += " -1.0*cs%i*sin(%f*time) +" % (i, ket_instance.energy[i])
            csR += " cs%i*cos(%f*time) +" % (i, ket_instance.energy[i])
        csI = csI[:-1] + ";\n"
        csR = csR[:-1] + ";\n"
    else:
        csI = "float csI = 0.0 +"
        csR = "float csR ="
        for i in range(len(shader_code)):
            csR += " cs%i +" % (i)
        csI = csI[:-1] + ";\n"
        csR = csR[:-1] + ";\n"

    fragment_shader += "    %s" % csI
    fragment_shader += "    %s" % csR

    if squared:
        fragment_shader += "    csR = pow(csR, 2.0) + pow(csI, 2.0);\n"
        fragment_shader += "    csI = 0.0;\n"

    if nd == 1:
        if blender is None:
            blender = "    gl_FragColor = gl_FragColor + vec4(csR, csI, -1.0*csR, 1)"
        fragment_shader += """    csR = smoothstep(0.9*tex[1], tex[1], csR);\n"""
        fragment_shader += """    csI = smoothstep(0.9*tex[1], tex[1], csI);\n"""
        fragment_shader += """%s;
}""" % blender
    if nd == 2:
        if blender is None:
            blender = "    gl_FragColor = gl_FragColor + vec4(csR, csI, -1.0*csR, 1)"
        fragment_shader += """%s;
}""" % blender
    if nd >= 3:
        if blender is None:
            blender = "    gl_FragColor = gl_FragColor + vec4(csR, csI, -1.0*csR, .1)"
        fragment_shader += """%s;
}""" % blender

    return fragment_shader, nd


def _appearance(ket_instance, web):
    nd = len(ket_instance.bra_sympy_expression.free_symbols)
    blend_factor = 0.1 if nd > 2 else 1.0
    if web:
        return (
            [1.0, 1.0, 1.0],
            False,
            "    gl_FragColor = vec4(.9*csR - csI,  .9*abs(csR) + csI, -1.0*csR - csI, %f)"
            % blend_factor,
        )
    return (
        [0.0, 0.0, 0.0],
        True,
        "    gl_FragColor = vec4(.9*csR ,  csI, -1.0*csR, %f)" % blend_factor,
    )


def _orbital_widget(ket_instance, squared, n_concentric, bg_color, additive, blender):
    try:
        import anywidget
        import traitlets
    except ImportError as exc:
        raise ImportError(
            "anywidget is required to draw orbitals in JupyterLite. "
            "Install it with: pip install anywidget"
        ) from exc

    if not _ESM_PATH.is_file():
        raise FileNotFoundError(
            "Missing bundled orbital viewer at %s. "
            "Rebuild it with: npm run build --prefix braketlab/js" % _ESM_PATH
        )

    class OrbitalView(anywidget.AnyWidget):
        _esm = _ESM_PATH

        shader_code = traitlets.Unicode("").tag(sync=True)
        squared = traitlets.Bool(False).tag(sync=True)
        n_dims = traitlets.Int(3).tag(sync=True)
        n_concentric = traitlets.Int(100).tag(sync=True)
        bg_color = traitlets.List([0.0, 0.0, 0.0]).tag(sync=True)
        additive = traitlets.Bool(True).tag(sync=True)

        def __init__(self, ket, squared=False, n_concentric=100, bg_color=None, additive=True, blender=None):
            super().__init__()
            self.squared = squared
            self.n_concentric = int(n_concentric)
            self.additive = additive
            if bg_color is not None:
                self.bg_color = list(bg_color)
            time_dependent = bool(np.sum(np.array(ket.energy) ** 2.0) > 1e-10)
            shader, nd = generate_webgl_shader(
                ket,
                time_dependent=time_dependent,
                blender=blender,
                squared=squared,
            )
            self.shader_code = shader
            self.n_dims = int(nd)

    return OrbitalView(
        ket_instance,
        squared=squared,
        n_concentric=n_concentric,
        bg_color=bg_color,
        additive=additive,
        blender=blender,
    )


def open_orbital_view(ket_instance, web=False, squared=False, n_concentric=100):
    """Return the 3D view for a ket.

    Evince on a normal kernel. The bundled anywidget under Emscripten.
    """
    bg_color, additive, blender = _appearance(ket_instance, web)
    if _in_browser():
        return _orbital_widget(
            ket_instance,
            squared=squared,
            n_concentric=n_concentric,
            bg_color=bg_color,
            additive=additive,
            blender=blender,
        )

    import evince as ev

    return ev.BraketView(
        ket_instance,
        additive=additive,
        bg_color=bg_color,
        blender=blender,
        squared=squared,
        n_concentric=n_concentric,
    )
