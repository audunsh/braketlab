"""
Browser-based orbital visualization via anywidget.

Provides OrbitalView, an anywidget-based widget that renders quantum
mechanical wavefunctions using Three.js with Gaussian splatting-inspired
billboard rendering.

Replaces the Evince BraketView for browser-native deployment.
"""

import pathlib

_WIDGET_ESM = """
import * as THREE from "https://esm.sh/three@0.170.0?bundle";
import { OrbitControls } from "https://esm.sh/three@0.170.0/examples/jsm/controls/OrbitControls.js?bundle";
import { VRButton } from "https://esm.sh/three@0.170.0/examples/jsm/webxr/VRButton.js?bundle";

function createSplatter(fragmentShader, instanceCount, additive) {
  const vertexShader = `
    varying vec3 tex;
    varying vec2 vUv;
    attribute vec3 aRadius;
    void main() {
      vUv = uv;
      float cameraLen = length(cameraPosition);
      vec3 transformed = position * aRadius.x * cameraLen;
      tex = transformed + 3.0 * cameraPosition;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(tex, 1.0);
    }
  `;

  const baseGeometry = new THREE.SphereGeometry(1.0, 16, 16);
  const instancedGeometry = new THREE.InstancedBufferGeometry().copy(baseGeometry);
  instancedGeometry.maxInstancedCount = instanceCount;

  const aRadius = new Float32Array(instanceCount * 3);
  const aCurve = new Float32Array(instanceCount * 3);

  for (let i = 0; i < instanceCount; i++) {
    const t = i / instanceCount;
    aRadius[i * 3] = 2.0 * Math.pow(t, 1.5) + 2.0;
    aRadius[i * 3 + 1] = 0.0;
    aRadius[i * 3 + 2] = 0.0;
    aCurve[i * 3] = 0.0;
    aCurve[i * 3 + 1] = 0.0;
    aCurve[i * 3 + 2] = 0.0;
  }

  instancedGeometry.setAttribute("aRadius",
    new THREE.InstancedBufferAttribute(aRadius, 3, false));
  instancedGeometry.setAttribute("aCurve",
    new THREE.InstancedBufferAttribute(aCurve, 3, false));

  const material = new THREE.ShaderMaterial({
    fragmentShader,
    vertexShader,
    uniforms: {
      time: { value: 0.0 },
      resolution: { value: new THREE.Vector2() },
      user_color: { value: new THREE.Vector3(1, 1, 1) },
    },
    side: THREE.BackSide,
    blending: additive ? THREE.AdditiveBlending : THREE.SubtractiveBlending,
    depthWrite: false,
    transparent: true,
  });

  return new THREE.InstancedMesh(instancedGeometry, material, instanceCount);
}

function createSurface(fragmentShader, additive) {
  const vertexShader = `
    uniform float time;
    varying vec2 vUv;
    varying vec3 pos;
    varying vec3 tex;
    void main() {
      vUv = uv;
      pos = position;
      tex = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;

  const material = new THREE.ShaderMaterial({
    uniforms: {
      time: { value: 0.0 },
      resolution: { value: new THREE.Vector2() },
      user_color: { value: new THREE.Vector3(1, 1, 1) },
    },
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
    blending: additive ? THREE.AdditiveBlending : THREE.SubtractiveBlending,
    depthWrite: false,
  });

  const geometry = new THREE.PlaneGeometry(10000, 10000);
  return new THREE.Mesh(geometry, material);
}

function render({ model, el }) {
  const container = document.createElement("div");
  container.style.width = "100%";
  el.appendChild(container);

  const scene = new THREE.Scene();
  const width = el.clientWidth || 800;
  const height = width * 0.6;

  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  camera.position.z = 5;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);

  const bgColor = model.get("bg_color") || [0, 0, 0];
  renderer.setClearColor(new THREE.Color(...bgColor), 1);

  if (model.get("vr_enabled")) {
    renderer.xr.enabled = true;
    container.appendChild(VRButton.createButton(renderer));
  }

  container.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);

  function updateOrbital() {
    while (scene.children.length > 0) {
      const obj = scene.children[0];
      scene.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    const shaderCode = model.get("shader_code");
    if (!shaderCode) return;

    const nDims = model.get("n_dims") || 3;
    const additive = model.get("additive") !== false;

    if (nDims < 3) {
      scene.add(createSurface(shaderCode, additive));
    } else {
      scene.add(createSplatter(shaderCode, 100, additive));
    }
  }

  updateOrbital();
  model.on("change:shader_code", updateOrbital);

  renderer.setAnimationLoop(() => {
    for (const child of scene.children) {
      if (child.material && child.material.uniforms && child.material.uniforms.time) {
        child.material.uniforms.time.value += 0.01;
      }
    }
    renderer.render(scene, camera);
  });

  return () => {
    renderer.setAnimationLoop(null);
    renderer.dispose();
    controls.dispose();
  };
}

export default { render };
"""

try:
    import anywidget
    import traitlets

    class OrbitalView(anywidget.AnyWidget):
        """
        Interactive 3D orbital viewer widget.

        Renders a braketlab ket using WebGL fragment shaders
        generated from the SymPy expression tree. Uses Three.js
        loaded from ESM CDN for zero-install browser rendering.

        Works in JupyterLite (Pyodide), JupyterLab, classic Jupyter, and Colab.
        """

        _esm = traitlets.Unicode(_WIDGET_ESM).tag(sync=True)

        shader_code = traitlets.Unicode("").tag(sync=True)
        wgsl_code = traitlets.Unicode("").tag(sync=True)
        energies = traitlets.List([]).tag(sync=True)
        mode = traitlets.Unicode("splat").tag(sync=True)
        time_dependent = traitlets.Bool(False).tag(sync=True)
        squared = traitlets.Bool(False).tag(sync=True)
        n_dims = traitlets.Int(3).tag(sync=True)

        bg_color = traitlets.List([0.0, 0.0, 0.0]).tag(sync=True)
        additive = traitlets.Bool(True).tag(sync=True)
        vr_enabled = traitlets.Bool(False).tag(sync=True)

        def __init__(self, ket_instance=None, mode="splat", squared=False,
                     bg_color=None, additive=True, vr_enabled=False, **kwargs):
            super().__init__(**kwargs)

            self.mode = mode
            self.squared = squared
            self.additive = additive
            self.vr_enabled = vr_enabled
            if bg_color is not None:
                self.bg_color = list(bg_color)

            if ket_instance is not None:
                self.set_ket(ket_instance)

        def set_ket(self, ket_instance):
            """Update the displayed orbital from a ket."""
            from .codegen import generate_fragment_shader, generate_wgsl_fragment_shader

            time_dep = any(abs(e) > 1e-10 for e in ket_instance.energy)

            blender = None
            if not self.additive:
                blender = (
                    "    gl_FragColor = vec4("
                    ".9*csR - csI, .9*abs(csR) + csI, -1.0*csR - csI, %s)"
                )

            glsl_shader, nd = generate_fragment_shader(
                ket_instance, time_dependent=time_dep, squared=self.squared,
                blender=blender,
            )
            wgsl_shader, _ = generate_wgsl_fragment_shader(
                ket_instance, time_dependent=time_dep, squared=self.squared
            )

            self.shader_code = glsl_shader
            self.wgsl_code = wgsl_shader
            self.energies = list(ket_instance.energy)
            self.time_dependent = time_dep
            self.n_dims = nd

except ImportError:
    class OrbitalView:  # type: ignore[no-redef]
        """Stub when anywidget is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "anywidget is required for OrbitalView. "
                "Install it with: pip install anywidget"
            )
