import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

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
  for (let i = 0; i < instanceCount; i++) {
    const t = i / instanceCount;
    aRadius[i * 3] = 2.0 * Math.pow(t, 1.5) + 2.0;
    aRadius[i * 3 + 1] = 0.0;
    aRadius[i * 3 + 2] = 0.0;
  }
  instancedGeometry.setAttribute(
    "aRadius",
    new THREE.InstancedBufferAttribute(aRadius, 3, false)
  );

  const material = new THREE.ShaderMaterial({
    fragmentShader,
    vertexShader,
    uniforms: {
      time: { value: 0.0 },
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
      user_color: { value: new THREE.Vector3(1, 1, 1) },
    },
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
    blending: additive ? THREE.AdditiveBlending : THREE.SubtractiveBlending,
    depthWrite: false,
    transparent: true,
  });
  return new THREE.Mesh(new THREE.PlaneGeometry(10000, 10000), material);
}

function render({ model, el }) {
  const container = document.createElement("div");
  container.style.width = "100%";
  container.style.height = "480px";
  el.appendChild(container);

  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl", { antialias: true, alpha: true });
  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true });
  const width = el.clientWidth || 800;
  const height = 480;
  renderer.setSize(width, height);

  const bgColor = model.get("bg_color") || [0, 0, 0];
  renderer.setClearColor(new THREE.Color(bgColor[0], bgColor[1], bgColor[2]), 1);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  camera.position.z = 5;
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
    const count = model.get("n_concentric") || 100;
    if (nDims < 3) {
      scene.add(createSurface(shaderCode, additive));
    } else {
      scene.add(createSplatter(shaderCode, count, additive));
    }
  }

  updateOrbital();
  model.on("change:shader_code", updateOrbital);
  model.on("change:n_concentric", updateOrbital);

  renderer.setAnimationLoop(() => {
    for (const child of scene.children) {
      if (child.material && child.material.uniforms && child.material.uniforms.time) {
        child.material.uniforms.time.value += 0.01;
      }
    }
    controls.update();
    renderer.render(scene, camera);
  });

  return () => {
    renderer.setAnimationLoop(null);
    renderer.dispose();
    controls.dispose();
  };
}

export default { render };
