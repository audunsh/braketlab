/**
 * OrbitalRenderer: Main renderer for quantum mechanical orbitals.
 *
 * Manages the Three.js scene, camera, controls, and delegates to
 * GaussianSplatter (3D) or SurfaceRenderer (1D/2D) based on
 * the dimensionality of the wavefunction.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { VRButton } from "three/examples/jsm/webxr/VRButton.js";
import { GaussianSplatter } from "./GaussianSplatter.js";

export interface RenderOptions {
  bgColor?: [number, number, number];
  additive?: boolean;
  vrEnabled?: boolean;
  width?: number;
  height?: number;
}

export class OrbitalRenderer {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  container: HTMLElement;
  private splatter: GaussianSplatter | null = null;
  private surfaceMesh: THREE.Mesh | null = null;
  private animationId: number = 0;
  private options: Required<RenderOptions>;

  constructor(container: HTMLElement, options: RenderOptions = {}) {
    this.container = container;
    this.options = {
      bgColor: options.bgColor ?? [0, 0, 0],
      additive: options.additive ?? true,
      vrEnabled: options.vrEnabled ?? false,
      width: options.width ?? (container.clientWidth || 800),
      height: options.height ?? ((container.clientWidth || 800) * 0.6),
    };

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      75,
      this.options.width / this.options.height,
      0.1,
      1000
    );
    this.camera.position.z = 5;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.options.width, this.options.height);
    const [r, g, b] = this.options.bgColor;
    this.renderer.setClearColor(new THREE.Color(r, g, b), 1);

    if (this.options.vrEnabled) {
      this.renderer.xr.enabled = true;
      container.appendChild(VRButton.createButton(this.renderer));
    }

    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);

    window.addEventListener("resize", () => this.onResize());
    this.animate();
  }

  /**
   * Render an orbital from a GLSL fragment shader string.
   */
  renderOrbital(fragmentShader: string, nDims: number, nConcentric: number = 100): void {
    this.clear();

    if (nDims < 3) {
      this.renderSurface(fragmentShader);
    } else {
      this.renderSplats(fragmentShader, nConcentric);
    }
  }

  private renderSurface(fragmentShader: string): void {
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
      },
      vertexShader,
      fragmentShader,
      side: THREE.DoubleSide,
      blending: this.options.additive
        ? THREE.AdditiveBlending
        : THREE.SubtractiveBlending,
      depthWrite: false,
    });

    const geometry = new THREE.PlaneGeometry(10000, 10000);
    this.surfaceMesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.surfaceMesh);
  }

  private renderSplats(fragmentShader: string, instanceCount: number): void {
    this.splatter = new GaussianSplatter(
      fragmentShader,
      instanceCount,
      this.options.additive
    );
    this.scene.add(this.splatter.mesh);
  }

  private animate = (): void => {
    this.renderer.setAnimationLoop(() => {
      for (const child of this.scene.children) {
        if (child instanceof THREE.Mesh || child instanceof THREE.InstancedMesh) {
          const mat = (child as THREE.Mesh).material;
          if (mat instanceof THREE.ShaderMaterial && mat.uniforms.time) {
            mat.uniforms.time.value += 0.01;
          }
        }
      }
      this.renderer.render(this.scene, this.camera);
    });
  };

  clear(): void {
    while (this.scene.children.length > 0) {
      const obj = this.scene.children[0];
      this.scene.remove(obj);
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose();
        if (obj.material instanceof THREE.Material) obj.material.dispose();
      }
    }
    this.splatter = null;
    this.surfaceMesh = null;
  }

  private onResize(): void {
    const w = this.container.clientWidth || this.options.width;
    const h = w * 0.6;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  dispose(): void {
    this.renderer.setAnimationLoop(null);
    this.clear();
    this.renderer.dispose();
    this.controls.dispose();
  }
}
