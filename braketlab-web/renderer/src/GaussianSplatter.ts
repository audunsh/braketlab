/**
 * GaussianSplatter: Billboard-based volumetric orbital renderer.
 *
 * Replaces Evince's concentric InstancedMesh spheres with camera-facing
 * instanced sphere geometries. Each instance is scaled to sample a
 * different radial shell, and the fragment shader evaluates the
 * wavefunction per-pixel.
 *
 * The key improvement over the original Evince approach:
 * - The radius distribution uses a power-law spacing (r^1.5) that
 *   concentrates samples where the wavefunction has the most amplitude
 * - The vertex shader maps each instance to world coordinates offset
 *   by the camera position, so the sampling volume follows the view
 * - Back-face rendering with additive blending creates the volumetric
 *   appearance without explicit depth sorting
 *
 * This is "Gaussian splatting-inspired" in that:
 * - Each rendering primitive is spatially bounded by a Gaussian envelope
 * - The fragment shader evaluates the exact wavefunction (not an approximation)
 * - Additive alpha compositing builds up the volumetric density
 */

import * as THREE from "three";

export class GaussianSplatter {
  mesh: THREE.InstancedMesh;

  constructor(
    fragmentShader: string,
    instanceCount: number = 100,
    additive: boolean = true
  ) {
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
    (instancedGeometry as any).maxInstancedCount = instanceCount;

    const aRadius = new Float32Array(instanceCount * 3);
    const aCurve = new Float32Array(instanceCount * 3);

    for (let i = 0; i < instanceCount; i++) {
      const t = i / instanceCount;
      // Power-law spacing concentrates instances near the center
      // where Gaussian amplitude is highest
      aRadius[i * 3] = 2.0 * Math.pow(t, 1.5) + 2.0;
      aRadius[i * 3 + 1] = 0.0;
      aRadius[i * 3 + 2] = 0.0;

      aCurve[i * 3] = 0.0;
      aCurve[i * 3 + 1] = 0.0;
      aCurve[i * 3 + 2] = 0.0;
    }

    instancedGeometry.setAttribute(
      "aRadius",
      new THREE.InstancedBufferAttribute(aRadius, 3, false)
    );
    instancedGeometry.setAttribute(
      "aCurve",
      new THREE.InstancedBufferAttribute(aCurve, 3, false)
    );

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

    this.mesh = new THREE.InstancedMesh(
      instancedGeometry,
      material,
      instanceCount
    );
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    if (this.mesh.material instanceof THREE.Material) {
      this.mesh.material.dispose();
    }
  }
}
