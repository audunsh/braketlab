// Billboard vertex shader for orbital splatting.
// Each instance is a camera-facing quad scaled by aRadius.
// tex coordinates are passed to the fragment shader for
// wavefunction evaluation.

uniform float time;
uniform vec2 resolution;

varying vec2 vUv;
varying vec3 pos;
varying vec3 tex;

attribute vec3 aRadius;

void main() {
    vUv = uv;

    // Scale billboard by radius, proportional to camera distance
    float cameraLen = length(cameraPosition);
    vec3 transformed = position * aRadius.x * cameraLen;

    // tex = world-space coordinates for wavefunction evaluation
    tex = transformed + 3.0 * cameraPosition;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(tex, 1.0);
}
