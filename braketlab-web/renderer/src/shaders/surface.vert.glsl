// Surface vertex shader for 1D/2D orbital visualization.
// Maps a large plane to screen space; the fragment shader
// evaluates the wavefunction at each fragment.

uniform float time;
uniform vec2 resolution;

varying vec2 vUv;
varying vec3 pos;
varying vec3 tex;

void main() {
    vUv = uv;
    pos = position;
    tex = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
