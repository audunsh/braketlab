/**
 * VRManager: WebXR session management for orbital visualization.
 *
 * Handles VR session lifecycle, controller input for orbital
 * manipulation (rotation, parameter adjustment), and hand tracking.
 */

import * as THREE from "three";
import { VRButton } from "three/examples/jsm/webxr/VRButton.js";

export class VRManager {
  private renderer: THREE.WebGLRenderer;
  private controllers: THREE.Group[] = [];
  private vrButton: HTMLElement | null = null;

  constructor(renderer: THREE.WebGLRenderer, container: HTMLElement) {
    this.renderer = renderer;
  }

  enable(): void {
    this.renderer.xr.enabled = true;
    this.vrButton = VRButton.createButton(this.renderer);
    this.renderer.domElement.parentElement?.appendChild(this.vrButton);

    const controller0 = this.renderer.xr.getController(0);
    const controller1 = this.renderer.xr.getController(1);

    controller0.addEventListener("selectstart", () => this.onSelectStart(0));
    controller0.addEventListener("selectend", () => this.onSelectEnd(0));
    controller1.addEventListener("selectstart", () => this.onSelectStart(1));
    controller1.addEventListener("selectend", () => this.onSelectEnd(1));

    this.controllers = [controller0, controller1];
  }

  getControllers(): THREE.Group[] {
    return this.controllers;
  }

  disable(): void {
    this.renderer.xr.enabled = false;
    if (this.vrButton) {
      this.vrButton.remove();
      this.vrButton = null;
    }
  }

  private onSelectStart(index: number): void {
    // Future: orbital manipulation via controllers
  }

  private onSelectEnd(index: number): void {
    // Future: release interaction
  }

  dispose(): void {
    this.disable();
    this.controllers = [];
  }
}
