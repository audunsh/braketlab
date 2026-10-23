/**
 * anywidget-compatible entry point for the orbital renderer.
 *
 * This module exports the render() and initialize() functions expected
 * by the anywidget ESM protocol. When the Python OrbitalView widget
 * syncs its traitlets, this JS code creates/updates the Three.js
 * visualization.
 *
 * Protocol: https://anywidget.dev/en/afm/
 */

import { OrbitalRenderer, type RenderOptions } from "./OrbitalRenderer.js";

interface AnyWidgetModel {
  get(key: string): any;
  set(key: string, value: any): void;
  on(event: string, callback: (...args: any[]) => void): void;
  save_changes(): void;
}

interface AnyWidgetRenderContext {
  model: AnyWidgetModel;
  el: HTMLElement;
}

let renderer: OrbitalRenderer | null = null;

export function render({ model, el }: AnyWidgetRenderContext): () => void {
  const container = document.createElement("div");
  container.style.width = "100%";
  container.style.minHeight = "400px";
  el.appendChild(container);

  // Wait for container to be in DOM and have dimensions
  requestAnimationFrame(() => {
    const options: RenderOptions = {
      bgColor: model.get("bg_color") || [0, 0, 0],
      additive: model.get("additive") ?? true,
      vrEnabled: model.get("vr_enabled") ?? false,
      width: container.clientWidth || 800,
      height: (container.clientWidth || 800) * 0.6,
    };

    renderer = new OrbitalRenderer(container, options);
    updateOrbital(model);

    model.on("change:shader_code", () => updateOrbital(model));
    model.on("change:mode", () => updateOrbital(model));
    model.on("change:bg_color", () => {
      if (!renderer) return;
      const [r, g, b] = model.get("bg_color");
      renderer.renderer.setClearColor(new (globalThis as any).THREE.Color(r, g, b), 1);
    });
  });

  return () => {
    if (renderer) {
      renderer.dispose();
      renderer = null;
    }
  };
}

function updateOrbital(model: AnyWidgetModel): void {
  if (!renderer) return;
  const shaderCode = model.get("shader_code");
  if (!shaderCode) return;

  const nDims = model.get("n_dims") || 3;
  renderer.renderOrbital(shaderCode, nDims);
}
