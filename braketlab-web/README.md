# Braketlab Web

Browser-native quantum mechanics with Dirac notation, running entirely in your browser via Pyodide and JupyterLite.

This is the modernized version of [Braketlab](https://github.com/audunsh/braketlab), rebuilt for standalone deployment on GitHub Pages. No server required: the Python algebra engine runs via Pyodide (Python compiled to WebAssembly), and orbitals are rendered in real-time using Three.js WebGL with Gaussian splatting-inspired billboard shading.

---

## Running Locally

### Prerequisites

- Python 3.10 or later
- Node.js 18 or later (for the renderer build)

### 1. Install Python dependencies

```bash
cd braketlab-web
pip install -e ".[dev]"
pip install jupyterlite-core jupyterlite-pyodide-kernel jupyter-server build
```

The first command installs `braketlab_core` and its dependencies (`sympy`, `numpy`, `scipy`, `anywidget`) in editable mode. The second installs the JupyterLite toolchain.

### 2. Run the Python test suite

```bash
pytest tests/ -v
```

All 43 tests should pass. To also run the original Braketlab tests:

```bash
cd ..
pytest test/ -v
```

### 3. Build the renderer (optional)

```bash
cd renderer
npm install
npm run build
```

This bundles the TypeScript renderer into `renderer/dist/widget.js`. For iterative development with automatic rebuilds:

```bash
npm run dev
```

### 4. Build the braketlab_core wheel

The JupyterLite site needs a `.whl` file so the Pyodide kernel can install `braketlab_core`:

```bash
cd braketlab-web
python -m build --wheel --outdir dist/
```

### 5. Build and serve the JupyterLite site

```bash
cd braketlab-web/site
jupyter lite build --force
jupyter lite serve
```

Then open `http://localhost:8000` in your browser.

The build configuration is in `site/jupyter_lite_config.json`. It:
- Bundles the notebooks from `../notebooks` as contents
- Bundles the `braketlab_core` wheel from `../dist/` so `%pip install braketlab-core` works inside Pyodide

The first build takes a minute or two (fetches Pyodide); subsequent builds are cached.

---

## Project Structure

```
braketlab-web/
├── braketlab_core/         # Pyodide-compatible algebra engine
│   ├── algebra.py          # basisfunction, ket, operator classes
│   ├── basis.py            # GTOs, STOs, hydrogen, harmonic oscillator
│   ├── integrals.py        # Obara-Saika analytical integrals + MC fallback
│   ├── codegen.py          # SymPy → GLSL/WGSL shader generation
│   └── viewer.py           # anywidget OrbitalView (Three.js renderer bridge)
├── renderer/               # TypeScript/Three.js orbital renderer
│   ├── src/
│   │   ├── widget.ts           # anywidget ESM entry point
│   │   ├── OrbitalRenderer.ts  # Scene, camera, animation loop
│   │   ├── GaussianSplatter.ts # Billboard splatting engine
│   │   └── VRManager.ts        # WebXR session management
│   └── package.json
├── notebooks/              # Tutorial notebooks
│   ├── 00_getting_started.ipynb
│   └── 01_basis_functions.ipynb
├── site/                   # GitHub Pages deployment config
│   ├── index.html
│   └── jupyter_lite_config.json
├── tests/                  # Test suite
│   ├── test_core.py
│   └── test_integrals.py
└── pyproject.toml
```

---

## Using braketlab_core in a Notebook

```python
import braketlab_core as bk
from braketlab_core import basis

# Create a p_z orbital
pz = basis.get_gto(1.0, 1, 0)

# Compute the self-overlap (should be 1.0)
print(pz.bra @ pz)

# Apply kinetic operator
T = bk.get_kinetic_operator()
print(pz.bra @ (T * pz))

# Visualize in 3D (requires anywidget + a notebook kernel)
pz.view()
```

The `view()` method returns an `OrbitalView` anywidget. It works in:
- JupyterLite (Pyodide, via GitHub Pages)
- JupyterLab / classic Jupyter
- Google Colab
- VS Code notebooks

---

## Deploying to GitHub Pages

Push to the `main` branch. The GitHub Actions workflow at `.github/workflows/deploy.yml` will:

1. Install Python and Node.js dependencies
2. Build the `braketlab_core` wheel
3. Build the renderer bundle
4. Run `jupyter lite build` with the tutorial notebooks
5. Deploy the site to GitHub Pages

Enable GitHub Pages in your repository settings (Settings → Pages → Source: GitHub Actions).

---

## Architecture Overview

```
SymPy expression (algebra.py)
       │
       ▼
  codegen.py
  sp.ccode() / sp.wgsl()
       │
       ▼
  GLSL/WGSL fragment shader string
       │  (synced via anywidget traitlets)
       ▼
  viewer.py (Python) ──────► widget.ts (JavaScript)
                                    │
                              Three.js scene
                              GaussianSplatter
                              (instanced spheres,
                               additive blending,
                               per-fragment wavefunction eval)
                                    │
                              WebXR / VR mode
```

The fragment shader evaluates the wavefunction at each pixel in real time on the GPU. Time-dependent wavefunctions (superpositions of eigenstates) use `cos(E_n * t)` / `sin(E_n * t)` terms driven by a `time` uniform incremented per frame.
