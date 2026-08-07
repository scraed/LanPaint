# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

LanPaint is a ComfyUI extension that implements a training-free diffusion inpainting sampler based on Langevin dynamics ("think mode"). It lets any diffusion model iterate multiple times within each denoising step before committing to an output, improving inpainting quality without a specialized model.

## Commands

```bash
# Run all tests
pytest

# Lint
ruff check .

# Format
ruff format .

# Type check (requires mypy)
mypy
```

There is no build step — this is installed directly as a ComfyUI custom node by cloning into `custom_nodes/LanPaint`.

## Architecture

### Entry point and ComfyUI integration

`__init__.py` is the ComfyUI entry point. When ComfyUI loads this module, it imports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` from `src/LanPaint/nodes.py`. The `WEB_DIRECTORY = "./web"` tells ComfyUI where to find the frontend JS.

When imported **without** ComfyUI (e.g., in CI), `_install_lightweight_runtime_stubs()` creates dummy `torch`, `comfy`, `nodes`, and `comfyui_version` modules so `nodes.py` can still be imported for node discovery. The real stubs are in `src/LanPaint/types.py` (`LangevinState` NamedTuple).

### Core algorithm (`src/LanPaint/lanpaint.py`)

`LanPaint.__call__()` is the main algorithm. It runs `n_steps` Langevin dynamics sub-iterations within each outer denoising step:

1. Replaces the masked region with the noise-scaled known latent (`scale_latent_inpaint`)
2. In each inner step, computes a score function via `score_model()` — which calls the diffusion model to get `x_0` and `x_0_BIG` (high-CFG) predictions — then runs one Langevin sub-step
3. After iterations, denoises the result to produce the final `x_0` output

The `LanPaintEarlyStopper` (`src/LanPaint/earlystop.py`) can terminate inner iterations early based on semantic convergence or a custom distance function, contributed by `@godnight10061`.

### Monkey-patching mechanism (`src/LanPaint/nodes.py`)

`override_sample_function()` is a context manager that temporarily replaces three functions on ComfyUI's `comfy.samplers` module:
- `CFGGuider.outer_sample` → `CFGGuider_LanPaint.outer_sample` (handles mask preparation and WAN22 video models)
- `CFGGuider.predict_noise` → `CFGGuider_LanPaint.predict_noise` (dual CFG output — normal + BIG)
- `KSAMPLER.sample` → `KSAMPLER.sample` (injects `LanPaint` as the paint method for inpainting steps)

These are monkey-patches, not subclass overrides, because ComfyUI internally constructs `CFGGuider` and `KSAMPLER` instances directly. The monkey-patches are scoped to a single `nodes.common_ksampler()` call.

### Sampler nodes

There are four sampler nodes, two "basic" and two "advanced":

- **LanPaint_KSampler** / **LanPaint_KSamplerAdvanced** — use `nodes.common_ksampler()`. The advanced variant exposes all LanPaint hyperparameters (Lambda, StepSize, Beta, Friction, EarlyStop, InnerThreshold, InnerPatience).
- **LanPaint_SamplerCustom** / **LanPaint_SamplerCustomAdvanced** — use `comfy.sample.sample_custom()` / `guider.sample()`, for use with custom sigmas/samplers/guiders.

All sampler nodes attach LanPaint parameters to `model` (the model patcher object) and set `model_options["video_inpainting"]` for video mode.

Additional nodes:
- **LanPaint_MaskBlend** — blends before/after images with a Gaussian-smoothed mask for seamless boundaries
- **LanPaint_UpSale_LatentNoiseMask** — generates a checkerboard noise mask (currently commented out in `NODE_CLASS_MAPPINGS`)

### Numerical utilities (`src/LanPaint/utils.py`)

`StochasticHarmonicOscillator` simulates the Langevin dynamics step analytically. It computes the exact mean and covariance of the position/velocity after time `t` and samples from a multivariate normal. The module also contains numerically stable implementations of `(e^x - 1)/x`, `(e^x - 1 - x)/x^2`, hyperbolic functions, and helper coefficients (`zeta1`, `zeta2`, `Zcoefs`).

### Dual CFG

LanPaint uses two classifier-free guidance scales simultaneously:
- `cfg` — the standard CFG scale used for the known-region score
- `cfg_BIG` — a second (often higher) CFG scale used for the masked-region score via `score_model()`. In "Prompt First" mode, `cfg_BIG = 0*cfg - 0.5 = -0.5`, which effectively disables the second guidance.

### Frontend (`web/lanpaint_info.js`)

A ComfyUI extension that adds a "More Info, Bug Report, Star on GitHub" button to each LanPaint sampler node in the UI.

### Version compatibility

`COMFYUI_VERSION_060_OR_NEWER` gates behavior differences for mask reshaping between ComfyUI versions < 0.6.0 and >= 0.6.0, which changed the latent tensor dimension convention.

## Testing

Tests use pytest. The test suite is designed to run without ComfyUI installed — `conftest.py` adds the project root to `sys.path`, and `_install_lightweight_runtime_stubs()` provides dummy modules. The primary integration test (`test_package_imports_without_comfy`) validates that the package can be imported and node mappings are present.

CI uses `comfy-org/node-diff` to validate backwards compatibility of node interfaces on PRs.

## Running ComfyUI on this machine

### Installed locations

| What | Path | Version |
|---|---|---|
| User data (custom_nodes, models, output) | `E:\CompyUI` | — |
| ComfyUI Desktop app (Electron shell) | `C:\Users\scraed\AppData\Local\Programs\@comfyorgcomfyui-electron` | 0.21.1 bundled |
| **Actual ComfyUI (used by Desktop)** | `C:\Users\scraed\ComfyUI-Installs\ComfyUI\ComfyUI\` | **0.24.1** |
| Python venv | `E:\CompyUI\.venv` | Python 3.12.6 |
| Shared models | `C:\Users\scraed\ComfyUI-Shared\models\` | — |
| Extra models (Y drive) | `Y:\ComfyData\models\` | — |

### Launch headless

The ComfyUI at `ComfyUI-Installs` is the one to use — the bundled Electron version (0.21.1) is outdated and lacks nodes like `Ideogram4Scheduler`, `DualModelGuider`, `CFGOverride`.

```powershell
$env:PYTHONUTF8=1
$env:PYTHONIOENCODING='utf-8'
& E:\CompyUI\.venv\Scripts\python.exe `
  C:\Users\scraed\ComfyUI-Installs\ComfyUI\ComfyUI\main.py `
  --base-directory E:\CompyUI `
  --listen 127.0.0.1 --port 8188 `
  --disable-auto-launch
```

The `PYTHONUTF8` and `PYTHONIOENCODING` env vars are required on this machine (Chinese Windows, GBK codec chokes on emoji in custom node logs).

### Model paths

ComfyUI loads `extra_model_paths.yaml` from **the same directory as `main.py`**, not from `--base-directory`. The config at `C:\Users\scraed\ComfyUI-Installs\ComfyUI\ComfyUI\extra_model_paths.yaml` points to both the Shared and Y-drive model collections.

### GPU

Dual NVIDIA RTX A6000 (49GB VRAM each), PyTorch 2.9.1+cu130.

### Workflows

Saved workflows are at `E:\CompyUI\user\default\workflows\`. There are 74 workflows covering LanPaint inpainting, Qwen image edit, Flux, HunYuan, HiDream, Wan video, and Ideogram4 generation.

### Running a workflow — practical notes

**Workflow format:** LanPaint examples are PNGs with embedded workflow JSON (`PIL.Image.open(png).info['workflow']`). Save to `E:\CompyUI\user\default\workflows\` to load them from the web UI.

**Common errors and their fixes:**

| Error | Cause | Fix |
|---|---|---|
| Models missing | `extra_model_paths.yaml` not at the ComfyUI root (same dir as `main.py`) | Create one with `base_path` sections pointing to `C:\Users\scraed\ComfyUI-Shared\` and `Y:\ComfyData\` |
| Nodes missing | UI-only nodes (`MarkdownNote`, `PreviewAny`) | Usually safe to ignore |
| Missing image input | `LoadImage` node needs a mask | Upload `Masked_Load_Me_in_Loader.png` from the example folder |
| "Failed to convert input to FLOAT" | `widgets_values` array is too short for the current node signature — params were added/inserted since the workflow was created | Expand `widgets_values` to match the current number of non-linked inputs, using correct types (INT, FLOAT, STRING, COMBO) |

**Testing workflow changes:** The easiest way to verify a LanPaint workflow works is to load it in the headless ComfyUI web interface (Playwright or manually), upload the mask if needed, and click Run. Monitor the page title for `[N%]` progress or grep the server log for "Prompt executed".

**UTF-8 on Chinese Windows:** Always set `$env:PYTHONUTF8=1` and `$env:PYTHONIOENCODING='utf-8'` before launching — the GBK codec cannot handle emoji in custom node logs.

### Workflow JSON debugging

**Link format differs by context:** Top-level `links` are arrays `[id, from_node, from_slot, to_node, to_slot, "TYPE"]`. Subgraph links (in `definitions.subgraphs[].links`) are dicts `{"id": N, "origin_id": N, ...}`. Mixing formats causes silent failures.

**ComfyUI clears link fields on save:** After loading+saving a workflow in ComfyUI, `inputs[].link` and `outputs[].links` fields on nodes are often reset to `null`/`[]` — even though the correct links still exist in the `links` array. This makes the frontend render nodes as disconnected. After any ComfyUI save, audit and restore these fields.

**Subgraph instance ↔ definition name matching:** Instance inputs (on the subgraph node) must have `name` matching the definition `inputs[].name`. Mismatch causes "No link found in parent graph" errors.

**Virtual nodes:** Inside subgraphs, `-10` = input node, `-20` = output node. Links from `-10` use the slot matching the definition input index.

**Node ID uniqueness:** IDs must be unique across top-level AND all subgraph nodes combined. Duplicates cause silent connection failures.

**VAE dimension alignment:** Always use `VAEEncode → VAEDecode → GetImageSize` to derive target dimensions before the actual encode path that feeds the sampler. VAEs require input dimensions divisible by a model-specific factor (e.g., 8). The round-trip forces alignment and captures the clean dimensions for all downstream `ImageScale` nodes. Without this, the mask and latent may have mismatched dimensions in `SetLatentNoiseMask`, causing cryptic errors. Don't remove this pattern unless you fully understand the VAE's input constraints.

**Widget value ordering when replacing nodes:** When swapping a node (e.g., `KSampler` → `LanPaint_KSampler`), the old widget values array does not map 1:1 to the new node's `INPUT_TYPES`. Always clear the old array and set widget values to exactly match the new node's non-linked inputs — in the correct order, with the correct count. Appending new params to stale old values shifts everything and produces NaN in the UI.

### Workflow PNG metadata conventions

Example directories follow this pattern:
| File | Metadata |
|------|----------|
| `Masked_Load_Me_in_Loader.png` | Plain PNG, no metadata |
| `Original_No_Mask.png` | Plain PNG, no metadata |
| `InPainted_Drag_Me_to_ComfyUI.png` | Must have embedded `workflow` + `prompt` tEXt chunks (auto from SaveImage) |

**Strip metadata:** `img = Image.open(src); img.save(dst, 'PNG')` — Pillow drops tEXt chunks on re-save.

**example_workflows/:** Each workflow gets a `.json` + `.jpg` pair. The `.jpg` is a preview derived from the output PNG: `img.convert('RGB').save('name.jpg', 'JPEG', quality=95)`.

### Kill server safely

Match the ComfyUI install path to avoid killing other Python programs:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'ComfyUI-Installs.*main\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

### Playwright/Browser automation

- **Beforeunload dialog:** ComfyUI shows `系统可能不会保存您所做的更改` when navigating away from a modified workflow. Call `browser_handle_dialog(accept=true)` to dismiss.
- **File choosers:** LoadImage nodes with IMAGEUPLOAD widgets spawn file chooser modals on page load. Dismiss with `browser_file_upload(paths=[])` — there may be multiple.
- **IndexedDB cache:** After modifying a workflow JSON on disk, the frontend may load a cached version. Close the workflow tab and re-open it (hard refresh alone is not sufficient).
- **Run button:** Use `page.getByTestId('queue-button').click()` — more reliable than text matching.
- **Progress:** `page.evaluate('() => document.title')` — format `[N%][M%] Node`. Completed when title returns to `*WorkflowName - ComfyUI`.
- **LoadImage via JS:** `app.graph.getNodeById(id).widgets.find(w => w.name === 'image').callback('filename.png')` to set the image without file picker.

## Commit conventions

Do NOT include the `Co-Authored-By: Codex <noreply@anthropic.com>` trailer in commit messages. All commits should be attributed solely to the git user.
