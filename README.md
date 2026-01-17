<div align="center">

# LanPaint: Universal Inpainting Sampler with "Think Mode"
[![TMLR PDF](https://img.shields.io/badge/TMLR-PDF-8A2BE2?logo=openreview&logoColor=white)](https://openreview.net/pdf?id=JPC8JyOUSW)
[![Python Benchmark](https://img.shields.io/badge/🐍-Python_Benchmark-3776AB?logo=python)](https://github.com/scraed/LanPaintBench)
[![ComfyUI Extension](https://img.shields.io/badge/ComfyUI-Extension-7B5DFF)](https://github.com/comfyanonymous/ComfyUI)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/charrywhite/LanPaint)
[![Blog](https://img.shields.io/badge/📝-Blog-9cf)](https://scraed.github.io/scraedBlog/)
[![GitHub stars](https://img.shields.io/github/stars/scraed/LanPaint)](https://github.com/scraed/LanPaint/stargazers)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/yN5wYDE6W4)
</div>


Universally applicable inpainting ability for every model. LanPaint sampler lets the model "think" through multiple iterations before denoising, enabling you to invest more computation time for superior inpainting quality.  

This is the official implementation of ["LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling"](https://arxiv.org/abs/2502.03491), accepted by TMLR. The repository is for ComfyUI extension. Local Python benchmark code is published here: [LanPaintBench](https://github.com/scraed/LanPaintBench).

## Citation

```
@article{
zheng2025lanpaint,
title={LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling},
author={Candi Zheng and Yuan Lan and Yang Wang},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=JPC8JyOUSW},
note={}
}
```
**🎉 NEW 2026: Join our discord!**

[Join our Discord](https://discord.gg/aCGZutBV) to share experiences, discuss features, and explore future development.

**🎬 NEW: LanPaint now supports inpainting and outpainting based on Z-Image!**

| Original | Masked | Inpainted |
|:--------:|:------:|:---------:|
| ![Original Z-image](https://github.com/scraed/LanPaint/blob/master/examples/Example_21/Original_No_Mask.png) | ![Masked Z-image](https://github.com/scraed/LanPaint/blob/master/examples/Example_21/Masked_Load_Me_in_Loader.png) | ![Inpainted Z-image](https://github.com/scraed/LanPaint/blob/master/examples/Example_21/InPainted_Drag_Me_to_ComfyUI.png) |


**🎬 NEW: LanPaint now supports video inpainting and outpainting based on Wan 2.2!**

<div align="center">

| Original Video | Mask (edit T-shirt text) | Inpainted Result |
|:--------------:|:----:|:----------------:|
| ![Original](https://github.com/scraed/LanPaint/blob/master/examples/Original_No_Mask-example18.gif) | ![Mask](https://github.com/scraed/LanPaint/blob/master/examples/Example_18/Masked_Load_Me_in_Loader.png) | ![Result](https://github.com/scraed/LanPaint/blob/master/examples/Inpainted_81frames_Drag_Me_to_ComfyUI_example18.gif) |

*Video Inpainting Example: 81 frames with temporal consistency*

</div>

Check our latest [Wan 2.2 Video Examples](#video-examples-beta), [Wan 2.2 Image Examples](#example-wan22-inpaintlanpaint-k-sampler-5-steps-of-thinking), and 
[Qwen Image Edit 2509](#example-qwen-edit-2509-inpaint) support.
  

## Table of Contents
- [Features](#features)
- [Quickstart](#quickstart)
- [How to Use Examples](#how-to-use-examples)
- [Video Examples (Beta)](#video-examples-beta)
  - [Wan 2.2 Video Inpainting](#wan-22-video-inpainting)
  - [Wan 2.2 5B Video Inpainting](#wan-22-5b-video-inpainting)
  - [Wan 2.2 Video Outpainting](#wan-22-video-outpainting)
  - [Resource Consumption](#resource-consumption)
- [Image Examples](#image-examples)
  - [Flux.2.Dev](#example-flux2dev-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [Flux 2 klein](#example-flux-2-klein-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [Z-image](#example-z-image-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [Hunyuan T2I](#example-hunyuan-t2i-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [Wan 2.2 T2I](#example-wan22-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [Wan 2.2 T2I with reference](#example-wan22-partial-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [Qwen Image Edit 2511 2509](#example-qwen-edit-2509-inpaint)
  - [Qwen Image Edit 2508](#example-qwen-edit-2508-inpaint)
  - [Qwen Image](#example-qwen-image-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [HiDream](#example-hidream-inpaint-lanpaint-k-sampler-5-steps-of-thinking)
  - [SD 3.5](#example-sd-35-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [Flux](#example-flux-inpaintlanpaint-k-sampler-5-steps-of-thinking)
  - [SDXL](#example-sdxl-0-character-consistency-side-view-generation-lanpaint-k-sampler-5-steps-of-thinking)
- [Usage](#usage)
  - [Basic Sampler](#basic-sampler)
  - [Advanced Sampler](#lanpaint-ksampler-advanced)
  - [Tuning Guide](#lanpaint-ksampler-advanced-tuning-guide)
- [Community Showcase](#community-showcase-)
- [FAQ](#faq)
- [Updates](#updates)
- [ToDo](#todo)
- [Citation](#citation)

## Features

- **Universal Compatibility** – Works instantly with almost any model (**Z-image, Hunyuan, Wan 2.2, Qwen Image/Edit, HiDream, SD 3.5, Flux-series, SDXL, SD 1.5 or custom LoRAs**) and ControlNet.  
![Inpainting Result 13](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_13.jpg) 
- **No Training Needed** – Works out of the box with your existing model.  
- **Easy to Use** – Same workflow as standard ComfyUI KSampler.  
- **Flexible Masking** – Supports any mask shape, size, or position for inpainting/outpainting.  
- **No Workarounds** – Generates 100% new content (no blending or smoothing) without relying on partial denoising.  
- **Beyond Inpainting** – You can even use it as a simple way to generate consistent characters. 

**Warning**: LanPaint has degraded performance on distillation models, such as Flux.dev, due to a similar [issue with LORA training](https://medium.com/@zhiwangshi28/why-flux-lora-so-hard-to-train-and-how-to-overcome-it-a0c70bc59eaf). Please use low flux guidance (1.0-2.0) to mitigate this [issue](https://github.com/scraed/LanPaint/issues/30).

## Quickstart

1. **Install ComfyUI**: Follow the official [ComfyUI installation guide](https://docs.comfy.org/get_started) to set up ComfyUI on your system. Or ensure your ComfyUI version > 0.3.11.
2. **Install ComfyUI-Manager**: Add the [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) for easy extension management.  
3. **Install LanPaint Nodes**:  
   - **Via ComfyUI-Manager**: Search for "[LanPaint](https://registry.comfy.org/publishers/scraed/nodes/LanPaint)" in the manager and install it directly.  
   - **Manually**: Click "Install via Git URL" in ComfyUI-Manager and input the GitHub repository link:  
     ```
     https://github.com/scraed/LanPaint.git
     ```  
     Alternatively, clone this repository into the `ComfyUI/custom_nodes` folder.  
4. **Restart ComfyUI**: Restart ComfyUI to load the LanPaint nodes.  

Once installed, you'll find the LanPaint nodes under the "sampling" category in ComfyUI. Use them just like the default KSampler for high-quality inpainting!


## **How to Use Examples:**  
1. Navigate to the **example** folder (i.e example_1), download all pictures.  
2. Drag **InPainted_Drag_Me_to_ComfyUI.png** into ComfyUI to load the workflow.  
3. Download the required model (i.e clicking **Model Used in This Example**).  
4. Load the model in ComfyUI.
5. Upload **Masked_Load_Me_in_Loader.png** to the **"Load image"** node in the **"Mask image for inpainting"** group (second from left), or the **Prepare Image** node.  
7. Queue the task, you will get inpainted results from LanPaint. Some example also gives you inpainted results from the following methods for comparison:
   - **[VAE Encode for Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/)**
   - **[Set Latent Noise Mask](https://comfyui-wiki.com/en/tutorial/basic/how-to-inpaint-an-image-in-comfyui)**

## Video Examples (Beta)

LanPaint now supports video inpainting with Wan 2.2, enabling you to seamlessly inpaint masked regions across video frames while maintaining temporal consistency.

**Note:** LanPaint supports video inpainting for longer sequences (e.g., 81 frames), but processing time increases significantly (please check the [Resource Consumption](#resource-consumption) section for details) and performance may become unstable. For optimal results and stability, we recommend limiting video inpainting to **40 frames or fewer**.

### Wan 2.2 Video Inpainting 

*Example: Wan2.2 t2v 14B, 480p video (11:6), 40 frames, LanPaint K Sampler, 2 steps of thinking*

| Original Video | Mask (Add a white hat) | Inpainted Result |
|:--------------:|:----:|:----------------:|
| ![Original Video](https://github.com/scraed/LanPaint/blob/master/examples/Original_No_Mask_example17.gif) | ![Mask](https://github.com/scraed/LanPaint/blob/master/examples/Example_17/Masked_Load_Me_in_Loader.png) | ![Inpainted Result](https://github.com/scraed/LanPaint/blob/master/examples/Inpainted_40frames_Drag_Me_to_ComfyUI_example17.gif) |

[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_17)

You need to follow the ComfyUI version of [Wan2.2 T2V workflow](https://docs.comfy.org/tutorials/video/wan/wan2_2) to download and install the T2V model.

### Wan 2.2 5B Video Inpainting 

Similar to Wan 2.2 14B with slightly different workflow. [View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_17)

### Wan 2.2 Video Outpainting

Extend your videos beyond their original boundaries with LanPaint's video outpainting capability based on Wan 2.2. This feature allows you to expand the canvas of your videos while maintaining coherent motion and context.

*Example: Wan2.2 t2v 14B, 480p video (1:1 outpaint to 11:6), 40 frames, LanPaint K Sampler, 2 steps of thinking*

| Original Video | Mask (Expand to 880x480) | Outpainted Result |
|:--------------:|:----:|:-----------------:|
| ![Original Video](https://github.com/scraed/LanPaint/blob/master/examples/Original_Load_Me_in_Loader_example19.gif) | ![Mask](https://github.com/scraed/LanPaint/blob/master/examples/Mask_Example19_.png) | ![Outpainted Result](https://github.com/scraed/LanPaint/blob/master/examples/Outpainted_40frames_Drag_Me_to_ComfyUI_example19.gif) |

[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_19)

You need to follow the ComfyUI version of [Wan2.2 T2V workflow](https://docs.comfy.org/tutorials/video/wan/wan2_2) to download and install the T2V model.

### Resource Consumption


<table>
<thead>
<tr>
<th align="left">Processing Mode</th>
<th align="left">Resolution</th>
<th align="left">Frames Processed</th>
<th align="left">VRAM Required</th>
<th align="left">Total Runtime (20 steps)</th>
</tr>
</thead>
<tbody>
<tr style="background-color: #e8f4f8;">
<td><strong>Inpainting</strong></td>
<td>880×480 (11:6)</td>
<td>40 frames</td>
<td>39.8 GB</td>
<td><strong>05:37 min</strong></td>
</tr>
<tr style="background-color: #e8f4f8;">
<td><strong>Inpainting</strong></td>
<td>480×480 (1:1)</td>
<td>40 frames</td>
<td>38.0 GB</td>
<td><strong>05:35 min</strong></td>
</tr>
<tr style="background-color: #e8f4f8;">
<td><strong>Outpainting</strong></td>
<td>880×480 (11:6)</td>
<td>40 frames</td>
<td>40.2 GB</td>
<td><strong>05:36 min</strong></td>
</tr>
<tr style="background-color: #fff4e6;">
<td><strong>Inpainting</strong></td>
<td>880×480 (11:6)</td>
<td>81 frames</td>
<td>43.3 GB</td>
<td><strong>16:23 min</strong></td>
</tr>
<tr style="background-color: #fff4e6;">
<td><strong>Inpainting</strong></td>
<td>480×480 (1:1)</td>
<td>81 frames</td>
<td>39.8 GB</td>
<td><strong>14:25 min</strong></td>
</tr>
<tr style="background-color: #fff4e6;">
<td><strong>Outpainting</strong></td>
<td>880×480 (11:6)</td>
<td>81 frames</td>
<td>42.6 GB</td>
<td><strong>13:46 min</strong></td>
</tr>
</tbody>
</table>

<sub>**Test Platform**: All tests were conducted on an NVIDIA RTX Pro 6000.<br>
**Model Used**: `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` and `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`.<br>
**Processing Steps**:  20 sampling steps x 2 (LanPaint steps of thinking).</sub>

**Note:** Vram is required by the model, not LanPaint. To further reduce VRAM requirements, we recommend generating less frames and loading CLIP on CPU.

## Image Examples

### Example Hunyuan T2I: InPaint(LanPaint K Sampler, 5 steps of thinking)
We are excited to announce that LanPaint now supports inpainting with Hunyuan text to image generation.

[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_20)


You need to follow the ComfyUI version of [Hunyuan workflow](https://docs.comfy.org/tutorials/video/hunyuan-video#hunyuan-text-to-video-workflow) to download and install the model.

### Example Wan2.2: InPaint(LanPaint K Sampler, 5 steps of thinking)
We are excited to announce that LanPaint now supports Wan2.2 text to image generation with Wan2.2 T2V model.

![Inpainting Result 45](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_45.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_15)


You need to follow the ComfyUI version of [Wan2.2 T2V workflow](https://docs.comfy.org/tutorials/video/wan/wan2_2) to download and install the T2V model.

### Example Z-image: InPaint(LanPaint K Sampler, 5 steps of thinking)
LanPaint also supports inpainting with the Z-image text-to-image model.

<details open>
<summary>View Original / Masked / Inpainted Comparison</summary>

| Original | Masked | Inpainted |
|:--------:|:------:|:---------:|
| ![Original Z-image](https://github.com/scraed/LanPaint/blob/master/examples/Example_21/Original_No_Mask.png) | ![Masked Z-image](https://github.com/scraed/LanPaint/blob/master/examples/Example_21/Masked_Load_Me_in_Loader.png) | ![Inpainted Z-image](https://github.com/scraed/LanPaint/blob/master/examples/Example_21/InPainted_Drag_Me_to_ComfyUI.png) |

</details>

[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_21)

<details open>
<summary>View Z-image Outpainting (Original / Masked / Outpainted)</summary>

| Original | Masked | Outpainted |
|:--------:|:------:|:----------:|
| ![Original Z-image Outpaint](https://github.com/scraed/LanPaint/blob/master/examples/Example_22/Original_No_Mask.png) | ![Masked Z-image Outpaint](https://github.com/scraed/LanPaint/blob/master/examples/Example_22/Masked_Load_Me_in_Loader.png) | ![Outpainted Z-image](https://github.com/scraed/LanPaint/blob/master/examples/Example_22/InPainted_Drag_Me_to_ComfyUI.png) |

</details>

[View Outpaint Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_22)

You can download the Z-image model for ComfyUI from [Z-image](https://docs.comfy.org/zh-CN/tutorials/image/z-image/z-image-turbo).

### Example Wan2.2: Partial InPaint(LanPaint K Sampler, 5 steps of thinking)
Sometimes we don't want to inpaint completely new content, but rather let the inpainted image reference the original image. One option to achieve this is to inpaint with an edit model like Qwen Image Edit. Another option is to perform a partial inpaint: allowing the diffusion process to start at some middle steps rather than from 0.

![Inpainting Result 46](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_46.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_16)


You need to follow the ComfyUI version of [Wan2.2 T2V workflow](https://docs.comfy.org/tutorials/video/wan/wan2_2) to download and install the T2V model.


### Example Qwen Edit 2509: InPaint
Check our latest updated [Mased Qwen Edit Workflow](https://github.com/scraed/LanPaint/tree/master/examples/Example_14) for Qwen Image Edit 2509. Download the model at [Qwen Image Edit 2509 Comfy](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models). This workflow also supports Qwen Image Edit 2511.

![Qwen Result 3](https://github.com/scraed/LanPaint/blob/master/examples/LanPaintQwen_04.jpg) 

### Example Qwen Edit 2508: InPaint
![Qwen Result 2](https://github.com/scraed/LanPaint/blob/master/examples/LanPaintQwen_03.jpg) 
Check [Mased Qwen Edit Workflow](https://github.com/scraed/LanPaint/tree/master/examples/Example_14). You need to follow the ComfyUI version of [Qwen Image Edit workflow](https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit) to download and install the model.



### Example Qwen Image: InPaint(LanPaint K Sampler, 5 steps of thinking)

![Inpainting Result 14](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_14.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_11)


You need to follow the ComfyUI version of [Qwen Image workflow](https://docs.comfy.org/tutorials/image/qwen/qwen-image) to download and install the model.

The following examples utilize a random seed of 0 to generate a batch of 4 images for variance demonstration and fair comparison. (Note: Generating 4 images may exceed your GPU memory; please adjust the batch size as necessary.)

![Qwen Result 1](https://github.com/scraed/LanPaint/blob/master/examples/LanPaintQwen_01.jpg) 
Also check [Qwen Inpaint Workflow](https://github.com/scraed/LanPaint/tree/master/examples/Example_13) and [Qwen Outpaint Workflow](https://github.com/scraed/LanPaint/tree/master/examples/Example_12). You need to follow the ComfyUI version of [Qwen Image workflow](https://docs.comfy.org/tutorials/image/qwen/qwen-image) to download and install the model.

### Example HiDream: InPaint (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 8](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_11.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_8)

You need to follow the ComfyUI version of [HiDream workflow](https://docs.comfy.org/tutorials/image/hidream/hidream-i1) to download and install the model.

### Example HiDream: OutPaint(LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 8](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_13(1).jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_10)

You need to follow the ComfyUI version of [HiDream workflow](https://docs.comfy.org/tutorials/image/hidream/hidream-i1) to download and install the model. Thanks [Amazon90](https://github.com/Amazon90) for providing this example.

### Example SD 3.5: InPaint(LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 8](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_12.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_9)

You need to follow the ComfyUI version of [SD 3.5 workflow](https://comfyui-wiki.com/en/tutorial/advanced/stable-diffusion-3-5-comfyui-workflow) to download and install the model.

### Example Flux.2.Dev: InPaint(LanPaint K Sampler, 5 steps of thinking)

<details open>
<summary>View Original / Masked / Inpainted Comparison</summary>

| Original | Masked | Inpainted |
|:--------:|:------:|:---------:|
| ![Original Flux.2.Dev](https://github.com/scraed/LanPaint/blob/master/examples/Example_23/Original_No_Mask.png) | ![Masked Flux.2.Dev](https://github.com/scraed/LanPaint/blob/master/examples/Example_23/Masked_Load_Me_in_Loader.png) | ![Inpainted Flux.2.Dev](https://github.com/scraed/LanPaint/blob/master/examples/Example_23/InPainted_Drag_Me_to_ComfyUI.png) |

</details>

[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_23)

[Model Used in This Example](https://huggingface.co/Comfy-Org/flux2-dev)

(Note: Prompt First mode is disabled on Flux.2.Dev. As it does not use CFG guidance.)

### Example Flux 2 klein: InPaint(LanPaint K Sampler, 5 steps of thinking)

<details open>
<summary>View Original / Masked / Inpainted Comparison</summary>

| Original | Masked | Inpainted |
|:--------:|:------:|:---------:|
| ![Original Flux 2 klein](https://github.com/scraed/LanPaint/blob/master/examples/Example_24/Original_No_Mask.png) | ![Masked Flux 2 klein](https://github.com/scraed/LanPaint/blob/master/examples/Example_24/Masked_Load_Me_in_Loader.png) | ![Inpainted Flux 2 klein](https://github.com/scraed/LanPaint/blob/master/examples/Example_24/InPainted_Drag_Me_to_ComfyUI.png) |

</details>

[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_24)

[Model Used in This Example](https://docs.comfy.org/zh-CN/tutorials/flux/flux-2-klein)


### Example Flux: InPaint(LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 7](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_10.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_7)
[Model Used in This Example](https://huggingface.co/Comfy-Org/flux1-dev/blob/main/flux1-dev-fp8.safetensors) 
(Note: Prompt First mode is disabled on Flux. As it does not use CFG guidance.)

### Example SDXL 0: Character Consistency (Side View Generation) (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 6](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_09.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_6)
[Model Used in This Example](https://civitai.com/models/1188071?modelVersionId=1408658) 

(Tricks 1: You can emphasize the character by copy it's image multiple times with Photoshop. Here I have made one extra copy.)

(Tricks 2: Use prompts like multiple views, multiple angles, clone, turnaround. Use LanPaint's Prompt first mode (does not support Flux))

(Tricks 3: Remeber LanPaint can in-paint: Mask non-consistent regions and try again!)


### Example SDXL 1: Basket to Basket Ball (LanPaint K Sampler, 2 steps of thinking).
![Inpainting Result 1](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_04.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_1) 
[Model Used in This Example](https://civitai.com/models/1188071?modelVersionId=1408658) 
### Example SDXL 2: White Shirt to Blue Shirt (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 2](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_05.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_2)
[Model Used in This Example](https://civitai.com/models/1188071?modelVersionId=1408658)
### Example SDXL 3: Smile to Sad (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 3](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_06.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_3)
[Model Used in This Example](https://civitai.com/models/133005/juggernaut-xl)
### Example SDXL 4: Damage Restoration (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 4](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_07.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_4)
[Model Used in This Example](https://civitai.com/models/133005/juggernaut-xl)
### Example SDXL 5: Huge Damage Restoration (LanPaint K Sampler, 20 steps of thinking)
![Inpainting Result 5](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_08.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_5)
[Model Used in This Example](https://civitai.com/models/133005/juggernaut-xl)

Check more for use cases like inpaint on [fine tuned models](https://github.com/scraed/LanPaint/issues/12#issuecomment-2938662021) and [face swapping](https://github.com/scraed/LanPaint/issues/12#issuecomment-2938723501), thanks to [Amazon90](https://github.com/Amazon90).


## Usage

**Workflow Setup**  
Same as default ComfyUI KSampler - simply replace with LanPaint KSampler nodes. The inpainting workflow is the same as the [SetLatentNoiseMask](https://comfyui-wiki.com/zh/comfyui-nodes/latent/inpaint/set-latent-noise-mask) inpainting workflow.

**Note**
- LanPaint requires binary masks (values of 0 or 1) without opacity or smoothing. To ensure compatibility, set the mask's **opacity and hardness to maximum** in your mask editor. During inpainting, any mask with smoothing or gradients will automatically be converted to a binary mask.
- LanPaint relies heavily on your text prompts to guide inpainting - explicitly describe the content you want generated in the masked area. If results show artifacts or mismatched elements, counteract them with targeted negative prompts.

## Basic Sampler
![Samplers](https://github.com/scraed/LanPaint/blob/master/Nodes.JPG)  

- LanPaint KSampler: The most basic and easy to use sampler for inpainting.
- LanPaint KSampler (Advanced): Full control of all parameters.

### LanPaint KSampler
Simplified interface with recommended defaults:

- Steps: 20 - 50. More steps will give more "thinking" and better results.
- LanPaint NumSteps: The turns of thinking before denoising. Recommend 5 for most of tasks ( which means 5 times slower than sampling without thinking). Use 10 for more challenging tasks. 
- LanPaint Prompt mode: Image First mode and Prompt First mode. Image First mode focuses on the image, inpaint based on image context (maybe ignore prompt), while Prompt First mode focuses more on the prompt. Use Prompt First mode for tasks like character consistency. (Technically, it Prompt First mode change CFG scale to negative value in the BIG score to emphasis prompt, which will costs image quality.)

### LanPaint KSampler (Advanced)
Full parameter control:
**Key Parameters**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `Steps` | 0-100 | Total steps of diffusion sampling. Higher means better inpainting. Recommend 20-50. |
| `LanPaint_NumSteps` | 0-20 | Reasoning iterations per denoising step ("thinking depth"). Easy task: 2-5. Hard task: 5-10 |
| `LanPaint_Lambda` | 0.1-50 | Content alignment strength (higher = stricter). Recommend 4.0 - 10.0 |
| `LanPaint_StepSize` | 0.1-1.0 | The StepSize of each thinking step. Recommend 0.1-0.5. |
| `LanPaint_Beta` | 0.1-2.0 | The StepSize ratio between masked / unmasked region. Small value can compensate high lambda values. Recommend 1.0 |
| `LanPaint_Friction` | 0.0-100.0 | The friction of Langevin dynamics. Higher means more slow but stable, lower means fast but unstable. Recommend 10.0 - 20.0|
| `LanPaint_EarlyStop` | 0-10 | Stop LanPaint iteration before the final sampling step. Helps to remove artifacts in some cases. Recommend 1-5|
| `LanPaint_PromptMode` | Image First / Prompt First | Image First mode focuses on the image context, maybe ignore prompt. Prompt First mode focuses more on the prompt. |

For detailed descriptions of each parameter, simply hover your mouse over the corresponding input field to view tooltips with additional information.

### LanPaint Mask Blend
This node blends the original image with the inpainted image based on the mask. It is useful if you want the unmasked region to match the original image pixel perfectly.

## LanPaint KSampler (Advanced) Tuning Guide
For challenging inpainting tasks:  

1️⃣ **Boost Quality**
Increase **total number of sampling steps** (very important!), **LanPaint_NumSteps** (thinking iterations) or **LanPaint_Lambda** if the inpainted result does not meet your expectations.
  
2️⃣ **Boost Speed**
Decrease **LanPaint_NumSteps** to accelerate generation! If you want better results but still need fewer steps, consider:
    - **Increasing LanPaint_StepSize** to speed up the thinking process.
    - **Decreasing LanPaint_Friction** to make the Langevin dynamics converges more faster.
    
3️⃣ **Fix Unstability**:  
If you find the results have wired texture, try
- Reduce **LanPaint_Friction** to make the Langevin dynamics more stable. 
- Reduce **LanPaint_StepSize** to use smaller step size.
- Reduce **LanPaint_Beta** if you are using a high lambda value.

⚠️ **Notes**:  
- For effective tuning, **fix the seed** and adjust parameters incrementally while observing the results. This helps isolate the impact of each setting.  Better to do it with a batche of images to avoid overfitting on a single image.

## Community Showcase [](#community-showcase-)

Discover how the community is using LanPaint! Here are some user-created tutorials:

- [Ai绘画进阶148-三大王炸！庆祝高允贞出道6周年！T8即将直播？当AI绘画学会深度思考？！万能修复神器LanPaint，万物皆可修！-T8 Comfyui教程](https://www.youtube.com/watch?v=Z4DSTv3UPJo)
- [Ai绘画进阶151-真相了！T8竟是个AI？！LanPaint进阶（二），人物一致性，多视角实验性测试，新参数讲解，工作流分享-T8 Comfyui教程](https://www.youtube.com/watch?v=landiRhvF3k)
- [重绘和三视图角色一致性解决新方案！LanPaint节点尝试](https://www.youtube.com/watch?v=X0WbXdm6FA0)
- [ComfyUI: HiDream with Perturbation Upscale, LanPaint Inpainting (Workflow Tutorial)](https://www.youtube.com/watch?v=2-mGe4QVIIw&t=2785s)
- [ComfyUI必备LanPaint插件超详细使用教程](https://plugin.aix.ink/archives/lanpaint)

Submit a PR to add your tutorial/video here, or open an [Issue](https://github.com/scraed/LanPaint/issues) with details!

## FAQ
[Working togather with crop&stitch](https://github.com/scraed/LanPaint/issues/46)

## Updates
- 2025/08/08
    - Add Qwen image support
- 2025/06/21
    - Update the algorithm with enhanced stability and outpaint performance.
    - Add outpaint example
    - Supports Sampler Custom (Thanks to [MINENEMA](https://github.com/MINENEMA))
- 2025/06/04
    - Add more sampler support.
    - Add early stopping to advanced sampler.
- 2025/05/28
    - Major update on the Langevin solver. It is now much faster and more stable.
    - Greatly simplified the parameters for advanced sampler.
    - Fix performance issue on Flux and SD 3.5
- 2025/04/16
    - Added Primary HiDream support
- 2025/03/22
    - Added Primary Flux support
    - Added Tease Mode
- 2025/03/10
    - LanPaint has received a major update! All examples now use the LanPaint K Sampler, offering a simplified interface with enhanced performance and stability.
- 2025/03/06:
    - Bug Fix for str not callable error and unpack error. Big thanks to [jamesWalker55](https://github.com/jamesWalker55) and [EricBCoding](https://github.com/EricBCoding).

## ToDo
- Try Implement Detailer
- ~~Provide inference code on without GUI.~~ Check our local Python benchmark code [LanPaintBench](https://github.com/scraed/LanPaintBench).


## Citation

```
@article{
zheng2025lanpaint,
title={LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling},
author={Candi Zheng and Yuan Lan and Yang Wang},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=JPC8JyOUSW},
note={}
}
```





