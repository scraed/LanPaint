import json
import os
from contextlib import contextmanager
import math
# import nodes.py
import comfy
import nodes
import latent_preview
import torch
from comfy.utils import repeat_to_batch_size
from comfy.samplers import *
from comfy.model_base import ModelType
from .lanpaint import LanPaint
from comfy.model_base import WAN22
import comfyui_version

try:
    import torchaudio
except ImportError:  # torchaudio may be absent in CI/stub environments
    torchaudio = None

try:
    from comfy.ldm.minimax.model import time_shift_sigma, time_shift_slope
except Exception:  # not present in stub/CI environments
    time_shift_sigma = None
    time_shift_slope = None


def _detect_minimax_h3_audio(model_patcher, model_options, latent_shapes):
    """Detect a MiniMax H3 AV pack: a nested (video, audio) latent whose audio
    stream rides on a shifted sigma schedule. The diffusion model's own
    sigma_shift_video/sigma_shift_audio attributes mark the H3 schedule
    machinery; every other model keeps the plain video-schedule dynamics.
    Returns (latent_shapes, shift_v, shift_a) or None."""
    if latent_shapes is None or len(latent_shapes) < 2:
        return None
    model = getattr(model_patcher, "model", None)
    diff_model = getattr(model, "diffusion_model", None)
    shift_v = getattr(diff_model, "sigma_shift_video", None)
    shift_a = getattr(diff_model, "sigma_shift_audio", None)
    if shift_v is None or shift_a is None:
        return None
    # MiniMaxH3SigmaShift overrides live in transformer_options
    topts = model_options.get("transformer_options", {}) if isinstance(model_options, dict) else {}
    shift_v = topts.get("minimax_h3_sigma_shift_video", shift_v)
    shift_a = topts.get("minimax_h3_sigma_shift_audio", shift_a)
    return (latent_shapes, float(shift_v), float(shift_a))

def _version_tuple(value):
    return tuple(int(part) if part.isdigit() else 0 for part in value.split("."))

COMFYUI_VERSION_060_OR_NEWER = _version_tuple(comfyui_version.__version__) >= (0, 6, 0)

def reshape_mask(input_mask, output_shape,video_inpainting=False):
    dims = len(output_shape) - 2
    print('output shape',output_shape)
    scale_mode = "nearest-exact"
    print('input mask',input_mask.shape,type(input_mask),torch.max(input_mask),torch.min(input_mask))
    print('target output_shape',output_shape)
    print('input_mask.ndim:', input_mask.ndim, 'output_shape len:', len(output_shape))

    # Handle input mask dimensions
    if video_inpainting and input_mask.ndim == 3:
        # per-frame video mask [F, H, W] -> (1, 1, F, H, W) (frames at dim 2)
        input_mask = input_mask.unsqueeze(0).unsqueeze(1)
    elif input_mask.ndim == 1 and len(output_shape) == 4:
        # audio mask [F] at video frame rate -> the audio latent's token grid:
        # nearest-exact along time, then expanded to the (ch=2, tokens) layout
        f, t = input_mask.shape[0], output_shape[-1]
        input_mask = torch.nn.functional.interpolate(
            input_mask.float().unsqueeze(0).unsqueeze(0),
            size=(t,),
            mode="nearest-exact",
        )
        input_mask = input_mask.expand(1, 1, output_shape[-2], t)
    elif input_mask.ndim == 2:
        input_mask = input_mask.unsqueeze(0).unsqueeze(0)
    elif input_mask.ndim == 3:
        input_mask = input_mask.unsqueeze(1)

    # Handle 5D output shape (B, C, F, H, W) by ensuring input is 5D
    if len(output_shape) == 5 and input_mask.ndim == 4:
        if COMFYUI_VERSION_060_OR_NEWER:
            input_mask = input_mask.unsqueeze(2)  # (B, C, 1, H, W)

    if video_inpainting:  # Video case: (batch, channels, frames, height, width)
        ## legacy comfy < 0.6.0: frames-at-batch [F, 1, H, W] -> (1, 1, F, H, W)
        if not COMFYUI_VERSION_060_OR_NEWER and input_mask.ndim == 4:
            input_mask = input_mask.permute(1, 0, 2, 3).unsqueeze(0)

        # Temporal union: a latent token covers ~4 video frames (the VAE's
        # nominal temporal stride), and a token-level mask is all-or-nothing
        # (binarized at 0.5 downstream). Averaging the frames into a token
        # (trilinear) can erase sparse temporal strokes entirely; instead the
        # token takes the UNION (max) of its ~4 frames - it regenerates iff
        # any of them is painted. The last window is partial.
        if input_mask.shape[2] > 4:
            f = input_mask.shape[2]
            n_win = (f + 3) // 4
            pooled = torch.zeros(
                (input_mask.shape[0], input_mask.shape[1], n_win, input_mask.shape[3], input_mask.shape[4]),
                dtype=input_mask.dtype, device=input_mask.device)
            for i in range(n_win):
                pooled[:, :, i] = input_mask[:, :, i * 4 : (i + 1) * 4].amax(dim=2)
            input_mask = pooled

        target_frames = output_shape[2]
        target_height, target_width = output_shape[-2:]

        # 3D nearest-exact interpolation: (batch, channels, frames, height, width) -> (batch, channels, target_frames, target_height, target_width)
        temp_mask = torch.nn.functional.interpolate(
            input_mask,
            size=(target_frames, target_height, target_width),
            mode=scale_mode,
        )

        # temp_mask is already 5D: (batch, channels, target_frames, target_height, target_width)
        mask = temp_mask
        print('after mask',mask.shape)
        # Handle channel dimension expansion if needed
        if mask.shape[1] < output_shape[1]:
            mask = mask.repeat(1, output_shape[1], 1, 1, 1)[:, :output_shape[1]]
        # Handle batch dimension
        mask = repeat_to_batch_size(mask, output_shape[0])
    else:  # Original 2D image case
        if not COMFYUI_VERSION_060_OR_NEWER:
            mask = torch.nn.functional.interpolate(input_mask, size=output_shape[-2:], mode=scale_mode)
        else:
            mask = torch.nn.functional.interpolate(input_mask, size=output_shape[2:], mode=scale_mode)
        if mask.shape[1] < output_shape[1]:
            mask = mask.repeat((1, output_shape[1]) + (1,) * dims)[:,:output_shape[1]]
        mask = repeat_to_batch_size(mask, output_shape[0])


    return mask
def prepare_mask(noise_mask, shape, device,video_inpainting=False):
    return reshape_mask(noise_mask, shape,video_inpainting).to(device)
def sampling_function_LanPaint(model, x, timestep, uncond, cond, cond_scale, cond_scale_BIG, model_options={}, seed=None):
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out  = fn(args)

    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_), cfg_function(model, out[0], out[1], cond_scale_BIG, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)


class CFGGuider_LanPaint:
    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, **kwargs):
        print("CFGGuider outer_sample")
        self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        if isinstance(self.inner_model, WAN22):
            print("WAN22 detected")
            self.inner_model.extra_conds = super(WAN22, self.inner_model).extra_conds

        # MiniMax H3 AV packs carry an audio stream on a shifted sigma schedule;
        # the paint loop needs the per-stream times only for that exact model
        self.minimax_h3_audio = _detect_minimax_h3_audio(
            self.model_patcher, self.model_options, kwargs.get("latent_shapes", None))

        if denoise_mask is not None:
            video_inpainting = self.model_options.get("video_inpainting", False)
            if tuple(denoise_mask.shape) != tuple(noise.shape):
                # mask arrives already prepared to the latent shape (per-stream for
                # nested AV latents, packed flat); only re-reshape when it differs
                denoise_mask = prepare_mask(denoise_mask, noise.shape, device, video_inpainting)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, **kwargs)
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function_LanPaint(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, self.cfg_BIG, model_options=model_options, seed=seed)

#CFGGuider.outer_sample = CFGGuider_LanPaint.outer_sample
#CFGGuider.predict_noise = CFGGuider_LanPaint.predict_noise

class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
        self.audio_indicator = None  # [1, 1, N] flat pack: 1 on audio rows (MiniMax H3 only)
        self.audio_shifts = None  # (shift_video, shift_audio) for the audio sigma schedule
        #self.model_sigmas = torch.cat( (torch.tensor([0.], device = sigmas.device) , torch.tensor( self.inner_model.model_patcher.get_model_object("model_sampling").sigmas, device = sigmas.device) ) )
        #self.model_sigmas = torch.tensor( self.model_sigmas, dtype = self.sigmas.dtype )
    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None,**kwargs):
        ### For 1.5 and XL model
        # x is x_t in the notation of variance exploding diffusion model, x_t = x_0 + sigma * noise
        # sigma is the noise level
        ### For flux model 
        # x is rectified flow x_t = sigma * noise + (1.0 - sigma) * x_0

        IS_FLUX = self.inner_model.inner_model.model_type == ModelType.FLUX
        IS_FLOW = self.inner_model.inner_model.model_type == ModelType.FLOW
        #print("model class", type(self.inner_model.inner_model))
        #print("model type", self.inner_model.inner_model.model_type, "IS_FLUX", IS_FLUX, "IS_FLOW", IS_FLOW)
        #print("sigma", torch.mean(sigma).item(), torch.min(sigma).item(), torch.max(sigma).item())
        # unify the notations into variance exploding diffusion model
        if IS_FLUX or IS_FLOW:
            Flow_t = sigma
            abt = (1 - Flow_t)**2 / ((1 - Flow_t)**2 + Flow_t**2 )
            VE_Sigma = Flow_t / (1 - Flow_t)
            #print("t", torch.mean( sigma ).item(), "VE_Sigma", torch.mean( VE_Sigma ).item())


        else:
            VE_Sigma = sigma
            abt = 1/( 1+VE_Sigma**2 )
            Flow_t = (1-abt)**0.5 / ( (1-abt)**0.5 + abt**0.5  )

        # MiniMax H3 audio stream runs on its own shifted sigma schedule:
        # sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio).
        # The paint loop needs these per-stream times so the audio rows evolve
        # at their own noise level instead of the video's.
        current_times_audio = None
        audio_correction = None
        if self.audio_indicator is not None and self.audio_shifts is not None and time_shift_sigma is not None:
            shift_v, shift_a = self.audio_shifts
            Flow_a = time_shift_sigma(Flow_t, shift_v, shift_a)
            abt_a = (1 - Flow_a)**2 / ((1 - Flow_a)**2 + Flow_a**2)
            VE_a = Flow_a / (1 - Flow_a)
            current_times_audio = (VE_a, abt_a, Flow_a)
            # The flat-grid model output for the audio rows is the slope-scaled
            # velocity estimate, which overshoots the true denoised audio. The
            # Langevin target correction: x0_true = x + c*(x0_flat - x) with
            # c = sigma_a / (sigma_v * slope_a) on audio rows, 1 on video rows.
            ft = float(Flow_t)
            c = 1.0
            if ft > 1e-4 and time_shift_slope is not None:
                slope_a = time_shift_slope(Flow_t, shift_v, shift_a)
                c = float(Flow_a) / (ft * float(slope_a))
            audio_correction = (1.0 - self.audio_indicator) + c * self.audio_indicator

        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})

            denoise_mask = (denoise_mask > 0.5).float()

            latent_mask = 1 - denoise_mask
            current_times = (VE_Sigma, abt, Flow_t)

            current_step = torch.argmin( torch.abs( self.sigmas - torch.mean(sigma) ) )
            total_steps = len(self.sigmas)-1

            if total_steps - current_step <= self.LanPaint_early_stop:
                out = self.PaintMethod(x, self.latent_image, self.noise, sigma, latent_mask, current_times, model_options, seed, n_steps=0, current_times_audio=current_times_audio, audio_indicator=self.audio_indicator, audio_correction=audio_correction)
            else:
                out = self.PaintMethod(x, self.latent_image, self.noise, sigma, latent_mask, current_times, model_options, seed, current_times_audio=current_times_audio, audio_indicator=self.audio_indicator, audio_correction=audio_correction)
        else:
            out, _ = self.inner_model(x, sigma, model_options=model_options, seed=seed)

        # Add TAESD preview support - directly use the latent_preview module
        current_step = model_options.get("i", kwargs.get("i", 0))
        total_steps = model_options.get("total_steps", 0)

        # Only show preview every few steps to improve performance
        if current_step % 2 == 0:
            # Directly call the preview callback if it exists
            callback = model_options.get("callback", None)
            if callback is not None:
                callback({"i": current_step, "denoised": out, "x": x})

        return out

# Custom sampler class extending ComfyUI's KSAMPLER for LanPaint
class KSAMPLER(comfy.samplers.KSAMPLER):
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        #noise here is a randn noise from comfy.sample.prepare_noise
        #latent_image is the latent image as input of the KSampler node. For inpainting, it is the masked latent image. Otherwise it is zero tensor.
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        IS_FLUX = model_wrap.inner_model.model_type == ModelType.FLUX
        IS_FLOW = model_wrap.inner_model.model_type == ModelType.FLOW
        # unify the notations into variance exploding diffusion model
        if IS_FLUX:
            model_wrap.cfg_BIG = 1.0
        else:
            model_wrap.cfg_BIG = model_wrap.model_patcher.LanPaint_cfg_BIG
        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        # MiniMax H3 AV pack: mark the audio rows of the flat pack so the paint
        # loop can apply the audio's own shifted sigma schedule to them
        audio_layout = getattr(model_wrap, "minimax_h3_audio", None)
        if audio_layout is not None and time_shift_sigma is not None:
            latent_shapes, shift_v, shift_a = audio_layout
            video_n = math.prod(latent_shapes[0][1:])
            audio_indicator = torch.zeros(noise.shape, dtype=torch.float32, device=noise.device)
            audio_indicator[..., video_n:] = 1.0
            model_k.audio_indicator = audio_indicator
            model_k.audio_shifts = (shift_v, shift_a)

        model_k.PaintMethod = LanPaint(model_k.inner_model,
                                       model_wrap.model_patcher.LanPaint_NumSteps,
                                       model_wrap.model_patcher.LanPaint_Friction,
                                       model_wrap.model_patcher.LanPaint_Lambda,
                                       model_wrap.model_patcher.LanPaint_Beta,
                                       model_wrap.model_patcher.LanPaint_StepSize, 
                                       IS_FLUX = IS_FLUX, 
                                       IS_FLOW = IS_FLOW,
                                       EarlyStopThreshold = getattr(model_wrap.model_patcher, "LanPaint_InnerThreshold", 0.0),
                                       EarlyStopPatience = getattr(model_wrap.model_patcher, "LanPaint_InnerPatience", 1),
                                       EarlyStopHook = extra_args.get("model_options", {}).get("lanpaint_semantic_hook", None))
        model_k.LanPaint_early_stop = model_wrap.model_patcher.LanPaint_EarlyStop
        #if not inpainting, after noise_scaling, noise = noise * sigma, which is the noise added to the clean latent image in the variance exploding diffusion model notation.
        #if inpainting, after noise_scaling, noise = latent_image + noise * sigma, which is x_t in the variance exploding diffusion model notation for the known region.
        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)
        #print("LanPaint KSampler call sampler_function", self.sampler_function)
        # The main loop!
        #print("##########")
        #print("Sampling with ", self.sampler_function)
        #print("##########")
        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        #print("LanPaint KSampler end sampler_function")
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples

@contextmanager
def override_sample_function():
    original_outer_sample = comfy.samplers.CFGGuider.outer_sample
    comfy.samplers.CFGGuider.outer_sample = CFGGuider_LanPaint.outer_sample

    original_predict_noise = comfy.samplers.CFGGuider.predict_noise
    comfy.samplers.CFGGuider.predict_noise = CFGGuider_LanPaint.predict_noise

    original_sample = comfy.samplers.KSAMPLER.sample
    comfy.samplers.KSAMPLER.sample = KSAMPLER.sample

    # Route the stock per-stream mask prep (inside CFGGuider.sample) through
    # our prepare_mask so video masks get the temporal-union aggregation
    # (window-4 max + nearest-exact) from reshape_mask. 5D targets (video)
    # use the union path; everything else keeps the existing behavior.
    original_prepare_mask = comfy.sampler_helpers.prepare_mask

    def _prepare_mask_with_union(noise_mask, shape, device):
        return prepare_mask(noise_mask, shape, device, video_inpainting=(len(shape) == 5))

    comfy.sampler_helpers.prepare_mask = _prepare_mask_with_union

    try:
        yield
    finally:
        comfy.sampler_helpers.prepare_mask = original_prepare_mask
        comfy.samplers.KSAMPLER.sample = original_sample
        comfy.samplers.CFGGuider.predict_noise = original_predict_noise
        comfy.samplers.CFGGuider.outer_sample = original_outer_sample


class LanPaint_UpSale_LatentNoiseMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "scale": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "set_mask"


    CATEGORY = "latent/inpaint"

    def set_mask(self, samples, scale):
        s = samples.copy()
        samples = s['samples']
        # generate a mask with every scaleth pixel set to 1
        mask = torch.zeros(samples.shape[0], 1, samples.shape[2], samples.shape[3], device=samples.device) + 1
        mask[:, :, ::scale, ::scale] = 0
        s["noise_mask"] = mask
        return (s,)

#KSAMPLER_NAMES = ["euler", "dpmpp_2m", "uni_pc"]
KSAMPLER_NAMES = ["euler","euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                "dpm_fast",  "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", 
                   "deis", "res_multistep", "res_multistep_ancestral", 
                  "gradient_estimation",  "er_sde", "seeds_2", "seeds_3"]

class LanPaint_KSampler():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (KSAMPLER_NAMES, {"tooltip": "Recommended: euler."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 100, "tooltip": "The number of steps for the Langevin dynamics, representing the turns of thinking per step."}),
                "LanPaint_PromptMode": (["Image First", "Prompt First"], {"tooltip": "Image First: emphasis image quality, Prompt First: emphasis prompt following"}),
                "LanPaint_Info": ("STRING", {"default": "LanPaint KSampler.", "tooltip": "For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!"}),
                "Inpainting_mode": (["🖼️ Image Inpainting", "🎬 Video Inpainting"], {"default": "🖼️ Image Inpainting", "tooltip": "Choose Image mode for photos or Video mode for video frames with temporal consistency"}),
                  }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, LanPaint_NumSteps=5, LanPaint_PromptMode="Image First",  LanPaint_Info="",Inpainting_mode="🖼️ Image Inpainting"):

        model.LanPaint_StepSize = 0.2
        model.LanPaint_Lambda = 16.0
        model.LanPaint_Beta = 1.
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_Friction = 15.
        model.LanPaint_EarlyStop = 1
        model.LanPaint_InnerThreshold = 0.0
        model.LanPaint_InnerPatience = 1
        if LanPaint_PromptMode == "Image First":
            model.LanPaint_cfg_BIG = cfg
        else:
            model.LanPaint_cfg_BIG = 0*cfg - 0.5

        # Convert inpainting_mode to boolean for video_inpainting
        video_inpainting = (Inpainting_mode == "🎬 Video Inpainting")
        if not hasattr(model, 'model_options') or model.model_options is None:
            model.model_options = {}
        model.model_options["video_inpainting"] = video_inpainting

        with override_sample_function():
            return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
class LanPaint_KSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (KSAMPLER_NAMES, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 100, "tooltip": "The number of steps for the Langevin dynamics, representing the turns of thinking per step."}),
                "LanPaint_Lambda": ("FLOAT", {"default": 16., "min": 0.1, "max": 50.0, "step": 0.1, "round": 0.1, "tooltip": "The bidirectional guidance scale. Higher values align with known regions more closely, but may result in instability."}),
                "LanPaint_StepSize": ("FLOAT", {"default": 0.2, "min": 0.0001, "max": 1., "step": 0.01, "round": 0.001, "tooltip": "The step size for the Langevin dynamics. Higher values result in faster convergence but may be unstable."}),
                "LanPaint_Beta": ("FLOAT", {"default": 1., "min": 0.0001, "max": 5, "step": 0.1, "round": 0.1, "tooltip": "The step size ratio between masked / unmasked regions. Lower value can compensate high values of LanPaint_Lambda."}),
                "LanPaint_Friction": ("FLOAT", {"default": 15, "min": 0., "max": 50.0, "step": 0.1, "round": 0.1, "tooltip": "The friction parameter for fast langevin, lower values result in faster convergence but may be unstable."}),
                "LanPaint_PromptMode": (["Image First", "Prompt First"], {"tooltip": "Image First: emphasis image quality, Prompt First: emphasis prompt following"}),
                "LanPaint_EarlyStop": ("INT", {"default": 1, "min": 0, "max": 10000, "tooltip": "The number of steps to stop the LanPaint early, useful for preventing the image from irregular patterns."}),
                "LanPaint_Info": ("STRING", {"default": "LanPaint KSampler Adv.", "tooltip": "For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!"}),
                "Inpainting_mode": (["🖼️ Image Inpainting", "🎬 Video Inpainting"], {"default": "🖼️ Image Inpainting", "tooltip": "Choose Image mode for photos or Video mode for video frames with temporal consistency"}),
                "LanPaint_InnerThreshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001, "round": 0.0001, "tooltip": "Early stop threshold for Langevin iterations based on semantic distance. 0.0 to disable. (Contributed by godnight10061)"}),
                "LanPaint_InnerPatience": ("INT", {"default": 1, "min": 1, "max": 100, "tooltip": "Number of consecutive steps below threshold required to stop. (Contributed by godnight10061)"}),
                     },
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, LanPaint_NumSteps=5, LanPaint_Lambda=16.0, LanPaint_StepSize=0.2, LanPaint_Beta=1.0, LanPaint_Friction=15.0, LanPaint_PromptMode="Image First", LanPaint_EarlyStop=1, LanPaint_Info="", Inpainting_mode="🖼️ Image Inpainting", LanPaint_InnerThreshold=0.0, LanPaint_InnerPatience=1):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        model.LanPaint_StepSize = LanPaint_StepSize
        model.LanPaint_Lambda = LanPaint_Lambda
        model.LanPaint_Beta = LanPaint_Beta
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_Friction = LanPaint_Friction
        model.LanPaint_EarlyStop = LanPaint_EarlyStop
        model.LanPaint_InnerThreshold = LanPaint_InnerThreshold
        model.LanPaint_InnerPatience = LanPaint_InnerPatience
        if LanPaint_PromptMode == "Image First":
            model.LanPaint_cfg_BIG = cfg
        else:
            model.LanPaint_cfg_BIG = 0*cfg - 0.5

        # Convert inpainting_mode to boolean for video_inpainting
        video_inpainting = (Inpainting_mode == "🎬 Video Inpainting")
        if not hasattr(model, 'model_options') or model.model_options is None:
            model.model_options = {}
        model.model_options["video_inpainting"] = video_inpainting

        with override_sample_function():
            return nodes.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


class MaskBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE", {"tooltip": "Image before inpaint"}),
                "image2": ("IMAGE", {"tooltip": "Image after inpaint"}),
                "mask": ("MASK",),
                "blend_overlap": ("INT", {"default": 1, "min": 1, "max": 51, "step": 2, "tooltip": "The number of pixels to blend between the two images."})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "image/postprocessing"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, mask: torch.Tensor, blend_overlap: int):
        # smooth the binary 01 mask, keep 1 still 1, but smooth the transition from 1 to 0
        # for each mask pixel, find out the nearest 1 pixel, and set the mask value to the distance between the two pixels
        # check the size of mask and image1, image2, if not the same, assert error
        if image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
            raise ValueError(
                "Image size mismatch: Image1 and Image2 must have the same dimensions.\n"
                "Additionally, ensure both images have width and height that are multiples of 8.\n"
                "This is required because VAE decode always generates images with dimensions that are multiples of 8.\n"
                "If your input images are not multiples of 8, a size mismatch will occur during the decoding process.\n"
                "Please resize your images using an image resize node to ensure compatibility.\n"
                "Current sizes - Image1: {}x{}, Image2: {}x{}".format(
                    image1.shape[2], image1.shape[1], image2.shape[2], image2.shape[1]
                )
            )
        mask = mask.float()
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=blend_overlap, stride=1, padding=blend_overlap//2)
        # apply Gaussian blur with kernel size blend_overlap
        kernel = self.gaussian_kernel(blend_overlap)
        kernel = kernel.to(image1.device)
        kernel = kernel[None, None, ...]

        mask = torch.nn.functional.conv2d(mask[:,None,:,:], kernel, padding=blend_overlap//2)[:,0,:,:]


        blended_image = image1 * (1 - mask[...,None]) + image2 * mask[...,None]
        return (blended_image,)
    def gaussian_kernel(self,kernel_size):
        """
        Creates a 2D Gaussian kernel with the given size and standard deviation (sigma).
        """
        sigma = (kernel_size - 1)/4
        # Create a grid of (x, y) coordinates
        x = torch.arange(kernel_size).float() - kernel_size // 2
        y = torch.arange(kernel_size).float() - kernel_size // 2
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')

        # Compute the Gaussian function
        kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # Normalize the kernel

        return kernel

class Noise_EmptyNoise:
    def generate_noise(self, latent):
        return torch.zeros_like(latent["samples"])

class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed
    def generate_noise(self, latent):
        torch.manual_seed(self.seed)
        return torch.randn_like(latent["samples"])

# Custom sampler implementation mimmicking base comfy nodes_custom_sampler.py
class LanPaint_SamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "add_noise": ("BOOLEAN", {"default": True}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "sampler": ("SAMPLER",),
                     "sigmas": ("SIGMAS",),
                     "latent_image": ("LATENT",),
                     "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 100, "tooltip": "Number of steps for Langevin dynamics, representing turns of thinking per step."}),
                     "LanPaint_PromptMode": (["Image First", "Prompt First"], {"tooltip": "Image First: prioritizes image quality; Prompt First: prioritizes prompt adherence."}),
                     "LanPaint_Info": ("STRING", {"default": "LanPaint Custom Sampler.", "tooltip": "For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!"}),
                      }
               }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, sampler, sigmas, add_noise, noise_seed, cfg, positive, negative, latent_image, LanPaint_NumSteps, LanPaint_PromptMode, LanPaint_Info=""):
        model.LanPaint_StepSize = 0.2
        model.LanPaint_Lambda = 16.0
        model.LanPaint_Beta = 1.
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_Friction = 15.
        model.LanPaint_EarlyStop = 1
        model.LanPaint_InnerThreshold = 0.0
        model.LanPaint_InnerPatience = 1
        if LanPaint_PromptMode == "Image First":
            model.LanPaint_cfg_BIG = cfg
        else:
            model.LanPaint_cfg_BIG = 0 * cfg - 0.5
        with override_sample_function():
            latent = latent_image.copy()
            latent_image = latent["samples"]
            latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
            latent["samples"] = latent_image

            if not add_noise:
                noise = Noise_EmptyNoise().generate_noise(latent)
            else:
                noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            x0_output = {}
            callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
            return (out, out_denoised)

class LanPaint_SamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise": ("NOISE",),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     "LanPaint_NumSteps": ("INT", {"default": 5, "min": 0, "max": 100, "tooltip": "Number of steps for Langevin dynamics, representing turns of thinking per step."}),
                     "LanPaint_Lambda": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 50.0, "step": 0.1, "tooltip": "Bidirectional guidance scale. Higher values align with known regions but may cause instability."}),
                     "LanPaint_StepSize": ("FLOAT", {"default": 0.2, "min": 0.0001, "max": 1.0, "step": 0.01, "tooltip": "Step size for Langevin dynamics. Higher values speed convergence but may be unstable."}),
                     "LanPaint_Beta": ("FLOAT", {"default": 1.0, "min": 0.0001, "max": 5.0, "step": 0.1, "tooltip": "Step size ratio between masked/unmasked regions. Lower values balance high Lambda."}),
                     "LanPaint_Friction": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 50.0, "step": 0.1, "tooltip": "Friction parameter for fast Langevin. Lower values speed convergence but may be unstable."}),
                     "LanPaint_PromptMode": (["Image First", "Prompt First"], {"tooltip": "Image First: prioritizes image quality; Prompt First: prioritizes prompt adherence."}),
                     "LanPaint_EarlyStop": ("INT", {"default": 1, "min": 0, "max": 10000, "tooltip": "Steps to stop LanPaint early, preventing irregular patterns."}),
                     "LanPaint_Info": ("STRING", {"default": "LanPaint Custom Sampler Adv.", "tooltip": "For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!"}),
                     "LanPaint_InnerThreshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001, "round": 0.0001, "tooltip": "Early stop threshold for Langevin iterations based on semantic distance. 0.0 to disable. (Contributed by godnight10061)"}),
                     "LanPaint_InnerPatience": ("INT", {"default": 1, "min": 1, "max": 100, "tooltip": "Number of consecutive steps below threshold required to stop. (Contributed by godnight10061)"}),
                    }
               }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image, LanPaint_NumSteps, LanPaint_Lambda, LanPaint_StepSize, LanPaint_Beta, LanPaint_Friction, LanPaint_PromptMode, LanPaint_EarlyStop, LanPaint_Info="", LanPaint_InnerThreshold=0.0, LanPaint_InnerPatience=1):
        model = guider.model_patcher
        model.LanPaint_StepSize = LanPaint_StepSize
        model.LanPaint_Lambda = LanPaint_Lambda
        model.LanPaint_Beta = LanPaint_Beta
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_Friction = LanPaint_Friction
        model.LanPaint_EarlyStop = LanPaint_EarlyStop
        model.LanPaint_InnerThreshold = LanPaint_InnerThreshold
        model.LanPaint_InnerPatience = LanPaint_InnerPatience
        if LanPaint_PromptMode == "Image First":
            model.LanPaint_cfg_BIG = guider.cfg
        else:
            model.LanPaint_cfg_BIG = 0 * guider.cfg - 0.5
        with override_sample_function():
            latent = latent_image
            latent_image = latent["samples"]
            latent = latent.copy()
            latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
            latent["samples"] = latent_image

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            x0_output = {}
            callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
            samples = samples.to(comfy.model_management.intermediate_device())

            out = latent.copy()
            out["samples"] = samples
            if "x0" in x0_output:
                out_denoised = latent.copy()
                out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
            else:
                out_denoised = out
            return (out, out_denoised)


class LanPaint_MiniMaxAudioEncode:
    """MiniMax H3 audio encode.

    The sd.VAE wrapper expects channels-last audio ([B, L, C]) and converts
    internally, so the waveform is transposed exactly like ComfyUI's generic
    VAEEncodeAudio. This node adds mono->stereo upmixing on top. Audio
    inpainting masks are no longer handled here: the video mask editor
    produces the audio mask, which the workflow attaches to this latent with
    SetLatentNoiseMask.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio track to encode (resampled to the VAE's rate if needed)."}),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "audio"
    DESCRIPTION = "Encode audio with the MiniMax H3 audio VAE (correct layout). Audio inpainting masks come from the video mask editor via SetLatentNoiseMask."

    def encode(self, audio, vae):
        waveform = audio["waveform"]  # [B, C, L]
        sample_rate = audio["sample_rate"]
        vae_sr = getattr(vae, "audio_sample_rate", 32000)
        if vae_sr != sample_rate:
            if torchaudio is None:
                raise RuntimeError("torchaudio is required to resample audio for the MiniMax H3 audio VAE")
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sr)
        if waveform.shape[1] == 1:
            waveform = waveform.expand(-1, 2, -1)  # mono -> stereo

        # the sd.VAE wrapper takes channels-last audio and converts internally
        z = vae.encode(waveform.movedim(1, -1))  # [B, 32, 2, T]
        return ({"samples": z},)


class LanPaint_MiniMaxAudioDecode:
    """MiniMax H3 audio decode (wrapper returns channels-last [B, L, C]; this
    node converts back to the ComfyUI [B, C, L] audio convention)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Audio latent, or a nested AV latent (the audio stream is decoded)."}),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "audio"
    DESCRIPTION = "Decode a MiniMax H3 audio latent to a waveform."

    def decode(self, samples, vae):
        z = samples["samples"]
        if getattr(z, "is_nested", False):
            z = z.unbind()[-1]  # nested AV latent: take the audio stream
        waveform = vae.decode(z).movedim(-1, 1)  # wrapper [B, L, C] -> ComfyUI [B, C, L]
        sample_rate = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 32000))
        return ({"waveform": waveform, "sample_rate": sample_rate},)


try:
    from comfy_api.latest._input_impl.video_types import VideoFromFile
except Exception:  # comfy_api unavailable (stub/CI env): video output disabled
    VideoFromFile = None


class LanPaint_VideoMaskEditor:
    """Paint per-frame video masks and audio intervals on a video.

    Works like a simple LoadVideo node: it returns the selected video file as
    a VIDEO reference (no decoding) plus two masks. The video mask is a
    per-frame [F, H, W] tensor painted in the mask editor: the frontend
    ("Edit Video Mask" button) streams the same file and lets you paint masks
    on keyframes; frames between keyframes get the SDF-morphed mask. Painted
    keyframes are uploaded to the input folder as PNGs and their filenames are
    recorded in the hidden ``keyframes`` widget as {"<frame_idx>": "<file>.png"}.

    The audio mask is a [F] hard 0/1 tensor at video frame rate (1 = regenerate
    that moment of the audio, 0 = keep), built from time intervals in seconds
    recorded in the hidden ``audio_mask`` widget as [{"start": s, "end": e}].
    The same editor displays the audio waveform and paints intervals; the
    workflow attaches the mask to the audio latent (SetLatentNoiseMask) and the
    sampler resamples it to the audio latent tokens.

    Video mask convention: 1 = regenerate, 0 = keep. Both masks span the
    video's full frame count (read from the file container, no pixel decoding).
    """

    @classmethod
    def INPUT_TYPES(s):
        try:
            import folder_paths
            import os as _os

            input_dir = folder_paths.get_input_directory()
            files = sorted(
                f
                for f in _os.listdir(input_dir)
                if _os.path.isfile(_os.path.join(input_dir, f))
                and f.lower().endswith(
                    (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".gif")
                )
            )
        except Exception:  # folder_paths unavailable (stub env): no options yet
            files = []
        return {
            "required": {
                "video": (files, {"image_upload": True, "tooltip": "Source video file. Returned as the video output and used by the mask editor preview."}),
                "keyframes": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Hidden: keyframe mask files {\"frame\": \"file.png\"}. Written by the mask editor."}),
                "audio_mask": ("STRING", {"default": "[]", "multiline": True, "tooltip": "Hidden: audio inpainting intervals [{\"start\": s, \"end\": e}] in seconds. Written by the mask editor."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "MASK", "MASK")
    RETURN_NAMES = ("video", "mask", "audio_mask")
    FUNCTION = "run"
    CATEGORY = "video"
    DESCRIPTION = "Loads a video (like LoadVideo) and also outputs a per-frame video inpainting mask and a per-frame audio mask painted in the mask editor (1 = regenerate, 0 = keep)."

    def run(self, video=None, keyframes="{}", audio_mask="[]"):
        import folder_paths
        from .videomask import interpolate_masks, load_keyframe_png, parse_keyframes_widget, resize_masks

        if not video:
            raise ValueError("select a video file in the node first")
        if VideoFromFile is None:
            raise RuntimeError("the video output requires the ComfyUI runtime (comfy_api)")
        path = folder_paths.get_annotated_filepath(video)
        vf = VideoFromFile(path)
        count = int(vf.get_frame_count())
        w, h = vf.get_dimensions()  # (width, height)

        data = parse_keyframes_widget(keyframes)
        loaded = {}
        for idx, filename in data.items():
            try:
                path = folder_paths.get_annotated_filepath(filename)
                if path and os.path.isfile(path):
                    loaded[idx] = load_keyframe_png(path)
            except Exception:
                continue
        if loaded:
            seq = interpolate_masks(loaded, count)
            seq = resize_masks(seq, (w, h))
            out = torch.from_numpy(seq).float()
        else:
            # no keyframes (or all files missing): the mask stays empty
            out = torch.zeros(count, h, w, dtype=torch.float32)

        # audio mask: [F] hard 0/1 at video frame rate from intervals in seconds
        # (get_fps was renamed get_frame_rate, returning a Fraction, in newer ComfyUI)
        get_fps = getattr(vf, "get_fps", None)
        fps = float(get_fps() if get_fps else vf.get_frame_rate())
        audio_out = torch.zeros(count, dtype=torch.float32)
        try:
            intervals = json.loads(audio_mask) if isinstance(audio_mask, str) else audio_mask
        except Exception:
            intervals = []
        if isinstance(intervals, list):
            for it in intervals:
                try:
                    start = float(it.get("start", 0.0))
                    end = float(it.get("end", 0.0))
                except Exception:
                    continue
                if end > start:
                    f0 = max(0, int(math.floor(start * fps)))
                    f1 = min(count, int(math.ceil(end * fps)))
                    if f1 > f0:
                        audio_out[f0:f1] = 1.0

        return (vf, out, audio_out)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LanPaint_KSampler": LanPaint_KSampler,
    "LanPaint_KSamplerAdvanced": LanPaint_KSamplerAdvanced,
    "LanPaint_SamplerCustom" : LanPaint_SamplerCustom,
    "LanPaint_SamplerCustomAdvanced" : LanPaint_SamplerCustomAdvanced,
    "LanPaint_MaskBlend": MaskBlend,
    "LanPaint_MiniMaxAudioEncode": LanPaint_MiniMaxAudioEncode,
    "LanPaint_MiniMaxAudioDecode": LanPaint_MiniMaxAudioDecode,
    "LanPaint_VideoMaskEditor": LanPaint_VideoMaskEditor,
#    "LanPaint_UpSale_LatentNoiseMask": LanPaint_UpSale_LatentNoiseMask,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LanPaint_KSampler": "LanPaint KSampler",
    "LanPaint_KSamplerAdvanced": "LanPaint KSampler (Advanced)",
    "LanPaint_SamplerCustom" : "LanPaint Sampler Custom",
    "LanPaint_SamplerCustomAdvanced" : "LanPaint Sampler Custom (Advanced)",
    "LanPaint_MaskBlend": "LanPaint Mask Blend",
    "LanPaint_MiniMaxAudioEncode": "LanPaint MiniMax Audio Encode",
    "LanPaint_MiniMaxAudioDecode": "LanPaint MiniMax Audio Decode",
    "LanPaint_VideoMaskEditor": "LanPaint Video Mask Editor",
#    "LanPaint_UpSale_LatentNoiseMask": "LanPaint UpSale Latent Noise Mask"
}
