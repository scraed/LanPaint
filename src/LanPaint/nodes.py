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

# MiniMaxH3 was ModelType.FLOW before ComfyUI 0.31.0 and ModelType.FLOW_AV
# afterwards (the FLOW_AV enum + ModelSamplingAV audio carriage, commit
# bdcb886a). Both are rectified-flow schedules; treat them alike.
FLOW_MODEL_TYPES = (ModelType.FLOW, getattr(ModelType, "FLOW_AV", None))
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
    scale_mode = "nearest-exact"

    # Handle input mask dimensions
    if video_inpainting:
        # video masks -> (1, 1, F, H, W): raw [F, H, W], or the [F, 1, H, W]
        # shape SetLatentNoiseMask produces (frames at batch); a plain
        # [H, W] image mask is broadcast over the whole video; 5D passes
        if input_mask.ndim == 3:
            input_mask = input_mask.unsqueeze(0).unsqueeze(1)
        elif input_mask.ndim == 4:
            input_mask = input_mask.permute(1, 0, 2, 3).unsqueeze(0)
        elif input_mask.ndim == 2:
            input_mask = input_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif input_mask.ndim == 1 and len(output_shape) == 4:
        # audio mask [F] at video frame rate -> the audio latent's time axis:
        # nearest-exact along time, then expanded to the (ch, T) layout
        t = output_shape[-1]
        input_mask = torch.nn.functional.interpolate(
            input_mask.float().unsqueeze(0).unsqueeze(0),
            size=(t,),
            mode="nearest-exact",
        )
        input_mask = input_mask.expand(1, 1, output_shape[-2], t)
    elif input_mask.ndim == 4 and len(output_shape) == 4 and input_mask.shape[1] == 1 and input_mask.shape[3] == 1:
        # audio mask [F, 1] that SetLatentNoiseMask reshaped to [1, 1, F, 1]:
        # nearest F -> T, then expanded to (1, 1, ch, T)
        t = output_shape[-1]
        input_mask = torch.nn.functional.interpolate(input_mask, size=(t, 1), mode="nearest-exact")
        input_mask = input_mask.permute(0, 1, 3, 2).expand(1, 1, output_shape[-2], t)
    elif input_mask.ndim == 2:
        input_mask = input_mask.unsqueeze(0).unsqueeze(0)
    elif input_mask.ndim == 3:
        input_mask = input_mask.unsqueeze(1)

    # Handle 5D output shape (B, C, F, H, W) by ensuring input is 5D
    if len(output_shape) == 5 and input_mask.ndim == 4:
        if COMFYUI_VERSION_060_OR_NEWER:
            input_mask = input_mask.unsqueeze(2)  # (B, C, 1, H, W)

    if video_inpainting:  # Video case: (batch, channels, frames, height, width)

        target_frames = output_shape[2]
        target_height, target_width = output_shape[-2:]

        # EXPERIMENTAL: nearest-exact first, then the slice-level union.
        # The resample snaps each latent slice to a single picked frame
        # (src = floor((t+0.5)*F/T), ~1 in 4 frames), then the sliding max
        # over ~5 slices spreads the value to neighboring slices. NOTE: this
        # drops strokes painted on frames that no slice picks.
        input_mask = torch.nn.functional.interpolate(
            input_mask,
            size=(target_frames, target_height, target_width),
            mode=scale_mode,
        )

        mask = torch.nn.functional.max_pool3d(
            input_mask, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0))
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
def min_step_frac_effective_steps(n_steps, frac, min_frac):
    """Inner-step count under the MinStepFrac tail ramp.

    While the remaining-noise fraction stays above min_frac (or with
    min_frac == 0, the feature disabled) the count is unchanged and the step
    size keeps scaling with the noise inside LanPaint. Below the fraction the
    step size is pinned and the count ramps down linearly,
    NSteps * frac / min_frac, to zero."""
    if min_frac <= 0 or frac >= min_frac or n_steps <= 0:
        return n_steps
    return max(0, round(n_steps * frac / min_frac))

def _sanitize_param(value, default, allowed=None):
    """Coerce a widget value to a sane default.

    Old workflows and hand-edited JSON can supply values that no longer fit
    the widget (wrong type, or a value removed from a combo list). Fail back
    to the default instead of crashing the node.
    """
    if allowed is not None:
        return value if value in allowed else default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value

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
        IS_FLOW = self.inner_model.inner_model.model_type in FLOW_MODEL_TYPES
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

            # Only one knob changes at a time: while the remaining noise
            # fraction (1 - abt) stays above LanPaint_MinStepFrac the inner
            # step size scales with it (LanPaint pins it below the fraction)
            # and the count is constant. Below the fraction the count ramps
            # down linearly, NSteps * (1 - abt) / MinStepFrac, down to 0.
            n_eff = self.PaintMethod.n_steps
            if total_steps - current_step <= self.LanPaint_early_stop:
                n_eff = 0
            else:
                n_eff = min_step_frac_effective_steps(
                    n_eff, float((1.0 - abt).mean()), getattr(self, "LanPaint_min_step_frac", 1.0))
            out = self.PaintMethod(x, self.latent_image, self.noise, sigma, latent_mask, current_times, model_options, seed, n_steps=n_eff, current_times_audio=current_times_audio, audio_indicator=self.audio_indicator, audio_correction=audio_correction)
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
        IS_FLOW = model_wrap.inner_model.model_type in FLOW_MODEL_TYPES
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
                                       EarlyStopHook = extra_args.get("model_options", {}).get("lanpaint_semantic_hook", None),
                                       MinStepFrac = getattr(model_wrap.model_patcher, "LanPaint_MinStepFrac", 1.0))
        model_k.LanPaint_early_stop = model_wrap.model_patcher.LanPaint_EarlyStop
        model_k.LanPaint_min_step_frac = getattr(model_wrap.model_patcher, "LanPaint_MinStepFrac", 1.0)
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

_override_active = False


@contextmanager
def override_sample_function():
    # guard against nested entry: a second entry must not capture the already
    # patched functions as "originals" (that would restore the patches, not
    # the real functions, on exit)
    global _override_active
    if _override_active:
        yield
        return
    _override_active = True
    try:
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

        yield
    finally:
        comfy.sampler_helpers.prepare_mask = original_prepare_mask
        comfy.samplers.KSAMPLER.sample = original_sample
        comfy.samplers.CFGGuider.predict_noise = original_predict_noise
        comfy.samplers.CFGGuider.outer_sample = original_outer_sample
        _override_active = False


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
                  },
            "hidden": {
                    # Retired hyperparameters. Old prompts (saved before these
                    # were removed) still pass them; accept and ignore the
                    # values -- the fixed defaults below are always used.
                    "LanPaint_MinStepFrac": "DEFAULT",
                  },
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, LanPaint_NumSteps=5, LanPaint_PromptMode="Image First",  LanPaint_Info="",Inpainting_mode="🖼️ Image Inpainting", **kwargs):
        LanPaint_NumSteps = _sanitize_param(LanPaint_NumSteps, 5)
        LanPaint_PromptMode = _sanitize_param(LanPaint_PromptMode, "Image First", allowed=("Image First", "Prompt First"))
        Inpainting_mode = _sanitize_param(Inpainting_mode, "🖼️ Image Inpainting", allowed=("🖼️ Image Inpainting", "🎬 Video Inpainting"))

        model.LanPaint_StepSize = 0.2
        model.LanPaint_Lambda = 5.0
        model.LanPaint_Beta = 1.
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_MinStepFrac = 1.0
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
                "LanPaint_Lambda": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1, "round": 0.1, "tooltip": "The bidirectional guidance scale. Higher values align with known regions more closely, but may result in instability."}),
                "LanPaint_StepSize": ("FLOAT", {"default": 0.2, "min": 0.0001, "max": 1., "step": 0.01, "round": 0.001, "tooltip": "The step size for the Langevin dynamics. Higher values result in faster convergence but may be unstable."}),
                "LanPaint_PromptMode": (["Image First", "Prompt First"], {"tooltip": "Image First: emphasis image quality, Prompt First: emphasis prompt following"}),
                "LanPaint_Info": ("STRING", {"default": "LanPaint KSampler Adv.", "tooltip": "For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!"}),
                "Inpainting_mode": (["🖼️ Image Inpainting", "🎬 Video Inpainting"], {"default": "🖼️ Image Inpainting", "tooltip": "Choose Image mode for photos or Video mode for video frames with temporal consistency"}),
                     },
            "hidden": {
                    # Retired hyperparameters. Old prompts (saved before these
                    # were removed) still pass them; accept and ignore the
                    # values -- the fixed defaults below are always used.
                    "LanPaint_Beta": "DEFAULT",
                    "LanPaint_Friction": "DEFAULT",
                    "LanPaint_EarlyStop": "DEFAULT",
                    "LanPaint_InnerThreshold": "DEFAULT",
                    "LanPaint_InnerPatience": "DEFAULT",
                    "LanPaint_MinStepFrac": "DEFAULT",
                  },
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, LanPaint_NumSteps=5, LanPaint_Lambda=5.0, LanPaint_StepSize=0.2, LanPaint_PromptMode="Image First", LanPaint_Info="", Inpainting_mode="🖼️ Image Inpainting", **kwargs):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        LanPaint_NumSteps = _sanitize_param(LanPaint_NumSteps, 5)
        LanPaint_Lambda = _sanitize_param(LanPaint_Lambda, 5.0)
        LanPaint_StepSize = _sanitize_param(LanPaint_StepSize, 0.2)
        LanPaint_PromptMode = _sanitize_param(LanPaint_PromptMode, "Image First", allowed=("Image First", "Prompt First"))
        Inpainting_mode = _sanitize_param(Inpainting_mode, "🖼️ Image Inpainting", allowed=("🖼️ Image Inpainting", "🎬 Video Inpainting"))
        model.LanPaint_StepSize = LanPaint_StepSize
        model.LanPaint_Lambda = LanPaint_Lambda
        model.LanPaint_Beta = 1.0
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_MinStepFrac = 1.0
        model.LanPaint_Friction = 15.0
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
        return gaussian_kernel_2d(kernel_size)

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
        LanPaint_NumSteps = _sanitize_param(LanPaint_NumSteps, 5)
        LanPaint_PromptMode = _sanitize_param(LanPaint_PromptMode, "Image First", allowed=("Image First", "Prompt First"))
        model.LanPaint_StepSize = 0.2
        model.LanPaint_Lambda = 5.0
        model.LanPaint_Beta = 1.
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_MinStepFrac = 1.0
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
                     "LanPaint_Lambda": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1, "tooltip": "Bidirectional guidance scale. Higher values align with known regions but may cause instability."}),
                     "LanPaint_StepSize": ("FLOAT", {"default": 0.2, "min": 0.0001, "max": 1.0, "step": 0.01, "tooltip": "Step size for Langevin dynamics. Higher values speed convergence but may be unstable."}),
                     "LanPaint_PromptMode": (["Image First", "Prompt First"], {"tooltip": "Image First: prioritizes image quality; Prompt First: prioritizes prompt adherence."}),
                     "LanPaint_Info": ("STRING", {"default": "LanPaint Custom Sampler Adv.", "tooltip": "For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!"}),
                    },
            "hidden": {
                    # Retired hyperparameters. Old prompts (saved before these
                    # were removed) still pass them; accept and ignore the
                    # values -- the fixed defaults below are always used.
                    "LanPaint_Beta": "DEFAULT",
                    "LanPaint_Friction": "DEFAULT",
                    "LanPaint_EarlyStop": "DEFAULT",
                    "LanPaint_InnerThreshold": "DEFAULT",
                    "LanPaint_InnerPatience": "DEFAULT",
                    "LanPaint_MinStepFrac": "DEFAULT",
                  },
               }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image, LanPaint_NumSteps, LanPaint_Lambda, LanPaint_StepSize, LanPaint_PromptMode, LanPaint_Info="", **kwargs):
        LanPaint_NumSteps = _sanitize_param(LanPaint_NumSteps, 5)
        LanPaint_Lambda = _sanitize_param(LanPaint_Lambda, 5.0)
        LanPaint_StepSize = _sanitize_param(LanPaint_StepSize, 0.2)
        LanPaint_PromptMode = _sanitize_param(LanPaint_PromptMode, "Image First", allowed=("Image First", "Prompt First"))
        model = guider.model_patcher
        model.LanPaint_StepSize = LanPaint_StepSize
        model.LanPaint_Lambda = LanPaint_Lambda
        model.LanPaint_Beta = 1.0
        model.LanPaint_NumSteps = LanPaint_NumSteps
        model.LanPaint_Friction = 15.0
        model.LanPaint_EarlyStop = 1
        model.LanPaint_InnerThreshold = 0.0
        model.LanPaint_InnerPatience = 1
        model.LanPaint_MinStepFrac = 1.0
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
    from comfy_api.latest._input_impl.video_types import VideoFromFile, VideoFromComponents
    from comfy_api.latest._util.video_types import VideoComponents
except Exception:  # comfy_api unavailable (stub/CI env): video output disabled
    VideoFromFile = None
    VideoFromComponents = None
    VideoComponents = None


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
    sampler resamples it to the audio latent's time axis.

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
                "video": (files, {"video_upload": True, "tooltip": "Source video file. Returned as the video output and used by the mask editor preview."}),
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


class LanPaint_AVEncode:
    """Encode a video's frames and audio into a nested AV latent, attaching
    the video and audio masks for inpainting.

    Replaces the workflow chain GetVideoComponents -> VAEEncode ->
    SetLatentNoiseMask + MiniMaxAudioEncode -> SetLatentNoiseMask ->
    Concat AV Latent with a single node. The video file's frames and audio
    track are decoded here (via the VIDEO reference), encoded with the two
    VAEs, and the masks ride per-stream inside the returned nested latent,
    so the sampler can inpaint video and audio from one mask source.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video (from the mask editor or LoadVideo). Frames and audio track are decoded and encoded."}),
                "vae": ("VAE", {"tooltip": "The video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "The audio VAE (e.g. the MiniMax H3 audio VAE)."}),
                "mask": ("MASK", {"tooltip": "Per-frame video mask [F, H, W] (1 = regenerate, 0 = keep)."}),
                "audio_mask": ("MASK", {"tooltip": "Audio mask [F] or [F, 1] at video frame rate (1 = regenerate that moment of the audio, 0 = keep)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "video"
    DESCRIPTION = "Encode a video's frames and audio into a nested AV latent with per-stream video and audio masks (1 = regenerate, 0 = keep)."

    def encode(self, video, vae, audio_vae, mask, audio_mask):
        components = video.get_components()
        frames = components.images  # [B, H, W, C] in 0..1
        z_video = vae.encode(frames[:, :, :, :3])

        audio = components.audio
        if audio is None:
            raise ValueError("the video has no audio track to encode")
        z_audio = LanPaint_MiniMaxAudioEncode().encode(audio, audio_vae)[0]["samples"]

        # audio mask: accept [F] or [F, 1]; the per-stream prep resamples it
        # to the audio latent's time axis
        if audio_mask.ndim == 2 and audio_mask.shape[1] == 1:
            audio_mask = audio_mask[:, 0]

        return ({
            "samples": comfy.nested_tensor.NestedTensor((z_video, z_audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((mask, audio_mask)),
        },)


def gaussian_kernel_2d(kernel_size):
    """2D Gaussian kernel, sigma = (size - 1) / 4 (same as MaskBlend)."""
    if kernel_size <= 1:
        return torch.ones(1, 1)  # no blending: identity kernel
    sigma = (kernel_size - 1) / 4
    x = torch.arange(kernel_size).float() - kernel_size // 2
    x_grid, y_grid = torch.meshgrid(x, x, indexing="ij")
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def merge_video_with_mask(orig, inpainted, mask, blend_overlap):
    """Blend the inpainted frames into the original with a MaskBlend-style
    boundary: dilate the mask (max_pool), smooth it (Gaussian), then
    ``orig * (1 - m) + inpainted * m``."""
    m = mask.float()
    if m.ndim == 4:  # [F, 1, H, W] from SetLatentNoiseMask
        m = m[:, 0]
    elif m.ndim == 2:  # single image mask [H, W]
        m = m.unsqueeze(0)
    count = min(orig.shape[0], inpainted.shape[0])
    orig = orig[:count]
    inpainted = inpainted[:count]
    if m.shape[0] == 1:  # one mask frame covers every frame (image case)
        m = m[:1].expand(count, -1, -1)
    elif m.shape[0] < count:
        raise ValueError("the mask has fewer frames than the image")
    else:
        m = m[:count]
    if tuple(m.shape[1:]) != tuple(orig.shape[1:3]):
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=orig.shape[1:3], mode="nearest-exact"
        )[:, 0]
    m = m.unsqueeze(1)
    m = torch.nn.functional.max_pool2d(
        m, kernel_size=blend_overlap, stride=1, padding=blend_overlap // 2
    )
    kernel = gaussian_kernel_2d(blend_overlap).to(orig.device)[None, None]
    m = torch.nn.functional.conv2d(m, kernel, padding=blend_overlap // 2)[:, 0]
    return orig * (1 - m[..., None]) + inpainted * m[..., None]


def merge_audio_with_mask(orig, inpainted, mask, crossfade, orig_sr, result_sr):
    """Blend the inpainted audio into the original inside the mask intervals.
    ``orig``/``inpainted`` are [B, C, L] waveforms (resampled to the original
    rate first); the per-frame mask is resampled to samples and a linear ramp
    of ``crossfade`` seconds is applied at each interval edge (0 = hard cut)."""
    if result_sr != orig_sr:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required to resample the inpainted audio")
        inpainted = torchaudio.functional.resample(inpainted, result_sr, orig_sr)
    if inpainted.shape[-1] != orig.shape[-1]:
        n = min(inpainted.shape[-1], orig.shape[-1])
        orig = orig[..., :n]
        inpainted = inpainted[..., :n]
    am = mask.float()
    if am.ndim == 4:  # [1, 1, F, 1] from SetLatentNoiseMask
        am = am[0, 0]
    elif am.ndim == 2 and am.shape[1] == 1:
        am = am[:, 0]
    n = orig.shape[-1]
    if am.shape[0] == n:
        w = am
    else:
        w = torch.nn.functional.interpolate(
            am.float().unsqueeze(0).unsqueeze(0), size=(n,), mode="nearest-exact"
        )[0, 0]
    if crossfade > 0 and orig_sr > 0:
        cf = max(1, int(round(crossfade * orig_sr)))
        kernel = torch.ones(1, 1, cf, device=orig.device) / cf
        # replicate padding: the box ramp applies only at true 0<->1
        # transitions, not at the mask's data ends (zero padding would bleed
        # a partial blend into the fully-masked tail). cf-1 total padding keeps
        # the output length exactly n (even kernels included).
        w_pad = torch.nn.functional.pad(
            w[None, None], (cf // 2, cf - 1 - cf // 2), mode="replicate"
        )
        w = torch.nn.functional.conv1d(w_pad, kernel)[0, 0][..., :n]
    if orig.shape[1] != inpainted.shape[1]:  # match channel counts before blending
        if orig.shape[1] == 1:
            orig = orig.expand(-1, inpainted.shape[1], -1)  # mono source: upmix
        elif inpainted.shape[1] == 1:
            inpainted = inpainted.expand(-1, orig.shape[1], -1)
        else:
            # surround original vs stereo inpainted: keep the original's first
            # channels (a downmix rather than a crash)
            orig = orig[:, : inpainted.shape[1]]
    return orig * (1 - w[None, None]) + inpainted * w[None, None]


class LanPaint_AVDecode:
    """Decode a nested AV latent and merge the inpainted video and audio with
    the original video, keeping the source's fps and bit depth.

    The result frames replace the original only inside the video mask
    (boundary blended like LanPaint_MaskBlend, ``blend_overlap`` pixels);
    the result audio replaces the original only inside the audio mask
    intervals (with a short crossfade at the edges). The decoded frames are
    resized to match the original dimensions exactly (VAE decodes can round
    the size). The output video carries the original file's frame rate and
    bit depth. A source without an audio track falls back to the inpainted
    audio.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The nested AV latent to decode (from the sampler)."}),
                "video": ("VIDEO", {"tooltip": "The original video (from the mask editor): source of the kept content, fps and bit depth."}),
                "vae": ("VAE", {"tooltip": "The video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "The audio VAE (e.g. the MiniMax H3 audio VAE)."}),
                "mask": ("MASK", {"tooltip": "Per-frame video mask [F, H, W] (1 = regenerate, 0 = keep)."}),
                "audio_mask": ("MASK", {"tooltip": "Audio mask [F] or [F, 1] at video frame rate (1 = regenerate that moment of the audio, 0 = keep)."}),
                "blend_overlap": ("INT", {"default": 11, "min": 1, "max": 51, "step": 2, "tooltip": "Boundary blend width in pixels between the inpainted and original video (MaskBlend-style)."}),
                "audio_crossfade": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.005, "tooltip": "Crossfade in seconds at audio interval edges (0 = hard cut)."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "AUDIO")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "decode"
    CATEGORY = "video"
    DESCRIPTION = "Decode the nested AV latent, merge the inpainted video and audio with the original (mask-blended), and write the result with the original fps and bit depth."

    def decode(self, samples, video, vae, audio_vae, mask, audio_mask, blend_overlap, audio_crossfade):
        if VideoFromComponents is None or VideoComponents is None:
            raise RuntimeError("the video output requires the ComfyUI runtime (comfy_api)")

        components = video.get_components()
        orig_frames = components.images  # [F, H, W, C] in 0..1
        orig_audio = components.audio  # {"waveform": [B, C, L], "sample_rate"} or None

        z_video, z_audio = samples["samples"].unbind()

        result_frames = vae.decode(z_video)  # [F, H, W, C] in 0..1
        if len(result_frames.shape) == 5:  # [1, F, H, W, C]: combine batches
            result_frames = result_frames.reshape(
                -1, result_frames.shape[-3], result_frames.shape[-2], result_frames.shape[-1]
            )
        result_audio = LanPaint_MiniMaxAudioDecode().decode(
            {"samples": z_audio}, audio_vae
        )[0]  # {"waveform": [B, C, L], "sample_rate": vae rate}

        # VAE decodes can round the size: match the original dimensions exactly
        target_h, target_w = orig_frames.shape[1], orig_frames.shape[2]
        if tuple(result_frames.shape[1:3]) != (target_h, target_w):
            result_frames = torch.nn.functional.interpolate(
                result_frames.movedim(-1, 1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)

        merged_frames = merge_video_with_mask(orig_frames, result_frames, mask, blend_overlap)

        merged_audio = result_audio
        if orig_audio is not None:
            merged_audio = merge_audio_with_mask(
                orig_audio["waveform"].float(),
                result_audio["waveform"].float(),
                audio_mask,
                audio_crossfade,
                orig_audio["sample_rate"],
                result_audio["sample_rate"],
            )
            merged_audio = {"waveform": merged_audio, "sample_rate": orig_audio["sample_rate"]}

        return (
            VideoFromComponents(
                VideoComponents(
                    images=merged_frames,
                    audio=merged_audio,
                    frame_rate=video.get_frame_rate(),
                ),
                bit_depth=video.get_bit_depth(),
            ),
            merged_audio,
        )


class LanPaint_ImageEncode:
    """Encode an image and attach the inpainting mask to the latent.

    Replaces the chain VAEEncode -> SetLatentNoiseMask (plus the manual
    ImageScale of the mask) with one node. The mask is snapped to the latent's
    actual spatial size (nearest-exact, the LanPaint mask convention), so
    whatever the VAE's padding does, the mask always matches. Works with
    image VAEs (4D latent) and video VAEs encoding a single image (5D latent
    [B, C, T, H, W], e.g. the Hunyuan video VAE): the mask is attached as
    [1, 1, T, H, W] and the sampler's prep handles it. Without a mask this is
    a plain encode.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to encode (1 = regenerate region comes from the mask)."}),
                "vae": ("VAE", {"tooltip": "The VAE."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Inpainting mask [H, W] (1 = regenerate, 0 = keep). Snapped to the latent size automatically."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "image"
    DESCRIPTION = "Encode an image and attach an inpainting mask to the latent (replaces VAEEncode + SetLatentNoiseMask)."

    def encode(self, image, vae, mask=None):
        z = vae.encode(image)
        ndim = len(z.shape)
        if ndim not in (4, 5):
            raise ValueError(
                f"LanPaint_ImageEncode expects a 4D or 5D latent, got {ndim}D"
            )
        latent = {"samples": z}
        if mask is not None:
            m = mask.float()
            if m.ndim == 4:  # [1, 1, H, W] from SetLatentNoiseMask
                m = m[0, 0]
            elif m.ndim == 3:
                m = m[0]
            if ndim == 4:
                target = z.shape[-2:]
                if tuple(m.shape) != target:
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(0).unsqueeze(0), size=target, mode="nearest-exact"
                    )[0, 0]
                latent["noise_mask"] = m.unsqueeze(0).unsqueeze(0)
            else:  # 5D video-style VAE: snap to (T, H, W), one mask slice per latent frame
                t, h, w = z.shape[-3:]
                if tuple(m.shape) != (h, w):
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest-exact"
                    )[0, 0]
                latent["noise_mask"] = m.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(1, 1, t, h, w)
        return (latent,)


class LanPaint_ImageDecode:
    """Decode an inpainted latent and merge it with the original image.

    Replaces the chain VAEDecode -> LanPaint_MaskBlend (plus the manual
    ImageScale alignment) with one node. The decoded image is resized to the
    original's exact dimensions (VAE decodes can round the size), then the
    inpainted pixels replace the original only inside the mask, with a
    MaskBlend-style boundary (``blend_overlap`` pixels). Without an image this
    is a plain decode.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The inpainted latent to decode."}),
                "vae": ("VAE", {"tooltip": "The VAE."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The original image. When given, the decoded output is resized to its exact dimensions."}),
                "mask": ("MASK", {"tooltip": "The inpainting mask (1 = take the inpainted pixels, 0 = keep the original)."}),
                "blend_overlap": ("INT", {"default": 9, "min": 1, "max": 51, "step": 2, "tooltip": "Boundary blend width in pixels between the inpainted and original image (MaskBlend-style)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "image"
    DESCRIPTION = "Decode an inpainted latent, resize to the original image's exact dimensions, and merge with the original inside the mask (replaces VAEDecode + MaskBlend)."

    def decode(self, samples, vae, image=None, mask=None, blend_overlap=9):
        z = samples["samples"]
        img = vae.decode(z)
        if len(img.shape) == 5:  # [1, F, H, W, C]: combine batches (video-style VAE)
            img = img.reshape(
                -1, img.shape[-3], img.shape[-2], img.shape[-1]
            )
        if image is None:
            return (img,)
        target_h, target_w = image.shape[1], image.shape[2]
        if tuple(img.shape[1:3]) != (target_h, target_w):
            img = torch.nn.functional.interpolate(
                img.movedim(-1, 1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
        if mask is None:
            return (img,)
        return (merge_video_with_mask(image, img, mask, blend_overlap),)


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
    "LanPaint_AVEncode": LanPaint_AVEncode,
    "LanPaint_AVDecode": LanPaint_AVDecode,
    "LanPaint_ImageEncode": LanPaint_ImageEncode,
    "LanPaint_ImageDecode": LanPaint_ImageDecode,
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
    "LanPaint_AVEncode": "LanPaint AV Encode",
    "LanPaint_AVDecode": "LanPaint AV Decode",
    "LanPaint_ImageEncode": "LanPaint Image Encode",
    "LanPaint_ImageDecode": "LanPaint Image Decode",
#    "LanPaint_UpSale_LatentNoiseMask": "LanPaint UpSale Latent Noise Mask"
}
